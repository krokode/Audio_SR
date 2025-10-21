import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Subpixel1D(nn.Module):
    """
    1D subpixel shuffling (reverse of conv that packs time into channels).
    Input shape: (B, C*r, T)
    Output shape: (B, C, T*r)
    """
    def __init__(self, r):
        super().__init__()
        self.r = r

    def forward(self, x):
        batch, c_mul_r, t = x.size()
        if c_mul_r % self.r != 0:
            raise ValueError("Channel dimension must be divisible by upscale factor r")
        c = c_mul_r // self.r
        x = x.view(batch, c, self.r, t)       # (B, C, r, T)
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, C, T, r)
        return x.view(batch, c, t * self.r)    # (B, C, T*r)


class TFiLM(nn.Module):
    """
    TFiLM layer: splits input into temporal blocks, pools, feeds an RNN (LSTM),
    produces gamma/beta per block and applies affine modulation to each block.
    Input: (B, C, T)
    Output: (B, C, T)
    """
    def __init__(self, channels, block_len=32, rnn_hidden=128, pooling='max'):
        super().__init__()
        self.channels = channels
        self.block_len = block_len
        self.pooling = pooling
        self.rnn = nn.LSTM(input_size=channels, hidden_size=rnn_hidden, batch_first=True)
        self.to_gamma = nn.Linear(rnn_hidden, channels)
        self.to_beta = nn.Linear(rnn_hidden, channels)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        bl = self.block_len
        if T < bl:
            # pad right to at least one block
            pad = bl - T
            x = F.pad(x, (0, pad))
            T = x.shape[2]

        n_blocks = T // bl
        if n_blocks == 0:
            raise ValueError("block_len too large for sequence length")

        # trim to exact multiple
        x = x[:, :, : n_blocks * bl]
        # reshape to (B, n_blocks, C, bl)
        xb = x.view(B, C, n_blocks, bl).permute(0, 2, 1, 3)  # (B, n_blocks, C, bl)

        # pool over temporal positions inside each block -> (B, n_blocks, C)
        if self.pooling == 'max':
            pooled = xb.max(dim=-1).values
        elif self.pooling == 'avg':
            pooled = xb.mean(dim=-1)
        else:
            raise ValueError("pooling must be 'max' or 'avg'")

        # RNN over blocks
        rnn_out, _ = self.rnn(pooled)  # (B, n_blocks, rnn_hidden)
        gamma = self.to_gamma(rnn_out)  # (B, n_blocks, C)
        beta = self.to_beta(rnn_out)

        # apply modulation blockwise
        # build list of (B, C, bl) blocks
        out_blocks = []
        for b in range(n_blocks):
            g = gamma[:, b, :].unsqueeze(-1)  # (B, C, 1)
            bt = beta[:, b, :].unsqueeze(-1)
            block = xb[:, b, :, :]  # (B, C, bl)
            out_blocks.append(g * block + bt)

        out = torch.cat(out_blocks, dim=-1)  # (B, C, n_blocks*bl)
        return out


def conv1d_same(in_ch, out_ch, kernel_size, dilation=1):
    pad = ((kernel_size - 1) // 2) * dilation
    return nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)


class DownBlock(nn.Module):
    """
    Downsampling block: conv -> dropout -> ReLU
    halves temporal dimension (stride=2)
    """
    def __init__(self, in_ch, out_ch, kernel_size=9, dropout=0.0, dilation=1):
        super().__init__()
        # stride 2 to downsample
        pad = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=2, padding=pad, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, T)
        return self.act(self.dropout(self.conv(x)))


class UpBlock(nn.Module):
    """
    Upsampling block: conv -> dropout -> ReLU -> subpixel1d to double temporal length
    We produce channels = out_ch * r then subpixel with r=2 -> out_ch channels doubled time
    """
    def __init__(self, in_ch, out_ch, kernel_size=9, dropout=0.0, dilation=1, upsample_r=2):
        super().__init__()
        self.upsample_r = upsample_r
        pad = ((kernel_size - 1) // 2) * dilation
        # produce out_ch * r channels
        self.conv = nn.Conv1d(in_ch, out_ch * upsample_r, kernel_size, padding=pad, dilation=dilation)
        self.subpixel = Subpixel1D(upsample_r)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)        # (B, out_ch*r, T)
        x = self.subpixel(x)    # (B, out_ch, T*r)
        x = self.dropout(x)
        return self.act(x)


class TimeSeriesSuperRes(nn.Module):
    """
    Full super-resolution model with TFiLM integration and skip connections.

    Args:
        in_channels: number of input channels (k in genomics or 1 for mono audio)
        K: number of downsampling / upsampling blocks
        base_kernel_len: base kernel length (will be adjusted per-level similar to paper)
        base_filters: starting number of filters (paper uses scheme min(26+k, 512))
        block_len: TFiLM block length (paper keeps T/B = 32)
        rnn_hidden: hidden size for TFiLM LSTM
        dropout: dropout inside blocks
        dilation: use dilated conv in later layers (paper uses dilation=2)
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 K=4,
                 block_len=32,
                 rnn_hidden=128,
                 dropout=0.0,
                 use_dilation=True):
        super().__init__()
        self.K = K
        self.block_len = block_len

        # Build down blocks
        self.downs = nn.ModuleList()
        self.tfilms_down = nn.ModuleList()
        in_ch = in_channels
        for k in range(1, K + 1):
            # filters and kernel length rules inspired by appendix B
            filters = min(26 + k, 512)
            kernel_len = max(2**(7 - k) + 1, 9) if k <= 6 else 9  # rough analog to paper's formula
            dilation = 2 if use_dilation else 1
            self.downs.append(DownBlock(in_ch, filters, kernel_size=kernel_len, dropout=dropout, dilation=dilation))
            self.tfilms_down.append(TFiLM(filters, block_len=block_len, rnn_hidden=rnn_hidden, pooling='max'))
            in_ch = filters

        # Bottleneck convs (a couple of conv layers without changing temporal size)
        self.bottleneck = nn.Sequential(
            conv1d_same(in_ch, in_ch, kernel_size=9),
            nn.ReLU(inplace=True),
            conv1d_same(in_ch, in_ch, kernel_size=9),
            nn.ReLU(inplace=True)
        )

        # Build up blocks (symmetric)
        self.ups = nn.ModuleList()
        self.tfilms_up = nn.ModuleList()
        for k in range(K, 0, -1):
            filters = min(26 + k, 512)
            kernel_len = max(2**(7 - k) + 1, 9)
            dilation = 2 if use_dilation else 1
            # input channels: previous filters (or bottleneck) + skip features (concatenated)
            in_ch = in_ch  # current in_ch from previous iteration
            # upblock will halve channels and double time; keep symmetric filters
            self.ups.append(UpBlock(in_ch, filters, kernel_size=kernel_len, dropout=dropout, dilation=dilation, upsample_r=2))
            self.tfilms_up.append(TFiLM(filters, block_len=block_len, rnn_hidden=rnn_hidden, pooling='max'))
            in_ch = filters  # after upblock output channels

        # final conv to produce out_channels and residual connection
        self.final_conv = conv1d_same(in_ch, out_channels, kernel_size=9)

    def forward(self, x):
        """
        x: (B, C_in, T)
        returns: (B, C_out, T_out) where T_out == T (residual added)
        """
        orig_len = x.shape[2]

        # store skip features
        skips = []
        cur = x
        for conv, tfilm in zip(self.downs, self.tfilms_down):
            cur = conv(cur)             # downsample (T -> T/2)
            cur = tfilm(cur)            # TFiLM on downsampled features
            skips.append(cur)

        cur = self.bottleneck(cur)

        # upsampling: symmetric with skip connections (concatenate along channel dim)
        for upblock, tfilm in zip(self.ups, self.tfilms_up):
            cur = upblock(cur)          # up: time x2
            # pop last skip and concatenate (note: matching temporal sizes assumed)
            if len(skips) == 0:
                # defensive: no skip
                pass
            else:
                skip = skips.pop()     # (B, C_skip, T_cur)
                # if time dims differ by 1 (odd lengths) we trim/pad
                if skip.size(2) != cur.size(2):
                    min_t = min(skip.size(2), cur.size(2))
                    skip = skip[:, :, :min_t]
                    cur = cur[:, :, :min_t]
                cur = torch.cat([cur, skip], dim=1)  # concat channels
            # apply TFiLM (on concatenated features; tfilm expects channels equal to out of upblock)
            # If channels mismatch, apply 1x conv to match — but in our design they should match.
            cur = tfilm(cur)

        # final conv and residual connection (learn y - x)
        # Ensure final temporal length equals original (may need to trim/pad)
        if cur.size(2) != orig_len:
            # if longer, trim; if shorter, pad with zeros on right
            if cur.size(2) > orig_len:
                cur = cur[:, :, :orig_len]
            else:
                cur = F.pad(cur, (0, orig_len - cur.size(2)))

        out = self.final_conv(cur)
        # residual: assume input channels == output channels or broadcastable
        if x.shape[1] == out.shape[1]:
            return out + x
        elif x.shape[1] == 1 and out.shape[1] > 1:
            # broadcast mono input to multiple channels before adding
            return out + x.repeat(1, out.shape[1], 1)
        else:
            # cannot directly add: just return out (user can wrap residual externally)
            return out




# ---------------------- Demo training on synthetic data ----------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------- Synthetic dataset ----------------------
def make_sine_batch(batch, channels, T, freqs=(1.0,10.0), noise=0.01, device='cpu'):
    t = torch.linspace(0, 1, T, device=device).unsqueeze(0).unsqueeze(0)  # 1,1,T
    out = torch.zeros((batch, channels, T), device=device)
    for b in range(batch):
        signal = torch.zeros((channels, T), device=device)
        for c in range(channels):
            f = random.uniform(freqs[0], freqs[1])
            phase = random.uniform(0, 2*math.pi)
            signal[c] = torch.sin(2*math.pi*f*t + phase).squeeze(0)
        signal += noise * torch.randn_like(signal)
        out[b] = signal
    return out

def downsample_signal(x, r):
    # x: B,C,T_high -> subsample every r-th sample
    return x[:, :, ::r]

def cubic_upsample(x_low, r, target_len):
    # x_low: B,C,T_low -> upsample to target_len using linear interpolation (approx cubic behavior)
    # torch.interpolate for 1D use mode='linear'
    return F.interpolate(x_low, size=target_len, mode='linear', align_corners=False)

# ---------------------- Quick training demo ----------------------
torch.manual_seed(0)
batch = 2
in_ch = 1
T_high = 512   # small for demo; paper uses 8192 patches
r = 4          # upsampling ratio (2,4,8 in paper)
T_low = T_high // r
block_len = T_high // 32  # enforce T/B = 32 as paper suggests

model = TimeSeriesSuperRes(in_channels=in_ch, out_channels=in_ch, K=4, block_len=block_len, rnn_hidden=64, dropout=0.0).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()

# one epoch over synthetic data
model.train()
for step in range(8):  # tiny number of steps for demo
    hi = make_sine_batch(batch, in_ch, T_high, freqs=(2.0,20.0), noise=0.01, device=DEVICE)
    low = downsample_signal(hi, r)              # subsampled low-res
    low_up = cubic_upsample(low, r, T_high)     # cubic (linear) upsample to match input length
    pred = model(low_up)                        # model predicts residual added inside model
    loss = criterion(pred, hi)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 1 == 0:
        print(f"Step {step:02d} loss {loss.item():.6f}  input shape {low_up.shape} pred shape {pred.shape}")

# run a forward on a test sample and show SNR improvement roughly
model.eval()
with torch.no_grad():
    hi = make_sine_batch(1, in_ch, T_high, freqs=(2.0,20.0), noise=0.0, device=DEVICE)
    low = downsample_signal(hi, r)
    low_up = cubic_upsample(low, r, T_high)
    pred = model(low_up)
    mse_input = F.mse_loss(low_up, hi).item()
    mse_model = F.mse_loss(pred, hi).item()
    def snr(mse, ref):
        # simple SNR in dB
        return 10 * math.log10((ref.pow(2).mean().item()) / (mse + 1e-12))
    ref_power = hi.pow(2).mean()
    print("MSE input (upsampled):", mse_input, "MSE model:", mse_model)
    print("SNR upsampled:", snr(mse_input, ref_power), "SNR model:", snr(mse_model, ref_power))

# Save model to disk for later reuse
torch.save(model.state_dict(), "/mnt/data/tfilm_superres_demo.pth")
print("Saved demo model to /mnt/data/tfilm_superres_demo.pth")