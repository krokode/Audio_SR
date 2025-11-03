import torch
import torch.nn as nn
from torchinfo import summary

class DownsamplingBlock(nn.Module):
    """
    Downsampling block as described in paper (Figure 2, Appendix B)
    Performs convolution, dropout, and ReLU nonlinearity
    Halves spatial dimension and increases filter size
    """
    def __init__(self, in_channels, out_channels, filter_length, stride=2):
        super(DownsamplingBlock, self).__init__()
        padding = (filter_length - 1) // 2  # Maintain temporal dimension before striding
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=filter_length,
            stride=stride,
            padding=padding
        )
        self.dropout = nn.Dropout(0.1)  # Dropout as mentioned in Appendix B
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x


class UpsamplingBlock(nn.Module):
    """
    Upsampling block as described in paper (Figure 2, Appendix B)
    Uses subpixel shuffling for upscaling to avoid artifacts
    """
    def __init__(self, in_channels, out_channels, filter_length, upscale_factor=2):
        super(UpsamplingBlock, self).__init__()
        self.upscale_factor = upscale_factor
        
        # First convolution before upscaling
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels * upscale_factor,  # Prepare for subpixel shuffle
            kernel_size=filter_length,
            padding=(filter_length - 1) // 2
        )
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        # Subpixel shuffling (1D version of paper's approach from Appendix B)
        # Reshape from (batch, channels * r, length) to (batch, channels, length * r)
        batch_size, channels, length = x.shape
        channels_out = channels // self.upscale_factor
        x = x.reshape(batch_size, channels_out, self.upscale_factor, length)
        x = x.permute(0, 1, 3, 2).reshape(batch_size, channels_out, length * self.upscale_factor)
        
        return x


class TFiLMLayer(nn.Module):
    """Temporal Feature-Wise Linear Modulation layer (Section 3)"""
    def __init__(self, channels, hidden_size, block_size):
        super(TFiLMLayer, self).__init__()
        self.block_size = block_size
        self.channels = channels
        
        # RNN for generating modulation parameters (LSTM as mentioned in paper)
        self.rnn = nn.LSTM(channels, hidden_size, batch_first=True)
        # Linear layers to generate gamma and beta
        self.gamma_net = nn.Linear(hidden_size, channels)
        self.beta_net = nn.Linear(hidden_size, channels)
        
    def forward(self, x):
        # x shape: (batch, channels, length)
        batch_size, channels, length = x.shape
        
        # Split into blocks along time dimension (Section 3)
        num_blocks = length // self.block_size
        if length % self.block_size != 0:
            # Pad if necessary
            padding = self.block_size - (length % self.block_size)
            x = nn.functional.pad(x, (0, padding))
            num_blocks = (length + padding) // self.block_size
        
        # Reshape to (batch, num_blocks, block_size, channels)
        x_blocks = x.reshape(batch_size, channels, num_blocks, self.block_size)
        x_blocks = x_blocks.permute(0, 2, 3, 1)  # (batch, num_blocks, block_size, channels)
        
        # Pool along block dimension (Section 3)
        x_pooled = torch.max(x_blocks, dim=2)[0]  # Max pooling: (batch, num_blocks, channels)
        
        # Process with RNN to get gamma and beta for each block
        gamma_betas, _ = self.rnn(x_pooled)
        gammas = self.gamma_net(gamma_betas)  # (batch, num_blocks, channels)
        betas = self.beta_net(gamma_betas)    # (batch, num_blocks, channels)
        
        # Apply feature-wise modulation to each block
        gammas = gammas.unsqueeze(2)  # (batch, num_blocks, 1, channels)
        betas = betas.unsqueeze(2)    # (batch, num_blocks, 1, channels)
        
        modulated_blocks = gammas * x_blocks + betas
        
        # Reshape back to original format
        modulated_blocks = modulated_blocks.permute(0, 3, 1, 2)  # (batch, channels, num_blocks, block_size)
        output = modulated_blocks.reshape(batch_size, channels, -1)
        
        # Remove padding if we added any
        if output.shape[2] > length:
            output = output[:, :, :length]
            
        return output


class TFiLMSuperResolution(nn.Module):
    """
    Complete TFiLM-based super-resolution model based on 1909.06628v3.pdf
    Using paper-exact formulas for channel dimensions
    """
    def __init__(self, 
                 input_channels=1,
                 num_blocks=4,  # K in the paper
                 tfilm_hidden_size=128,
                 block_size=256,  # B in the paper - from Section 5.2: T/B = 32
                 upscale_factor=4):  # r in the paper
        super(TFiLMSuperResolution, self).__init__()
        
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.upscale_factor = upscale_factor
        
        # PAPER-EXACT FILTER COUNTS (Appendix B)
        self.down_filters = [min(2**(6 + k), 512) for k in range(num_blocks)]
        self.up_filters = [min(2**(7 + (num_blocks - k)), 512) for k in range(num_blocks)]
        
        # PAPER-EXACT FILTER LENGTHS (Appendix B)
        self.down_lengths = [max(2**(7 - k) + 1, 9) for k in range(num_blocks)]
        self.up_lengths = [max(2**(7 - (num_blocks - k)) + 1, 9) for k in range(num_blocks)]
        
        print(f"Down filters: {self.down_filters}")
        print(f"Up filters: {self.up_filters}")
        
        # Build modules
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.tfilm_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # Downsampling path
        in_ch = input_channels
        for k in range(num_blocks):
            self.down_blocks.append(
                DownsamplingBlock(in_ch, self.down_filters[k], self.down_lengths[k])
            )
            self.tfilm_layers.append(
                TFiLMLayer(self.down_filters[k], tfilm_hidden_size, block_size)
            )
            in_ch = self.down_filters[k]
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Upsampling path with skip connections - FIXED CHANNEL DIMENSIONS
        # We need to track the input channels for each upsampling block
        up_in_channels = [self.down_filters[-1]]  # Start with bottleneck output channels
        
        for k in range(num_blocks):
            # Calculate input channels for this upsampling block
            if k == 0:
                # First upsampling block gets input from bottleneck
                current_in_ch = self.down_filters[-1]
            else:
                # Subsequent blocks get input from previous upsampling block + skip connection
                current_in_ch = self.up_filters[k-1]
            
            # Skip connection - match the upsampling block's expected input
            skip_out_channels = self.up_filters[k] // 2
            skip_conv = nn.Conv1d(
                self.down_filters[num_blocks - 1 - k],  # Reverse order for skip
                skip_out_channels,
                kernel_size=1
            )
            self.skip_connections.append(skip_conv)
            
            # Upsampling block - input channels = current_in_ch, output = up_filters[k]
            self.up_blocks.append(
                UpsamplingBlock(current_in_ch, self.up_filters[k] // 2, self.up_lengths[k], upscale_factor=2)
            )
            
            up_in_channels.append(self.up_filters[k])
        
        # Final output with residual connection (Appendix B)
        self.output_conv = nn.Conv1d(self.up_filters[-1], input_channels, kernel_size=1)
        
    def forward(self, x):
        #print(f"Input shape: {x.shape}")  # Debug line
        # Store original input for residual connection
        identity = x
        
        # Downsampling path
        skip_features = []
        for i, (down_block, tfilm_layer) in enumerate(zip(self.down_blocks, self.tfilm_layers)):
            x = down_block(x)
            #print(f"After down block {i}: {x.shape}")  # Debug line
            x = tfilm_layer(x)
            #print(f"After TFiLM {i}: {x.shape}")  # Debug line
            skip_features.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Upsampling path with skip connections
        for k, (up_block, skip_conv) in enumerate(zip(self.up_blocks, self.skip_connections)):
            # Get corresponding skip feature (reverse order)
            skip_idx = self.num_blocks - 1 - k
            skip = skip_conv(skip_features[skip_idx])
            
            # Apply upsampling
            x = up_block(x)
            
            # Ensure skip connection matches current spatial dimensions
            current_length = x.shape[2]
            if skip.shape[2] != current_length:
                # Resize skip connection to match current spatial dimensions
                skip = nn.functional.interpolate(skip, size=current_length, mode='linear', align_corners=False)
            
            # Concatenate with skip connection features
            x = torch.cat([x, skip], dim=1)
        
        # Final output with residual connection
        x = self.output_conv(x)
        
        # Ensure output matches input length for residual connection
        if x.shape[2] != identity.shape[2]:
            x = nn.functional.interpolate(x, size=identity.shape[2], mode='linear', align_corners=False)
        
        # Add residual connection: model learns y-x (Appendix B)
        output = x + identity
        
        return output