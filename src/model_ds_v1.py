import torch
import torch.nn as nn
import torch.nn.functional as F

class TFiLM(nn.Module):
    """Temporal Feature-Wise Linear Modulation layer"""
    def __init__(self, input_channels, hidden_size, block_size, pooling_stride=2, pooling_extent=2):
        super(TFiLM, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.pooling_stride = pooling_stride
        self.pooling_extent = pooling_extent
        
        self.rnn = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.gamma_linear = nn.Linear(hidden_size, input_channels)
        self.beta_linear = nn.Linear(hidden_size, input_channels)
        
    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        
        # Split into blocks along time dimension
        num_blocks = seq_len // self.block_size
        if seq_len % self.block_size != 0:
            # Use padding if sequence length is not divisible by block size
            padding = self.block_size - (seq_len % self.block_size)
            x = F.pad(x, (0, 0, 0, padding))
            seq_len = x.shape[1]
            num_blocks = seq_len // self.block_size
        
        x_blocks = x.reshape(batch_size, num_blocks, self.block_size, channels)
        
        # Pool along block dimension
        if self.pooling_stride > 1:
            x_pooled = F.max_pool1d(
                x_blocks.reshape(batch_size * num_blocks, self.block_size, channels).transpose(1, 2),
                kernel_size=self.pooling_extent,
                stride=self.pooling_stride
            ).transpose(1, 2)
            pooled_block_size = x_pooled.shape[1]
            x_pooled = x_pooled.reshape(batch_size, num_blocks, pooled_block_size, channels)
            x_pooled = x_pooled.mean(dim=2)
        else:
            x_pooled = x_blocks.mean(dim=2)
        
        # Process through RNN
        rnn_output, _ = self.rnn(x_pooled)
        
        # Generate gamma and beta
        gamma = self.gamma_linear(rnn_output)
        beta = self.beta_linear(rnn_output)
        
        # Apply modulation
        gamma_expanded = gamma.unsqueeze(2).expand(-1, -1, self.block_size, -1)
        beta_expanded = beta.unsqueeze(2).expand(-1, -1, self.block_size, -1)
        
        x_modulated = gamma_expanded * x_blocks + beta_expanded
        modulated_x = x_modulated.reshape(batch_size, seq_len, channels)
        
        return modulated_x

class UpsamplingBlock(nn.Module):
    """Upsampling block with configurable scale factor"""
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=2, dilation=1):
        super(UpsamplingBlock, self).__init__()
        self.scale_factor = scale_factor
        
        # Conv layer that increases channels for subpixel shuffling
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels * scale_factor,  # Important: multiply by scale factor
            kernel_size,
            padding=(kernel_size-1)//2 * dilation, 
            dilation=dilation
        )
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = F.relu(x)
        
        # Subpixel shuffling for upsampling
        batch_size, channels, length = x.shape
        x = x.reshape(batch_size, self.scale_factor, channels // self.scale_factor, length)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, channels // self.scale_factor, length * self.scale_factor)
        return x

class TFiLMSuperResolution(nn.Module):
    """
    Complete TFiLM-based super-resolution model with explicit upscale factor
    """
    def __init__(self, 
                 input_channels=1,
                 base_channels=64,
                 num_blocks=4,
                 tfilm_hidden_size=128,
                 block_size=256,
                 upscale_factor=4,  # Explicit upscale factor parameter
                 max_filters=512,
                 quality_mode=True):  # If True, maintain input length and focus on quality
        super(TFiLMSuperResolution, self).__init__()
        
        self.upscale_factor = 1 if quality_mode else upscale_factor
        self.num_blocks = num_blocks
        self.quality_mode = quality_mode
        
        # Calculate how many up/down sampling steps we need
        # In quality mode, we use blocks for feature extraction without changing length
        if quality_mode:
            self.num_upsample_steps = self.num_blocks  # Match downs with ups to preserve length
            print("Quality improvement mode: maintaining input length")
        else:
            # upscale_factor = 2^(num_upsample_steps)
            self.num_upsample_steps = int(torch.log2(torch.tensor(upscale_factor)).item())
            print(f"Upscaling mode - factor: {upscale_factor}")
            print(f"Number of upsampling steps needed: {self.num_upsample_steps}")
            
            # Adjust number of blocks if needed
            if self.num_upsample_steps > num_blocks:
                print(f"Warning: Requested upscale factor {upscale_factor} requires {self.num_upsample_steps} blocks, but only {num_blocks} available")
                self.num_upsample_steps = num_blocks
        
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.tfilm_layers = nn.ModuleList()
        
        # Downsampling path
        in_ch = input_channels
        for i in range(num_blocks):
            out_ch = min(base_channels * (2 ** i), max_filters)
            kernel_size = max(128 // (2 ** i) + 1, 9)
            self.down_blocks.append(
                nn.Conv1d(in_ch, out_ch, kernel_size, stride=2, padding=(kernel_size-1)//2)
            )
            self.tfilm_layers.append(TFiLM(out_ch, tfilm_hidden_size, block_size))
            in_ch = out_ch
        
        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Upsampling path - only create as many upsampling blocks as needed
        for i in range(self.num_upsample_steps):
            in_ch_up = min(base_channels * (2 ** (num_blocks - i - 1)), max_filters)
            out_ch_up = min(base_channels * (2 ** (num_blocks - i - 2)) if (num_blocks - i - 2) >= 0 else base_channels, max_filters)
            kernel_size = max(128 // (2 ** (num_blocks - i - 1)) + 1, 9)
            
            self.up_blocks.append(
                UpsamplingBlock(in_ch_up * 2, out_ch_up, kernel_size, scale_factor=2, dilation=1)
            )
        
        # Final output layer
        self.final_conv = nn.Conv1d(out_ch_up, input_channels, kernel_size=3, padding=1)
        
        # Initial cubic upscaling to match target length (as mentioned in paper)
        self.initial_upscale = None
        if upscale_factor > (2 ** self.num_upsample_steps):
            remaining_scale = upscale_factor // (2 ** self.num_upsample_steps)
            self.initial_upscale = nn.Upsample(scale_factor=remaining_scale, mode='linear', align_corners=False)
        
    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        original_length = x.shape[-1]
        
        # Apply initial upscaling if needed
        if self.initial_upscale is not None:
            x = self.initial_upscale(x)
        
        # Store input for residual connection
        input_residual = x
        
        # Downsampling path with skip connections
        skip_connections = []
        for i in range(self.num_blocks):
            x = self.down_blocks[i](x)
            x = F.relu(x)
            # Apply TFiLM modulation
            x_modulated = self.tfilm_layers[i](x.transpose(1, 2)).transpose(1, 2)
            skip_connections.append(x_modulated)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Upsampling path with skip connections
        for i in range(self.num_upsample_steps):
            # Concatenate with skip connection from downsampling path
            skip_idx = self.num_blocks - i - 1
            skip = skip_connections[skip_idx]
            
            # Ensure skip connection has the same spatial dimensions
            if skip.shape[-1] != x.shape[-1]:
                skip = F.interpolate(skip, size=x.shape[-1], mode='linear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = self.up_blocks[i](x)
        
        # Final output with residual connection
        x = self.final_conv(x)
        
        # Handle output length based on mode
        if self.quality_mode:
            # In quality mode, ensure output matches input length
            if x.shape[-1] != original_length:
                x = F.interpolate(x, size=original_length, mode='linear', align_corners=False)
        else:
            # In upscaling mode, ensure output is upscaled by the factor
            target_length = original_length * self.upscale_factor
            if x.shape[-1] != target_length:
                x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        
        # Add residual connection (model learns y - x)
        if input_residual.shape[-1] == x.shape[-1]:
            x = x + input_residual
        
        return x

# Factory function for different upscale factors
def create_tfilm_super_resolution(upscale_factor, quality_mode=False, **kwargs):
    """
    Create TFiLM super-resolution model.
    
    Args:
        upscale_factor: Factor by which to increase resolution (ignored if quality_mode=True)
        quality_mode: If True, maintain input length and focus on signal quality improvement
        **kwargs: Additional arguments passed to TFiLMSuperResolution
    """
    # For quality mode, use more blocks for better feature extraction
    if quality_mode:
        num_blocks = 4  # Fixed number for quality improvement
    else:
        # For upscaling, determine blocks based on factor
        num_blocks = max(2, int(torch.log2(torch.tensor(upscale_factor)).item()))
    
    model = TFiLMSuperResolution(
        upscale_factor=upscale_factor,
        num_blocks=num_blocks,
        quality_mode=quality_mode,
        **kwargs
    )
    return model

# Example usage with different upscale factors
if __name__ == "__main__":
    batch_size, seq_len, channels = 64, 2048, 1
    
    # Test different upscale factors
    for upscale_factor in [2, 4, 8]:
        print(f"\n=== Testing upscale factor: {upscale_factor} ===")
        
        model = create_tfilm_super_resolution(
            upscale_factor=upscale_factor,
            input_channels=channels,
            base_channels=64,
            tfilm_hidden_size=128,
            block_size=256,
            quality_mode=False
        )
        
        x = torch.randn(batch_size, channels, seq_len)
        output = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Actual upscale: {output.shape[-1] / x.shape[-1]:.1f}x")
        
        # Verify the model can handle the upscale
        assert output.shape[-1] == seq_len * upscale_factor, \
            f"Upscale failed: expected {seq_len * upscale_factor}, got {output.shape[-1]}"