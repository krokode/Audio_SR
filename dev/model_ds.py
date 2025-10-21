import torch
import torch.nn as nn
import torch.nn.functional as F

class TFiLM(nn.Module):
    """
    Temporal Feature-Wise Linear Modulation layer
    """
    def __init__(self, input_channels, hidden_size, block_size, pooling_stride=2, pooling_extent=2):
        super(TFiLM, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.pooling_stride = pooling_stride
        self.pooling_extent = pooling_extent
        
        # RNN for generating modulation parameters
        self.rnn = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Linear layers to generate gamma and beta from RNN hidden state
        self.gamma_linear = nn.Linear(hidden_size, input_channels)
        self.beta_linear = nn.Linear(hidden_size, input_channels)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, channels)
        Returns:
            modulated_x: Tensor of same shape as x
        """
        batch_size, seq_len, channels = x.shape
        
        # Split into blocks along time dimension
        num_blocks = seq_len // self.block_size
        if seq_len % self.block_size != 0:
            raise ValueError(f"Sequence length {seq_len} must be divisible by block size {self.block_size}")
        
        # Reshape to (batch_size, num_blocks, block_size, channels)
        x_blocks = x.reshape(batch_size, num_blocks, self.block_size, channels)
        
        # Pool along block dimension to reduce sequence length for RNN
        # Using max pooling as mentioned in the paper
        if self.pooling_stride > 1:
            # Apply max pooling along the block dimension
            x_pooled = F.max_pool1d(
                x_blocks.reshape(batch_size * num_blocks, self.block_size, channels).transpose(1, 2),
                kernel_size=self.pooling_extent,
                stride=self.pooling_stride
            ).transpose(1, 2)
            
            # Calculate new dimensions after pooling
            pooled_block_size = x_pooled.shape[1]
            x_pooled = x_pooled.reshape(batch_size, num_blocks, pooled_block_size, channels)
            
            # Average over the pooled block dimension to get (batch_size, num_blocks, channels)
            x_pooled = x_pooled.mean(dim=2)
        else:
            # Average over the block dimension
            x_pooled = x_blocks.mean(dim=2)
        
        # Process through RNN to get modulation parameters
        rnn_output, _ = self.rnn(x_pooled)
        
        # Generate gamma and beta for each block
        gamma = self.gamma_linear(rnn_output)  # (batch_size, num_blocks, channels)
        beta = self.beta_linear(rnn_output)    # (batch_size, num_blocks, channels)
        
        # Apply modulation to each block
        gamma_expanded = gamma.unsqueeze(2).expand(-1, -1, self.block_size, -1)
        beta_expanded = beta.unsqueeze(2).expand(-1, -1, self.block_size, -1)
        
        # Apply feature-wise modulation: gamma * x + beta
        x_modulated = gamma_expanded * x_blocks + beta_expanded
        
        # Reshape back to original dimensions
        modulated_x = x_modulated.reshape(batch_size, seq_len, channels)
        
        return modulated_x

class DownsamplingBlock(nn.Module):
    """Downsampling block with convolution, dropout, and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, dilation=1):
        super(DownsamplingBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=(kernel_size-1)//2 * dilation,
            dilation=dilation
        )
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = F.relu(x)
        return x

class UpsamplingBlock(nn.Module):
    """Upsampling block with subpixel shuffling"""
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=2, dilation=1):
        super(UpsamplingBlock, self).__init__()
        self.scale_factor = scale_factor
        # Conv layer that increases channels for subpixel shuffling
        self.conv = nn.Conv1d(
            in_channels, out_channels * scale_factor, kernel_size,
            padding=(kernel_size-1)//2 * dilation, dilation=dilation
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
    Complete TFiLM-based super-resolution model for time series
    """
    def __init__(self, 
                 input_channels=1,
                 base_channels=64,
                 num_blocks=4,
                 tfilm_hidden_size=128,
                 block_size=256,
                 max_filters=512):
        super(TFiLMSuperResolution, self).__init__()
        
        self.num_blocks = num_blocks
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.tfilm_layers = nn.ModuleList()
        
        # Downsampling path
        in_ch = input_channels
        for i in range(num_blocks):
            out_ch = min(base_channels * (2 ** i), max_filters)
            kernel_size = max(128 // (2 ** i) + 1, 9)
            self.down_blocks.append(DownsamplingBlock(in_ch, out_ch, kernel_size, stride=2, dilation=2))
            # Add TFiLM layer after each downsampling block
            self.tfilm_layers.append(TFiLM(out_ch, tfilm_hidden_size, block_size))
            in_ch = out_ch
        
        # Bottleneck layer (no down/up sampling)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Upsampling path
        for i in range(num_blocks):
            in_ch_up = min(base_channels * (2 ** (num_blocks - i)), max_filters)
            out_ch_up = min(base_channels * (2 ** (num_blocks - i - 1)), max_filters)
            kernel_size = max(128 // (2 ** (num_blocks - i - 1)) + 1, 9)
            self.up_blocks.append(UpsamplingBlock(in_ch_up * 2, out_ch_up, kernel_size, scale_factor=2, dilation=2))
        
        # Final output layer
        self.final_conv = nn.Conv1d(base_channels, input_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        
        # Downsampling path with skip connections
        skip_connections = []
        for i in range(self.num_blocks):
            x = self.down_blocks[i](x)
            # Apply TFiLM modulation
            x_modulated = self.tfilm_layers[i](x.transpose(1, 2)).transpose(1, 2)
            skip_connections.append(x_modulated)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Upsampling path with skip connections
        for i in range(self.num_blocks):
            # Concatenate with skip connection from downsampling path
            skip = skip_connections[self.num_blocks - i - 1]
            x = torch.cat([x, skip], dim=1)
            x = self.up_blocks[i](x)
        
        # Final output
        x = self.final_conv(x)
        return x

class SmallCNNWithTFiLM(nn.Module):
    """
    Small CNN with TFiLM layers for text classification (as described in paper)
    """
    def __init__(self, vocab_size, embedding_dim, num_classes, hidden_dim=128, num_filters=100, tfilm_hidden=64):
        super(SmallCNNWithTFiLM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        
        # TFiLM layers
        self.tfilm1 = TFiLM(num_filters, tfilm_hidden, block_size=64)
        self.tfilm2 = TFiLM(num_filters, tfilm_hidden, block_size=64)
        self.tfilm3 = TFiLM(num_filters, tfilm_hidden, block_size=64)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_filters, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        # Convolutional layers with TFiLM
        x = F.relu(self.conv1(x))
        x = self.tfilm1(x.transpose(1, 2)).transpose(1, 2)
        
        x = F.relu(self.conv2(x))
        x = self.tfilm2(x.transpose(1, 2)).transpose(1, 2)
        
        x = F.relu(self.conv3(x))
        x = self.tfilm3(x.transpose(1, 2)).transpose(1, 2)
        
        # Classification
        x = self.classifier(x)
        return x

# Example usage
if __name__ == "__main__":
    # Audio super-resolution example
    batch_size, seq_len, channels = 4, 8192, 1
    model = TFiLMSuperResolution(
        input_channels=channels,
        base_channels=64,
        num_blocks=4,
        tfilm_hidden_size=128,
        block_size=256
    )
    
    x = torch.randn(batch_size, channels, seq_len)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Text classification example
    vocab_size, embedding_dim, num_classes = 10000, 100, 2
    text_model = SmallCNNWithTFiLM(vocab_size, embedding_dim, num_classes)
    
    text_input = torch.randint(0, vocab_size, (batch_size, 256))
    text_output = text_model(text_input)
    print(f"Text input shape: {text_input.shape}")
    print(f"Text output shape: {text_output.shape}")