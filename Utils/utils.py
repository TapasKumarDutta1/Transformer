import torch
from torch import nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels):
        """
        Initialize the PatchEmbedding layer.

        Args:
            patch_size (int): The size of each patch.
            in_channels (int): The number of input channels.

        """
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, 3, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass of the PatchEmbedding layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after projecting and reshaping.

        """
        # Project input into patches
        x = self.proj(x)  # (B, E, H, W)
        B, C, H, W = x.shape

        # Reshape the tensor
        x = x.view(B, C, H * W)  # (B, E, H*W)
        x = x.transpose(1, 2)  # (B, H*W, E)
        return x

class Concatenate_CLS_Token(nn.Module):
    def __init__(self, N, embed_dim):
        """
        Initialize the Concatenate_CLS_Token layer.

        Args:
            N (int): Number of tokens.
            embed_dim (int): Dimension of the token embeddings.

        """
        super().__init__()
        self.CLS = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        """
        Forward pass of the Concatenate_CLS_Token layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after concatenating CLS tokens.

        """
        BATCH, _, _ = x.shape

        # Repeat the CLS token for each sample in the batch
        CLS = self.CLS.repeat(BATCH, 1, 1)

        # Concatenate CLS token with the input
        x = torch.cat([CLS, x], 1)
        return x

class Add_Positional_Embedding(nn.Module):
    def __init__(self, N, embed_dim):
        """
        Initialize the Add_Positional_Embedding layer.

        Args:
            N (int): Number of tokens.
            embed_dim (int): Dimension of the token embeddings.

        """
        super().__init__()
        self.N = N
        self.positional = nn.Parameter(torch.zeros(1, 1, int(embed_dim)))

    def forward(self, x):
        """
        Forward pass of the Add_Positional_Embedding layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after adding positional embeddings.

        """
        BATCH, _, _ = x.shape

        # Repeat the positional embedding for each sample in the batch
        positional = self.positional.repeat(BATCH, 1, 1)

        # Add positional embedding to the input
        x = x + positional
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=128, device='cuda'):
        """
        Positional Encoding for Transformer Models.

        Args:
            d_model (int): Dimension of the model.
            max_seq_len (int): Maximum sequence length.
            device : Device to run layer on.
        """
        super().__init__()
        self.d_model = d_model
        self.device  = device
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass of the PositionalEncoder.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying positional encoding.
        """
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False).to(
            self.device
        )
        return x
        
def positional_encoding(H, W, C):
    """
    Generate positional encoding for Transformer.

    Args:
        H (int): Height.
        W (int): Width.
        C (int): Channel dimension.

    Returns:
        Tensor: Positional encoding tensor.
    """
    # Create a sequence of numbers from 0 to H - 1 and reshape it to a column vector
    s = torch.arange(H)
    s = s.view(H, 1)

    # Create a sequence of even numbers from 0 to 2*C - 1 and reshape it to a row vector, then normalize
    d = 2 * torch.arange(C)
    d = d.view(1, C) / C

    # Calculate positional encoding values using sine and cosine functions
    z = s / (10000 ** d)
    pos = torch.stack([torch.sin(z[::2, :]), torch.cos(z[::2, :])], 1).view(-1, C)

    # Create positional encoding for X-coordinate by adding it to zeros
    pos_X = pos.unsqueeze(1) + torch.zeros((H, W, C))

    # Create a sequence of numbers from 0 to W - 1 and reshape it to a row vector
    t = torch.arange(W)
    t = t.view(1, W)

    # Reshape d to a column vector and normalize
    d = d.view(C, 1) / C

    # Calculate positional encoding values for Y-coordinate using sine and cosine functions
    z = t / (10000 ** d)
    pos = torch.stack([torch.sin(z[:, ::2]), torch.cos(z[:, ::2])], 1).view(-1, C)

    # Create positional encoding for Y-coordinate by adding it to zeros
    pos_Y = pos.unsqueeze(0) + torch.zeros((H, W, C))

    # Concatenate positional encodings for X and Y to get the final positional encoding
    pos = torch.cat([pos_X, pos_Y], -1)

    # Permute dimensions for compatibility with the model's input shape
    pos = pos.permute(2, 0, 1)

    # Add an additional dimension for the batch
    return pos.unsqueeze(0)
