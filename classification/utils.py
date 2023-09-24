import torch
from torch import nn

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
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=patch_size, stride=patch_size)

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

