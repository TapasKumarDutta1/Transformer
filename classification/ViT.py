import torch
from torch import nn
from Utils.utils import *
from Layers.layers import MultiHeadAttention
class VisionTransformer(nn.Module):
    def __init__(self, H=32, W=32, num_layers=8, embed_dim=4, MLP_size=2, num_class=2,
                 patch_size=8, num_head=2, batch_size=1, in_channel=512):
        """
        Initialize a Vision Transformer model.

        Args:
            H (int): Height of the input image.
            W (int): Width of the input image.
            num_layers (int): Number of transformer layers.
            embed_dim (int): Dimension of token embeddings.
            MLP_size (int): Size of the MLP layers in the transformer.
            num_class (int): Number of output classes.
            patch_size (int): Size of image patches.
            num_head (int): Number of attention heads.
            batch_size (int): Batch size (default 1).
            in_channel (int): Number of input channels (e.g., 3 for RGB images).

        """
        N = int(H * W / (patch_size ** 2))
        super(VisionTransformer, self).__init__()

        # Preprocessing layers
        self.preprocess = nn.Sequential(
            PatchEmbedding(patch_size, in_channel),
            Concatenate_CLS_Token(N, in_channel),
            Add_Positional_Embedding(N + 1, in_channel)  # Adding 1 for the CLS token
        )

        # Transformer layers
        self.transformer = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm([N + 1, in_channel]),
                MultiHeadAttention(num_head, in_channel, embed_dim)
            ) for _ in range(num_layers)
        ])

        # MLP head
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm([N + 1, in_channel]),
                nn.Linear(in_channel, MLP_size),
                nn.GELU(),
                nn.Linear(MLP_size, in_channel)
            ) for _ in range(num_layers)
        ])

        # Classification head
        self.head = nn.Linear(in_channel, num_class)

    def forward(self, x):
        x = self.preprocess(x)
        for i in range(len(self.transformer)):
            x1 = self.transformer[i](x)
            x = x + x1
            x1 = self.mlp[i](x)
            x = x + x1

        # Extract the CLS token and perform classification
        out = x[:, 0, :]
        out = self.head(out)
        return out
