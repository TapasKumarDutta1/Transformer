import torch
from torch import nn
from Utils.utils import *
from Layers.layers import MultiHeadAttention

class VisionTransformer(nn.Module):
    def __init__(self, H=384, W=384, num_layers=12, embed_dim=768, MLP_size=3072, num_class=2,
                 patch_size=16, num_head=12, batch_size=1):
        """
        Initialize a Vision Transformer model.

        Args:
            H (int): Height of the input image.
            W (int): Width of the input image.
            num_layers (int): Number of transformer layers.
            embed_dim (int): Dimension of token embeddings.
            MLP_size (int): Size of the intermediateMLP layers in the transformer.
            num_class (int): Number of output classes.
            patch_size (int): Size of image patches.
            num_head (int): Number of attention heads.
            batch_size (int): Batch size (default 1).

        """
        N = int(H * W / (patch_size ** 2))
        super(VisionTransformer, self).__init__()

        # Preprocessing layers
        self.preprocess = nn.Sequential(
            PatchEmbedding(patch_size, embed_dim),
            Concatenate_CLS_Token(N, embed_dim),
            Add_Positional_Embedding(N + 1, embed_dim)  # Adding 1 for the CLS token
        )

        # Transformer layers
        self.transformer = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttention(num_head, embed_dim, embed_dim)
            ) for _ in range(num_layers)
        ])

        self.layernorm = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm([embed_dim])
            ) for _ in range(num_layers)
        ])

        # MLP head
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm([embed_dim]),
                nn.Linear(embed_dim, MLP_size),
                nn.GELU(),
                nn.Linear(MLP_size, embed_dim)
            ) for _ in range(num_layers)
        ])

        # Classification head
        self.head = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        x = self.preprocess(x)
        for i in range(len(self.transformer)):
            x1 = self.transformer[i](x)
            x = x + x1
            x1 = self.mlp[i](x)
            x = x + x1
            x = self.layernorm[i](x)

        # Extract the CLS token and perform classification
        out = x[:, 0, :]
        out = self.head(out)
        return out
