import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, head, in_dim, dim):
        """
        Initialize the Multi-Head Attention layer.

        Args:
            head (int): Number of attention heads.
            in_dim (int): Input dimension.
            dim (int): Dimension of each head.

        """
        super().__init__()
        self.head = head
        self.dim = dim
        self.query_projection = nn.Linear(in_dim, dim * self.head)
        self.key_projection = nn.Linear(in_dim, dim * self.head)
        self.value_projection = nn.Linear(in_dim, dim * self.head)

        self.out = nn.Linear(dim * self.head, in_dim)

    def forward(self, x):
        """
        Forward pass of the Multi-Head Attention layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after multi-head attention computation.

        """
        query = self.query_projection(x)  # B, N, D*H
        key = self.key_projection(x)
        value = self.value_projection(x)

        B, N, D = x.shape

        query = query.view(B, N, self.head, self.dim)  # B, N, H, D
        key = key.view(B, N, self.head, self.dim)
        value = value.view(B, N, self.head, self.dim)

        query = query.transpose(1, 2)  # B, H, N, D
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)  # B, H, N, D

        attention_map = torch.matmul(query, key.transpose(-1, -2))  # B, H, N, N
        scaled_attention_map = attention_map / torch.sqrt(torch.tensor((self.dim)))
        scaled_attention_map = torch.nn.Softmax(-1)(scaled_attention_map)

        output = torch.matmul(scaled_attention_map, value)  # B, H, N, D
        output = output.transpose(1, 2)  # B, N, H, D
        output = output.reshape(B, N, self.head * self.dim)
        return self.out(output)
