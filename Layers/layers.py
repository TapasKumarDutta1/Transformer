import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, head, in_dim, dim, dropout=0.1):
        """
        Initialize the Multi-Head Attention layer.

        Args:
            head (int): Number of attention heads.
            in_dim (int): Input dimension.
            dim (int): Dimension of each head.

        """
        super().__init__()
        self.head = head
        self.dim  = dim // self.head
        self.query_projection = nn.Linear(in_dim, self.dim * self.head)
        self.key_projection   = nn.Linear(in_dim, self.dim * self.head)
        self.value_projection = nn.Linear(in_dim, self.dim * self.head)
        self.dropout = None
        if dropout  != None:
          self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.dim * self.head, in_dim)

    def forward(self, x, mask=None):
        """
        Forward pass of the Multi-Head Attention layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after multi-head attention computation.

        """
        if len(x) == 3:
            q, k, v  = x[0], x[1], x[2]
            B, N, D  = q.shape
            _, N1, _ = k.shape
            _, N2, _ = v.shape
        else:
            q, k, v = x, x, x
            B, N, D = x.shape
            N1, N2  = N, N
            
        query = self.query_projection(q)  # B, N, D*H
        key   = self.key_projection(k)
        value = self.value_projection(v)
        print(query.shape, key.shape, value.shape)
        query = query.view(B, N, self.head, self.dim)  # B, N, H, D
        key   = key.view(B, N1, self.head, self.dim)
        value = value.view(B, N2, self.head, self.dim)

        query = query.transpose(1, 2)  # B, H, N, D
        key   = key.transpose(1, 2)
        value = value.transpose(1, 2)  # B, H, N, D

        attention_map = torch.matmul(query, key.transpose(-1, -2))  # B, H, N, N
        scaled_attention_map = attention_map / torch.sqrt(torch.tensor((self.dim)))
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scaled_attention_map = scaled_attention_map.masked_fill(mask == 0, -1e9)
        
        scaled_attention_map = torch.nn.Softmax(-1)(scaled_attention_map)
        if self.dropout is not None:
            scaled_attention_map = self.dropout(scaled_attention_map)

        output = torch.matmul(scaled_attention_map, value)  # B, H, N, D
        output = output.transpose(1, 2)  # B, N, H, D
        output = output.reshape(B, N, self.head * self.dim)
        return self.out(output)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder Module for sequence modeling.

    Args:
        N (int): Number of tokens in the sequence.
        in_channel (int): Input channel dimension.
        embed_dim (int): Dimension of the token embeddings.
        num_head (int): Number of attention heads.
    """
    def __init__(self, N, in_channel, embed_dim, num_head):
        super().__init__()
        # Multi-Head Attention Layer
        self.MHA = MultiHeadAttention(num_head, in_channel, embed_dim)
        # Dropout Layer
        self.d1 = nn.Dropout(p=0.1)
        # Layer Normalization
        self.norm = nn.LayerNorm([N, in_channel])
        # MLP Layer
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // 4),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_channel // 4, in_channel)
        )
        # Layer Normalization
        self.norm1 = nn.LayerNorm([N, in_channel])

    def forward(self, x, positional_encoding):
        """
        Forward pass of the Transformer Encoder.

        Args:
            x (Tensor): Input tensor.
            positional_encoding (Tensor): Positional encoding tensor.

        Returns:
            Tensor: Output tensor.
        """
        a = self.MHA([x + positional_encoding, x + positional_encoding, x])
        x = x + self.d1(a)
        x = self.norm(x)
        x = x + self.mlp(x)
        x = self.norm1(x)
        return x

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder Module for sequence modeling.

    Args:
        N (int): Number of tokens in the sequence.
        in_channel (int): Input channel dimension.
        embed_dim (int): Dimension of the token embeddings.
        num_head (int): Number of attention heads.
        num_queries (int): Number of queries.
    """
    def __init__(self, N, in_channel, embed_dim, num_head, num_queries):
        super().__init__()
        self.num_queries = num_queries
        # Multi-Head Attention Layers
        self.MHA1 = MultiHeadAttention(num_head, in_channel, embed_dim)
        self.MHA2 = MultiHeadAttention(num_head, in_channel, embed_dim)
        # Dropout Layers
        self.d1 = nn.Dropout(p=0.1)
        self.d2 = nn.Dropout(p=0.1)
        # Layer Normalization
        self.norm1 = nn.LayerNorm([self.num_queries, in_channel])
        self.norm2 = nn.LayerNorm([self.num_queries, in_channel])

    def forward(self, x, memory, positional_encoding, query_embed):
        """
        Forward pass of the Transformer Decoder.

        Args:
            x (Tensor): Input tensor.
            memory (Tensor): Memory tensor.
            positional_encoding (Tensor): Positional encoding tensor.
            query_embed (Tensor): Query embedding tensor.

        Returns:
            Tensor: Output tensor.
        """
        y = self.MHA1([x + query_embed, x + query_embed, x])
        x = x + self.d1(y)
        x = self.norm1(x)
        z = self.MHA2([x + query_embed, memory + positional_encoding, memory])
        z = x + self.d2(z)
        z = self.norm2(z)
        return z

class Transformer(nn.Module):
    """
    Complete Transformer Model for sequence modeling.

    Args:
        num_enc (int): Number of encoder layers.
        num_dec (int): Number of decoder layers.
        in_channel (int): Input channel dimension.
        num_head (int): Number of attention heads.
        N (int): Number of tokens in the sequence.
        embed_dim (int): Dimension of the token embeddings.
        positional_encoding (Tensor): Positional encoding tensor.
        query_embed (Tensor): Query embedding tensor.
        num_queries (int): Number of queries.
    """
    def __init__(self, num_enc, num_dec, in_channel, num_head, N, embed_dim, positional_encoding, query_embed, num_queries):
        super().__init__()
        self.positional_encoding = positional_encoding
        self.num_queries = num_queries
        self.query_embed = query_embed

        # Transformer Encoder Layers
        self.encoder_list = nn.ModuleList([TransformerEncoder(N, in_channel, embed_dim, num_head) for _ in range(num_enc)])

        # Transformer Decoder Layers
        self.decoder_list = nn.ModuleList([TransformerDecoder(N, in_channel, embed_dim, num_head, num_queries) for _ in range(num_dec)])

    def forward(self, x):
        """
        Forward pass of the Transformer model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            List[Tensor]: List of output tensors.
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        x = x.transpose(2, 1)

        # Encoding
        for layer in self.encoder_list:
            x = layer(x, self.positional_encoding)

        target = torch.zeros_like(self.query_embed)
        outputs = []

        # Decoding
        for layer in self.decoder_list:
            target = layer(target, x, self.positional_encoding, self.query_embed)
            outputs.append(target)

        return outputs

