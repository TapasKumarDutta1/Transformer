from Layers.layers import MultiHeadAttention, Norm
from Utils.utils import PositionalEncoder
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, in_channel, embed_dim, num_head, dropout=0):
        """
        Transformer Encoder Module.

        Args:
            in_channel (int): Input channels.
            embed_dim (int): Embedding dimension.
            num_head (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.MHA = MultiHeadAttention(num_head, in_channel, embed_dim, dropout)
        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.norm = Norm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, in_channel),
        )
        self.norm1 = Norm(embed_dim)

    def forward(self, x, mask=None):
        """
        Forward pass of the Transformer Encoder.

        Args:
            x (Tensor): Input tensor.
            mask (Tensor, optional): Mask tensor.

        Returns:
            Tensor: Output tensor from the Transformer Encoder.
        """
        x1 = self.norm(x)
        a = self.MHA(x1, mask)
        x = x + self.d1(a)
        x2 = self.norm1(x)
        x = x + self.d2(self.mlp(x2))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, in_channel, embed_dim, num_head, dropout=0):
        """
        Transformer Decoder Module.

        Args:
            in_channel (int): Input channels.
            embed_dim (int): Embedding dimension.
            num_head (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.MHA1 = MultiHeadAttention(num_head, in_channel, embed_dim, dropout)
        self.MHA2 = MultiHeadAttention(num_head, in_channel, embed_dim, dropout)
        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.d3 = nn.Dropout(dropout)
        self.norm1 = Norm(embed_dim)
        self.norm2 = Norm(embed_dim)
        self.norm3 = Norm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, in_channel),
        )

    def forward(self, x, memory, src_mask=None, trg_mask=None):
        """
        Forward pass of the Transformer Decoder.

        Args:
            x (Tensor): Input tensor for the decoder.
            memory (Tensor): Encoded input sequence from the encoder.
            src_mask (Tensor, optional): Source mask tensor.
            trg_mask (Tensor, optional): Target mask tensor.

        Returns:
            Tensor: Output tensor from the Transformer Decoder.
        """
        x1 = self.norm1(x)
        x1 = self.MHA1(x1, trg_mask)
        x = x + self.d1(x1)
        x2 = self.norm2(x)
        x2 = self.MHA2([x2, memory, memory], src_mask)
        x = x + self.d2(x2)
        x3 = self.norm3(x)
        x = x + self.d3(self.mlp(x3))
        return x


class Transformer(nn.Module):
    def __init__(self, N, num_head, src_vocab, trg_vocab, embed_dim):
        """
        Transformer Model.

        Args:
            N (int): Number of encoder and decoder layers.
            num_head (int): Number of attention heads.
            src_vocab (int): Source vocabulary size.
            trg_vocab (int): Target vocabulary size.
            embed_dim (int): Embedding dimension.
        """
        super().__init__()
        num_enc, num_dec = N, N
        self.src_embed = nn.Embedding(src_vocab, embed_dim)
        self.trg_embed = nn.Embedding(trg_vocab, embed_dim)
        self.out = nn.Linear(embed_dim, trg_vocab)
        self.norm1 = Norm(embed_dim)
        self.norm2 = Norm(embed_dim)
        self.pe_enc = PositionalEncoder(embed_dim)
        self.pe_dec = PositionalEncoder(embed_dim)

        self.encoder_list = nn.ModuleList(
            [TransformerEncoder(embed_dim, embed_dim, num_head) for _ in range(num_enc)]
        )

        self.decoder_list = nn.ModuleList(
            [TransformerDecoder(embed_dim, embed_dim, num_head) for _ in range(num_dec)]
        )

    def encode(self, x, x_mask):
        """
        Encodes the input sequence.

        Args:
            x (Tensor): Input tensor.
            x_mask (Tensor): Mask tensor for input.

        Returns:
            Tensor: Encoded output tensor.
        """
        x = self.src_embed(x)
        x = self.pe_enc(x)
        for layer in self.encoder_list:
            x = layer(x, x_mask)
        x = self.norm1(x)
        return x

    def decode(self, y, x, x_mask, y_mask):
        """
        Decodes the input sequence.

        Args:
            y (Tensor): Input tensor for decoding.
            x (Tensor): Encoded input sequence from the encoder.
            x_mask (Tensor): Mask tensor for input.
            y_mask (Tensor): Mask tensor for output.

        Returns:
            Tensor: Decoded output tensor.
        """
        y = self.trg_embed(y)
        y = self.pe_dec(y)
        for layer in self.decoder_list:
            y = layer(y, x, x_mask, y_mask)
        y = self.norm2(y)
        return y

    def forward(self, x, y, x_mask=None, y_mask=None):
        """
        Forward pass of the Transformer model.

        Args:
            x (Tensor): Input tensor.
            y (Tensor): Output tensor.
            x_mask (Tensor, optional): Mask tensor for input.
            y_mask (Tensor, optional): Mask tensor for output.

        Returns:
            Tensor: List of output tensors.
        """
        x = self.encode(x, x_mask)
        y = self.decode(y, x, x_mask, y_mask)
        return self.out(y)
