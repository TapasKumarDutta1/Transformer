from Layers.layers import MultiHeadAttention
from Utils.utils import positional_encoding
import torch
import torchvision.models as models
from torch import nn


class detection_transformerDETR(nn.Module):
    """
    DETR (Detection Transformer) Model.

    Args:
        size (int): Size.
        batch_size (int): Batch size.
        device: Device.
        num_queries (int): Number of queries.
        hidden_dim (int): Hidden dimension.
        num_enc (int): Number of encoder layers.
        num_dec (int): Number of decoder layers.
        num_head (int): Number of attention heads.
        num_class (int): Number of classes.
        embed_dim (int): Embedding dimension.
    """
    def __init__(self, size, batch_size, device, num_queries=10, hidden_dim=8, num_enc=2, num_dec=2, num_head=2, num_class=80, embed_dim=2):
        """
        Initialize the DETR (Detection Transformer) model.

        Args:
            size (int): Size.
            batch_size (int): Batch size.
            device: Device.
            num_queries (int): Number of queries.
            hidden_dim (int): Hidden dimension.
            num_enc (int): Number of encoder layers.
            num_dec (int): Number of decoder layers.
            num_head (int): Number of attention heads.
            num_class (int): Number of classes.
            embed_dim (int): Embedding dimension.
        """
        super().__init__()
        H, W = size, size

        # Load a pretrained ResNet-50 model
        resnet50 = models.resnet50(pretrained=True)
        model = models.resnet50(pretrained=True)

        # Freeze batch normalization layers
        for name, param in model.named_parameters():
            if "bn" in name:
                param.requires_grad = False

        # Use the first 5 layers of ResNet-50 as the backbone
        self.backbone = torch.nn.Sequential(*list(resnet50.children())[:5])  # 256, H/4, W/4
        C, H, W = 256, H // 4, W // 4
        N = H * W
        self.num_queries = num_queries

        # Query Embedding
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim)).repeat((batch_size, 1, 1))
        self.query_embed = self.query_embed.to(device)

        # Convolutional Reduction Layer
        self.reduce = torch.nn.Sequential(torch.nn.Conv2d(C, hidden_dim, 1),
                                          torch.nn.BatchNorm2d(hidden_dim),
                                          torch.nn.ReLU())

        # Positional Encoding
        self.positional_encoding = positional_encoding(H, W, hidden_dim // 2)
        self.positional_encoding = self.positional_encoding.view(batch_size, hidden_dim, H * W)
        self.positional_encoding = self.positional_encoding.transpose(2, 1)
        self.positional_encoding = self.positional_encoding.to(device)

        # Transformer
        self.transformer = Transformer(num_enc, num_dec, hidden_dim, num_head, N, embed_dim, self.positional_encoding,
                                      self.query_embed, self.num_queries)

        # Class Embedding
        self.class_embed = torch.nn.Linear(hidden_dim, num_class + 1)

        # Position Embedding
        self.pos_embed = nn.Sequential(
            nn.LayerNorm([self.num_queries, hidden_dim]),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 4)
        )


    def forward(self, inp):
        """
        Forward pass of the DETR model.

        Args:
            inp (Tensor): Input tensor.

        Returns:
            List[Tensor]: List of output tensors.
        """
        x = self.backbone(inp)
        x = self.reduce(x)
        x = self.transformer(x)  # B, C, H, W
        class_pred = []
        pos_pred = []
        for i in x:
            class_pred.append(self.class_embed(i))
            pos_pred.append(torch.nn.Sigmoid()(self.pos_embed(i)))
        return class_pred, pos_pred
