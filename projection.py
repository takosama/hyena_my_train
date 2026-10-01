"""Import-safe Hyena projection used by the training script."""

from torch import Tensor, nn


class Projection(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        order: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 2,
    ):
        super().__init__()
        if embed_dim <= 0 or order < 1:
            raise ValueError("embed_dim and order must be positive")
        hidden_size = (order + 1) * embed_dim
        self.linear = nn.Linear(embed_dim, hidden_size)
        self.short_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=hidden_size,
        )
        self.hidden_size = hidden_size
        self.order = order
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """
        args
        - `x`: input tensor with shape (batches, length, embed dim)
        """
        # B: batch size, L: seq len, E: embed dim, N: order of hyena
        L = x.shape[1]
        x = self.linear(x).transpose(  # (B, L, E) -> (B, L, (N+1)*E)
            1, 2
        )  # (B, L, (N+1)*E) -> (B, (N+1)*E, L)
        x = self.short_conv(x)[..., :L]  # (B, (N+1)*E, L) -> (B, (N+1)*E, L)
        # (B, (N+1)*E, L) -> [(B, E, L)] * (N+1)
        return x.chunk(self.order + 1, dim=1)
