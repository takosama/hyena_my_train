import pytest
import torch

from projection import Projection


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("width", [1, 4, 8])
def test_all_channels_are_preserved(order, width):
    torch.manual_seed(0)
    model = Projection(width, order)
    x = torch.randn(2, 5, width, requires_grad=True)
    parts = model(x)
    assert len(parts) == order + 1
    assert all(p.shape == (2, width, 5) for p in parts)
    expected = model.short_conv(model.linear(x).transpose(1, 2))[..., :5]
    torch.testing.assert_close(torch.cat(parts, dim=1), expected)
    sum(p.square().sum() for p in parts).backward()
    assert torch.all(model.linear.weight.grad.abs().sum(dim=1) > 0)
    assert torch.isfinite(x.grad).all()


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        Projection(0)
    with pytest.raises(ValueError):
        Projection(4, order=0)
