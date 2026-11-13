import pytest
import torch
from torch import nn

from model import BikeNet


def test_model_output_shape():
    model = BikeNet(input_size=8)

    x = torch.randn(10, 8)
    predictions = model(x)

    assert predictions.shape == (10,)


def test_model_single_prediction():
    model = BikeNet(input_size=8)

    x = torch.randn(1, 8)
    predictions = model(x)

    assert predictions.shape == (1,)


def test_model_has_two_hidden_layers():
    model = BikeNet(input_size=8)

    linear_layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]

    assert len(linear_layers) == 3


def test_model_uses_relu():
    model = BikeNet(input_size=8)

    relu_layers = [layer for layer in model.network if isinstance(layer, nn.ReLU)]

    assert len(relu_layers) == 2


@pytest.mark.parametrize("hidden_size", [8, 16, 32])
def test_model_different_hidden_sizes(hidden_size):
    model = BikeNet(input_size=8, hidden_size=hidden_size)

    linear_layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]

    assert linear_layers[0].out_features == hidden_size
    assert linear_layers[1].in_features == hidden_size
    assert linear_layers[1].out_features == hidden_size
    assert linear_layers[2].in_features == hidden_size

    x = torch.randn(5, 8)
    predictions = model(x)

    assert predictions.shape == (5,)


def test_optimizer_updates_all_layers():
    torch.manual_seed(42)

    model = BikeNet(input_size=8)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    loss_fn = nn.MSELoss()

    x = torch.randn(16, 8)
    y = torch.randn(16)

    original_weights = [parameter.clone() for parameter in model.parameters()]

    predictions = model(x)
    loss = loss_fn(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for before, after in zip(original_weights, model.parameters(), strict=True):
        assert not torch.equal(before, after)
