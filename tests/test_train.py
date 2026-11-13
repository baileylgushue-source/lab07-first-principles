import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import BikeNet
from train import run_epoch, train_model


def make_small_loader():
    torch.manual_seed(42)

    x = torch.randn(8, 8)
    y = torch.randn(8)

    return DataLoader(
        TensorDataset(x, y),
        batch_size=8,
        shuffle=False,
    )


def test_run_epoch_returns_loss_and_mae():
    loader = make_small_loader()
    model = BikeNet()
    loss_fn = nn.MSELoss()

    loss, mae = run_epoch(model, loader, loss_fn)

    assert isinstance(loss, float)
    assert isinstance(mae, float)
    assert loss >= 0
    assert mae >= 0


def test_validation_does_not_update_weights():
    loader = make_small_loader()
    model = BikeNet()
    loss_fn = nn.MSELoss()

    original_weights = [parameter.clone() for parameter in model.parameters()]

    run_epoch(model, loader, loss_fn)

    for before, after in zip(original_weights, model.parameters(), strict=True):
        assert torch.equal(before, after)


def test_training_updates_weights():
    loader = make_small_loader()
    model = BikeNet()
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    original_weights = [parameter.clone() for parameter in model.parameters()]

    run_epoch(model, loader, loss_fn, optimizer)

    assert any(
        not torch.equal(before, after)
        for before, after in zip(original_weights, model.parameters(), strict=True)
    )


def test_small_batch_memorization():
    torch.manual_seed(42)

    x = torch.randn(8, 8)
    y = torch.randn(8)

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=8,
        shuffle=False,
    )

    model = BikeNet()
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    initial_loss, _ = run_epoch(model, loader, loss_fn)

    for _ in range(500):
        run_epoch(model, loader, loss_fn, optimizer)

    final_loss, _ = run_epoch(model, loader, loss_fn)

    assert final_loss < initial_loss
    assert final_loss < 0.01


def test_training_history():
    loader = make_small_loader()

    model, history, best_epoch, best_loss, best_mae = train_model(
        loader,
        loader,
        epochs=3,
        learning_rate=0.001,
    )

    assert len(history) == 3

    for row in history:
        assert "epoch" in row
        assert "train_loss" in row
        assert "val_loss" in row
        assert "val_mae" in row

    assert 1 <= best_epoch <= 3
    assert best_loss == min(row["val_loss"] for row in history)
    assert best_mae >= 0


def test_best_weights_restored():
    loader = make_small_loader()

    model, history, best_epoch, best_loss, best_mae = train_model(
        loader,
        loader,
        epochs=5,
        learning_rate=0.001,
    )

    restored_loss, restored_mae = run_epoch(
        model,
        loader,
        nn.MSELoss(),
    )

    assert abs(restored_loss - best_loss) < 0.0001
    assert abs(restored_mae - best_mae) < 0.0001
