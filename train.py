
import torch
from torch import nn

from model import BikeNet
from prepare import make_loaders, split_data


def run_epoch(model, loader, loss_fn, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_rows = 0

    for x, y in loader:
        if optimizer is None:
            with torch.no_grad():
                predictions = model(x)
                loss = loss_fn(predictions, y)
        else:
            predictions = model(x)
            loss = loss_fn(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(y)
        total_rows += len(y)

    return total_loss / total_rows


def train_model(train_loader, val_loader, epochs=100):
    torch.manual_seed(42)

    model = BikeNet()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model, train_loader, loss_fn, optimizer
        )

        val_loss = run_epoch(
            model, val_loader, loss_fn
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {
                k: v.clone() for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch}: "
            f"Train loss = {train_loss:.4f}, "
            f"Validation loss = {val_loss:.4f}"
        )

    model.load_state_dict(best_weights)

    return model


def main():
    torch.manual_seed(42)

    train_df, val_df, test_df = split_data(seed=42)

    train_loader, val_loader, test_loader, prep = make_loaders(
        train_df, val_df, test_df, batch_size=64
    )

    model = train_model(
        train_loader,
        val_loader,
        epochs=100,
    )


if __name__ == "__main__":
    main()
