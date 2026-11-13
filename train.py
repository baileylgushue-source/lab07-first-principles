import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from torch import nn

from model import BikeNet
from prepare import make_loaders, split_data

FEATURES = [
    "hour",
    "temperature_c",
    "rain_mm",
    "wind_kph",
    "is_weekend",
    "station_type",
]


def run_epoch(model, loader, loss_fn, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_absolute_error = 0.0
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
        total_absolute_error += (predictions - y).abs().sum().item()
        total_rows += len(y)

    mean_loss = total_loss / total_rows
    mean_absolute_error = total_absolute_error / total_rows

    return mean_loss, mean_absolute_error


def train_model(train_loader, val_loader, epochs=100, learning_rate=0.001):
    torch.manual_seed(42)

    model = BikeNet()
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    best_val_mae = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, loss_fn, optimizer)

        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_epoch = epoch
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch}: "
            f"Train loss = {train_loss:.4f}, "
            f"Validation loss = {val_loss:.4f}, "
            f"Validation MAE = {val_mae:.4f}"
        )

    model.load_state_dict(best_weights)

    print("Best epoch by validation loss:", best_epoch)

    return model, history, best_epoch, best_val_loss, best_val_mae


def calculate_errors(predictions, actual):
    squared_errors = [
        (prediction - target) ** 2
        for prediction, target in zip(predictions, actual, strict=True)
    ]

    absolute_errors = [
        abs(prediction - target)
        for prediction, target in zip(predictions, actual, strict=True)
    ]

    mse = sum(squared_errors) / len(squared_errors)
    mae = sum(absolute_errors) / len(absolute_errors)

    return {"mse": float(mse), "mae": float(mae)}


def evaluate_baselines(train_df, val_df, test_df, prep):
    # Training-mean baseline
    mean_rentals = train_df["rentals"].mean()

    mean_val_predictions = [mean_rentals] * len(val_df)
    mean_test_predictions = [mean_rentals] * len(test_df)

    mean_val = calculate_errors(
        mean_val_predictions,
        val_df["rentals"],
    )

    mean_test = calculate_errors(
        mean_test_predictions,
        test_df["rentals"],
    )

    # Linear regression baseline
    x_train = prep.transform(train_df[FEATURES])
    x_val = prep.transform(val_df[FEATURES])
    x_test = prep.transform(test_df[FEATURES])

    y_train = train_df["rentals"]

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)

    linear_val_predictions = linear_model.predict(x_val)
    linear_test_predictions = linear_model.predict(x_test)

    linear_val = calculate_errors(
        linear_val_predictions,
        val_df["rentals"],
    )

    linear_test = calculate_errors(
        linear_test_predictions,
        test_df["rentals"],
    )

    return {
        "training_mean": {
            "validation": mean_val,
            "test": mean_test,
        },
        "linear_regression": {
            "validation": linear_val,
            "test": linear_test,
        },
    }


def save_learning_curves(history, best_epoch, filename):
    frame = pd.DataFrame(history)

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.plot(
        frame["epoch"],
        frame["train_loss"],
        label="Training loss",
    )

    ax.plot(
        frame["epoch"],
        frame["val_loss"],
        label="Validation loss",
    )

    ax.axvline(
        best_epoch,
        color="grey",
        linestyle="--",
        label=f"Best epoch ({best_epoch})",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(42)

    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    train_df, val_df, test_df = split_data(seed=42)

    experiments = [
        {"name": "original", "learning_rate": 0.001},
        {"name": "experiment_1", "learning_rate": 0.003},
        {"name": "experiment_2", "learning_rate": 0.0003},
    ]

    experiment_results = []
    histories = []

    best_model = None
    best_result = None
    best_test_loader = None
    best_prep = None

    for experiment in experiments:
        name = experiment["name"]
        learning_rate = experiment["learning_rate"]

        print(f"\nRunning {name}")
        print(f"Learning rate: {learning_rate}")

        # Reset seed before creating loaders for each run.
        torch.manual_seed(42)

        train_loader, val_loader, test_loader, prep = make_loaders(
            train_df,
            val_df,
            test_df,
            batch_size=64,
        )

        model, history, best_epoch, best_val_loss, best_val_mae = train_model(
            train_loader,
            val_loader,
            epochs=100,
            learning_rate=learning_rate,
        )

        for row in history:
            histories.append(
                {
                    "experiment": name,
                    **row,
                }
            )

        result = {
            "experiment": name,
            "learning_rate": learning_rate,
            "epochs": 100,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_mae": best_val_mae,
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
        }

        experiment_results.append(result)

        if name == "original":
            curve_filename = reports / "learning_curves.png"
        else:
            curve_filename = reports / f"{name}_learning_curves.png"

        save_learning_curves(
            history,
            best_epoch,
            curve_filename,
        )

        # Choose the winning model using validation loss only.
        if best_result is None or best_val_loss < best_result["best_val_loss"]:
            best_result = result
            best_model = model
            best_test_loader = test_loader
            best_prep = prep

    # Save experiment history and comparison.
    pd.DataFrame(histories).to_csv(
        reports / "history.csv",
        index=False,
    )

    results_frame = pd.DataFrame(experiment_results)

    results_frame.to_csv(
        reports / "experiments.csv",
        index=False,
    )

    print("\nExperiment Results")
    print(results_frame.to_string(index=False))

    print("\nSelected Final Network")
    print("Experiment:", best_result["experiment"])
    print("Learning rate:", best_result["learning_rate"])
    print("Best epoch:", best_result["best_epoch"])
    print(
        "Validation MSE:",
        round(best_result["best_val_loss"], 4),
    )
    print(
        "Validation MAE:",
        round(best_result["best_val_mae"], 4),
    )

    # Final test evaluation happens after model selection.
    loss_fn = nn.MSELoss()

    test_mse, test_mae = run_epoch(
        best_model,
        best_test_loader,
        loss_fn,
    )

    # Evaluate both baselines on validation and test rows.
    baseline_results = evaluate_baselines(
        train_df,
        val_df,
        test_df,
        best_prep,
    )

    metrics = {
        "selected_experiment": best_result["experiment"],
        "learning_rate": best_result["learning_rate"],
        "best_epoch": best_result["best_epoch"],
        "neural_network": {
            "validation": {
                "mse": best_result["best_val_loss"],
                "mae": best_result["best_val_mae"],
            },
            "test": {
                "mse": test_mse,
                "mae": test_mae,
            },
        },
        "baselines": baseline_results,
    }

    with open(reports / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    print("\nFinal Model Test Results")
    print("Test MSE:", round(test_mse, 4))
    print("Test MAE:", round(test_mae, 4))

    print("\nBaseline Results")
    for name, values in baseline_results.items():
        print(f"\n{name}")
        print(
            "Validation MAE:",
            round(values["validation"]["mae"], 4),
        )
        print(
            "Test MAE:",
            round(values["test"]["mae"], 4),
        )

    print("\nSaved all reports in reports/")


if __name__ == "__main__":
    main()
