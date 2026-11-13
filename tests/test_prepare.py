import numpy as np
import torch

from prepare import (
    BikeDataset,
    make_loaders,
    prepare_features,
    split_data,
)


def test_split_sizes():
    train_df, val_df, test_df = split_data(seed=42)

    assert len(train_df) == 3600
    assert len(val_df) == 1200
    assert len(test_df) == 1200


def test_splits_do_not_overlap():
    train_df, val_df, test_df = split_data(seed=42)

    train_indices = set(train_df.index)
    val_indices = set(val_df.index)
    test_indices = set(test_df.index)

    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)


def test_split_is_reproducible():
    train_1, val_1, test_1 = split_data(seed=42)
    train_2, val_2, test_2 = split_data(seed=42)

    assert train_1.index.equals(train_2.index)
    assert val_1.index.equals(val_2.index)
    assert test_1.index.equals(test_2.index)


def test_preprocessor_output_shapes():
    train_df, val_df, test_df = split_data(seed=42)

    x_train, x_val, x_test, prep = prepare_features(train_df, val_df, test_df)

    assert x_train.shape[0] == len(train_df)
    assert x_val.shape[0] == len(val_df)
    assert x_test.shape[0] == len(test_df)

    assert x_train.shape[1] == x_val.shape[1]
    assert x_train.shape[1] == x_test.shape[1]


def test_preprocessor_fitted_on_training_data():
    train_df, val_df, test_df = split_data(seed=42)

    x_train, x_val, x_test, prep = prepare_features(train_df, val_df, test_df)

    scaler = prep.named_transformers_["num"]

    expected_means = (
        train_df[
            [
                "hour",
                "temperature_c",
                "rain_mm",
                "wind_kph",
                "is_weekend",
            ]
        ]
        .mean()
        .to_numpy()
    )

    np.testing.assert_allclose(scaler.mean_, expected_means)


def test_dataset_length():
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    labels = np.array([10.0, 20.0, 30.0])

    dataset = BikeDataset(features, labels)

    assert len(dataset) == 3


def test_dataset_shapes():
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([10.0, 20.0])

    dataset = BikeDataset(features, labels)

    x, y = dataset[0]

    assert x.shape == (2,)
    assert y.shape == torch.Size([])


def test_dataset_float32():
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([10.0, 20.0])

    dataset = BikeDataset(features, labels)

    x, y = dataset[0]

    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_loader_batch_shapes():
    train_df, val_df, test_df = split_data(seed=42)

    train_loader, val_loader, test_loader, prep = make_loaders(
        train_df, val_df, test_df, batch_size=64
    )

    x, y = next(iter(train_loader))

    assert x.shape[0] == 64
    assert y.shape[0] == 64
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_seeded_training_shuffle():
    train_df, val_df, test_df = split_data(seed=42)

    train_loader_1, _, _, _ = make_loaders(
        train_df, val_df, test_df, batch_size=64, seed=42
    )

    train_loader_2, _, _, _ = make_loaders(
        train_df, val_df, test_df, batch_size=64, seed=42
    )

    for (x1, y1), (x2, y2) in zip(train_loader_1, train_loader_2, strict=True):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)
