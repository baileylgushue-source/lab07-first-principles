import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import make_rentals

def split_data(seed: int = 42):
    frame = make_rentals()

    train_df, rest = train_test_split(
        frame,
        test_size=0.4,
        random_state=seed,
    )

    val_df, test_df = train_test_split(
        rest,
        test_size=0.5,
        random_state=seed,
    )

    return train_df, val_df, test_df

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )

NUMERIC = [
    "hour",
    "temperature_c",
    "rain_mm",
    "wind_kph",
    "is_weekend",
]

CATEGORICAL = ["station_type"]

TARGET = "rentals"


def prepare_features(train_df, val_df, test_df):
    prep = make_preprocessor()

    prep.fit(train_df[NUMERIC + CATEGORICAL])

    X_train = prep.transform(train_df[NUMERIC + CATEGORICAL])
    X_val = prep.transform(val_df[NUMERIC + CATEGORICAL])
    X_test = prep.transform(test_df[NUMERIC + CATEGORICAL])

    return X_train, X_val, X_test, prep


class BikeDataset(Dataset):
    """Holds bike rental features and targets as tensors."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

def make_loaders(train_df, val_df, test_df, batch_size=64):
    X_train, X_val, X_test, prep = prepare_features(
        train_df, val_df, test_df
    )

    train_dataset = BikeDataset(X_train, train_df[TARGET].to_numpy())
    val_dataset = BikeDataset(X_val, val_df[TARGET].to_numpy())
    test_dataset = BikeDataset(X_test, test_df[TARGET].to_numpy())

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, prep  