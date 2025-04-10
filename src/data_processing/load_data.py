import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from src.utils.knn_graph import knn_edge


def load_chameleon(csv_path, k=8):
    df = pd.read_csv(csv_path)

    sex = OneHotEncoder(sparse_output=False).fit_transform(df[['Type']])

    column = df.columns.drop(['Type', 'Rings'])
    feature = StandardScaler().fit_transform(df[column])

    features = np.concatenate([sex, feature], axis=1)
    x = torch.tensor(features, dtype=torch.float32)

    y_raw = df['Rings'].values
    y = torch.tensor(pd.cut(y_raw, bins=[0, 5, 10, 30], labels=[0, 1, 2]).astype(int), dtype=torch.long)

    edge_index = knn_edge(x, k=k)
    i = np.arange(len(y))

    # test 20%, validation 20%, train 60%
    train_idx, test_idx = train_test_split(i, test_size=0.2, random_state=42)
    train_idx, validation_idx = train_test_split(train_idx, test_size=0.25, random_state=42)

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    validation_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[train_idx] = True
    validation_mask[validation_idx] = True
    test_mask[test_idx] = True

    return {
        "x": x,
        "y": y,
        "edge_index": edge_index,
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "test_mask": test_mask
    }
