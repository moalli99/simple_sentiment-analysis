import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

from utils.processing import tockenize, padding, process_


def get_loaders(
    file_path,
    vocab_size,
    seq_len,
    batch_size,
    temp_size,
    test_size,
    random_state
):
    # Read dataset
    data = pd.read_csv(file_path)
    X, y = data['review'], data['sentiment']

    # Train-test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size, stratify=y_temp, random_state=random_state
    )


    # Tokenize
    X_train_tok, y_train_enc, X_val_tok, y_val_enc, X_test_tok, y_test_enc, word_dic = tockenize(
        X_train, y_train, X_val, y_val, X_test, y_test, vocab_size
    )

    # Padding
    X_train_pad = padding(X_train_tok, seq_len)
    X_val_pad = padding(X_val_tok, seq_len)
    X_test_pad = padding(X_test_tok, seq_len)

    # TensorDatasets
    train_data = TensorDataset(torch.tensor(X_train_pad), torch.tensor(y_train_enc))
    val_data = TensorDataset(torch.tensor(X_val_pad), torch.tensor(y_val_enc))
    test_data = TensorDataset(torch.tensor(X_test_pad), torch.tensor(y_test_enc))

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, word_dic

def get_data_sklearn(
    file_path="data/IMDB Dataset.csv",
    test_size=0.2,
    random_state=42
):
    # Read dataset
    data = pd.read_csv(file_path)
    X, y = data['review'], data['sentiment']

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Preprocess text
    X_train=X_train.apply(process_)
    X_val=X_val.apply(process_)

    y_train = [1 if label == 'positive' else 0 for label in y_train]
    y_val = [1 if label == 'positive' else 0 for label in y_val]


    return X_train, y_train, X_val, y_val
