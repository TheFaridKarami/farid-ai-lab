from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    LABEL_COLUMN,
    SEED,
    TEST_SIZE,
    TEXT_COLUMN,
    VALIDATION_SIZE,
)


def load_data(path: Path) -> pd.DataFrame:
    
    
    return pd.read_csv(
        path,
        sep ="\t",
        header = None,
        names = [LABEL_COLUMN, TEXT_COLUMN]
    )
    
    
def validate_data(df: pd.DataFrame) -> None:
    required_columns = {LABEL_COLUMN, TEXT_COLUMN}
    
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing required columns: {required_columns - set(df.columns)}"
        )

    if df[TEXT_COLUMN].isna().any():
        raise ValueError(
            f"Missing values found in '{TEXT_COLUMN}'."
        )
    
    valid_labels = {"ham", "spam"}
    
    if not df[LABEL_COLUMN].isin(valid_labels).all():
        raise ValueError("Dataset contains unexpected labels.")
    
    
    
    
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    
    """Remove duplicate SMS messages."""
    
    return df.drop_duplicates().reset_index(drop=True)
  
  
@dataclass

class DataSplits:
    
    X_train: pd.Series
    X_val: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    
    
    
def split_data(df: pd.DataFrame) -> DataSplits:
    texts = df[TEXT_COLUMN]
    labels = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=SEED,
    )

    validation_ratio = VALIDATION_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_ratio,
        stratify=y_train,
        random_state=SEED,
    )

    return DataSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
 
 