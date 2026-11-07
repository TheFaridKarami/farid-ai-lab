from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"


def load_data():
    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    return train_df, test_df



def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].str.strip()

    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()



def validate_data(df:pd.DataFrame):
    required_columns = {"text", "label"}
    
    if set(df.columns) != required_columns:
        
        raise ValueError(
            
            f"Expected columns {required_columns}, "
            f"but received {set(df.columns)}"
        )

    
    if df[["text", "label"]].isna().any().any():
        raise ValueError("Dataset contains missing values.")
    
    
    if df["text"].str.strip().eq("").any():
        raise ValueError("Dataset contains empty text.")
    
    
    if df["label"].str.strip().eq("").any():
        raise ValueError("Dataset contains empty labels.")
    
    
    


def split_data(
    df: pd.DataFrame,
    validation_size: float = 0.1,
    random_state: int = 42,
    
)  -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    X = df["text"]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_val, y_train, y_val



def prepare_data():
    
    train_df, test_df = load_data()
    
    validate_data(train_df)
    validate_data(test_df)

    train_df = normalize_text(train_df)
    test_df = normalize_text(test_df)

    train_df = remove_duplicate_rows(train_df)
    
    validate_data(train_df)
    validate_data(test_df)


    return train_df, test_df



# -----------------------------------------------------------------------------------------------



