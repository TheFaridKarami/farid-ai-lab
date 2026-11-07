import pandas as pd
import pytest
from data.data import (
    load_data,
    normalize_text,
    prepare_data,
    remove_duplicate_rows,
    split_data,
    validate_data,
)


def test_load_data():
    train_df, test_df = load_data()
    
    assert not train_df.empty
    assert not test_df.empty
    assert list(train_df.columns) == ["text", "label"]
    assert list(test_df.columns) == ["text", "label"]





def test_normalize_text():
    df = pd.DataFrame(
        {
            "text": [
                "  Hello this is a simple test  ",
                "How can I help you with this task ma'am ",
                " where are you kiddo ",
            ],
            "label": ["intent_a", "intent_b", "intent_c"],
        }
    )

    result = normalize_text(df)

    assert result["text"].tolist() == [
        "Hello this is a simple test",
        "How can I help you with this task ma'am",
        "where are you kiddo",
    ]


    assert df["text"].iloc[0] == "  Hello this is a simple test  "
    
    



def test_remove_duplicates():
   df = pd.DataFrame(
        {
            "text": [
                "Check my balance",
                "Check my balance",
                "Transfer money",
            ],
            "label": [
                "check_balance",
                "check_balance",
                "cash_withdrawal",
            ],
        }
    )
    
   result = remove_duplicate_rows(df)
   
   assert len(result) == 2
   assert result.duplicated().sum() == 0
   assert len(df) == 3
   
   
   


def test_split_data():
    
    df = pd.DataFrame({
        
        "text":[f"Example message {i}" for i in range(100)],
        
        "label":["balance"] * 50 + ["cash_withdrawal"] * 50
        
    })
    
    
    X_train, X_val, y_train, y_val = split_data(df)
    
    assert len(X_train) == 90
    assert len(X_val) == 10
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    
    assert set(y_train.unique()) == {"balance", "cash_withdrawal"}
    assert set(y_val.unique()) == {"balance", "cash_withdrawal"}
    
    assert set(X_train).isdisjoint(set(X_val))
    
    
    

def test_prepare_data():
    
    train_df, test_df = prepare_data()
    
    assert not train_df.empty
    assert not test_df.empty
    
    assert train_df["text"].str.strip().equals(train_df["text"])
    assert test_df["text"].str.strip().equals(test_df["text"])

    assert not train_df.duplicated().any()
    
    
    
def test_validate_data_missing_values():
    
       df = pd.DataFrame(
        {
            "text": ["Check my balance", None],
            "label": ["check_balance", "cash_withdrawal"],
        }
    )
       
       try:
            validate_data(df)
       except ValueError as error:
           
            assert str(error) == "Dataset contains missing values."
       else:
            raise AssertionError("Expected ValueError for missing values.")



def test_validate_data_empty_text():
    
    df = pd.DataFrame(
        {
            "text": ["Check my balance", "   "],
            "label": ["check_balance", "cash_withdrawal"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Dataset contains empty text.",
    ):
        validate_data(df)