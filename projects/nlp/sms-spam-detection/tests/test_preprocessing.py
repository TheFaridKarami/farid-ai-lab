import numpy as np
import pandas as pd
import pytest
from src.preprocessing import (
    create_vectorizer,
    fit_vectorizer,
    validate_non_empty_sequences,
    validate_right_padding,
    vectorize_text,
)


def test_create_vectorizer():
    
    vectorizer = create_vectorizer(
        max_tokens=10_000,
        sequence_length=120,
        standardize=None,
    )

    text = pd.Series(["hello world"])
    fit_vectorizer(vectorizer, text)

    sequence = vectorize_text(vectorizer, text)

    assert sequence.shape == (1, 120)
    
    
    
def test_fit_vectorizer_uses_training_text():
    
    train_test = pd.Series([
        "Hello world",
        "I love Machine learning",
        "I have to save my ass"
    ])
    
    
    vectorizer = create_vectorizer(
        max_tokens= 100,
        sequence_length= 10,
    )
    
    fit_vectorizer(vectorizer, train_test)
    
    vocabulary = vectorizer.get_vocabulary()
    
    assert "Hello" in vocabulary
    assert "Machine" in vocabulary
    assert "makes" not in vocabulary
    
    
    

def test_vectorize_text_returns_fixed_length_integer_array():
    
    train_test = pd.Series([
        "Hello world",
        "I love Machine learning",
        "I have to save my ass"
    ])
    
    vectorizer = create_vectorizer(
        max_tokens=100,
        sequence_length= 10,
    )
    
    fit_vectorizer(vectorizer,train_test)
    
    sequence = vectorize_text(vectorizer, train_test)
    
    assert isinstance(sequence, np.ndarray)
    assert sequence.shape ==(3, 10)
    assert np.issubdtype(sequence.dtype, np.integer)
 
 
 
 
 
def test_validate_right_padding_accepts_valid_sequences():
    
    sequences = np.array([
        [3,4,6,0,0,0,0],
        [1,8,6,3,4,0,0]
    ])
    
    validate_right_padding(sequences)
    
    
    

def test_validate_right_padding_rejects_invalid_sequences():
    
    sequences = np.array([
        [3,4,0,0,34,1],
        [2,0,0,0,1,12]
    ])
    with pytest.raises(ValueError):
       validate_right_padding(sequences)
       
       


def test_validate_non_empty_sequences_accepts_valid_sequences():
    
    sequences = np.array(
        [
            [1, 2, 0, 0],
            [3, 4, 5, 0],
        ]
    )
    
    validate_non_empty_sequences(sequences)
    
    
    
def test_validate_non_empty_sequences_rejects_empty_sequences():
    
    sequences = np.array(
        [
            [0, 0, 0, 0, 0],
            [2, 4, 5, 0, 2],
        ]
    )
    
    with pytest.raises(ValueError):
        validate_non_empty_sequences(sequences)
