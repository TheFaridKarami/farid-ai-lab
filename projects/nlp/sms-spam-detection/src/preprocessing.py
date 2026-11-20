import keras
import numpy as np
import pandas as pd


def create_vectorizer(
    max_tokens: int,
    sequence_length: int,
    standardize: str | None = None,
) -> keras.layers.TextVectorization:

    """Create a Keras TextVectorization layer."""

    return keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=standardize,
        output_mode="int",
        output_sequence_length=sequence_length,
    )


def fit_vectorizer(
    vectorizer: keras.layers.TextVectorization,
    text: pd.Series,
) -> None:

    """Build the vectorizer vocabulary from training text."""

    vectorizer.adapt(text)
    


def vectorize_text(
    vectorizer: keras.layers.TextVectorization,
    text: pd.Series,
)->np.ndarray:
    """Convert text into integer token sequences."""

    return vectorizer(text).numpy()



def validate_right_padding(sequences) -> None:
    """Validate that padding is right-aligned in every sequence."""

    padding = sequences == 0

    invalid_padding = np.any(
        padding[:, :-1] & ~padding[:, 1:],
        axis=1,
    )

    if invalid_padding.any():
        raise ValueError("Invalid padding detected: non-zero token after padding.")



def validate_non_empty_sequences(sequences) -> None:
    """Validate that no sequence consists entirely of padding."""

    empty_sequences = np.all(sequences == 0, axis=1)

    if empty_sequences.any():
        raise ValueError("Empty sequences detected.")
    

