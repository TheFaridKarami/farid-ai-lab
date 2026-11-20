from pathlib import Path

import numpy as np
from gensim.models import FastText, KeyedVectors, Word2Vec
from keras.layers import TextVectorization


def train_word2vec(
    sentences: list[list[str]],
    window: int,
    vector_size: int,
    min_count: int,
    sg: int,
    epochs: int = 50,
    workers: int = 4,
) -> Word2Vec:
    """Train a Word2Vec model on tokenized sentences."""

    return Word2Vec(
        sentences=sentences,
        window=window,
        vector_size=vector_size,
        min_count=min_count,
        epochs=epochs,
        workers=workers,
        sg=sg,
    )


def create_embedding_matrix(
    vectorizer: TextVectorization,
    embeddings: KeyedVectors,
) -> np.ndarray:
    """Create an embedding matrix aligned with the vectorizer vocabulary."""

    matrix = np.zeros(
        (
            len(vectorizer.get_vocabulary()),
            embeddings.vector_size,
        ),
        dtype=np.float32,
    )

    for i, word in enumerate(vectorizer.get_vocabulary()):
        if embeddings.has_index_for(word):
            matrix[i] = embeddings[word]

    return matrix


def load_glove(path: Path) -> KeyedVectors:
    """Load pre-trained GloVe embeddings as Gensim KeyedVectors."""
    print("glove is being loaded .....")
    return KeyedVectors.load_word2vec_format(
        path,
        binary=False,
        no_header=True,
    )
    
    
def train_fasttext(
    sentences: list[list[str]],
    vector_size: int,
    window:int = 5,
    workers: int = 4,
    min_count : int = 2,
    epochs: int = 50,
    sg: int = 0
    
) -> FastText:
    """Train FastText on tokenized sentences."""
    return FastText(
        sentences=sentences,
        vector_size= vector_size,
        window=window,
        workers=workers,
        min_count= min_count,
        epochs=epochs)
    
    
