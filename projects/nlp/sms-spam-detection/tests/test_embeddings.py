from keras.layers import TextVectorization
from src.embeddings import (
    create_embedding_matrix,
    load_glove,
    train_fasttext,
    train_word2vec,
)
from src.config import GLOVE_PATH


def test_create_embedding_matrix():
    sentences = [
        ["I", "Need", "Help", "Please"],
        ["I", "Do", "Not", "like", "masturbation"],
        ["How", "can", "I", "quit", "it"]
    ]
    
    model =train_word2vec(
        sentences,
        window=5, 
        min_count=2,
        vector_size=64,
        epochs=50,
        workers=4,
        sg = 0,
    )
    
    assert len(model.wv) == 1
    assert model.wv.vector_size == 64



def test_train_word2vec():
    
    sentences = [
        ["Hello", "Farid"],
        ["Hello", "USA"]
    ]
    
    
    model = train_word2vec(
        sentences=sentences,
        window=2,
        vector_size=20,
        min_count=1,
        sg= 0,
        epochs=50,
        workers= 4
    )
    
    
    assert len(model.wv) == 3
    assert model.wv.vector_size == 20
    
    
def test_load_glove():
    data = load_glove(GLOVE_PATH)
    
    assert data is not None
    
    
def test_train_fasttext():
    sentences = [
        ["hello", "welcome", "to", "the", "use"],
        ["This", "is", "where", "you", "can"],
        ["start", "a", "better", "life", "forever"]
    ]
    
    model  = train_fasttext(
        sentences=sentences,
        window=2,
        min_count=1,
        vector_size=64,
        sg = 1
    )
    
    
    assert model.wv.vector_size == 64
    assert len(model.wv) == 15