from sklearn.feature_extraction.text import TfidfVectorizer
from src.models.classical import (
    train_linear_svc,
    train_logistic_regression,
    train_naive_bayes,
)
from src.models.rnn import build_gru, build_lstm, build_simple_rnn


def test_train_logistic_regression():
    
    texts = [
        "hello, how are you please call me when you arrived",
        "You can use our product for free",
        "I love you, and I am sorry for everything",
        "send number one to win 10 million dollar prise",
        "you just won a free ticket to china",
        "hello mom, when do you come back home"
    ]
    
    labels = [0, 1, 0, 1, 1, 0]
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    model = train_logistic_regression(X,labels)
    predictions = model.predict(X)
    
    assert len(predictions) == len(labels)
    assert set(predictions).issubset({0, 1})
    
    
def test_train_linear_svc():
    texts = [
        "hello, how are you please call me when you arrived",
        "You can use our product for free",
        "I love you, and I am sorry for everything",
        "send number one to win 10 million dollar prise",
        "you just won a free ticket to china",
        "hello mom, when do you come back home"
    ]
    
    labels = [0, 1, 0, 1, 1, 0]
    
    vectorizer = TfidfVectorizer()
    
    X = vectorizer.fit_transform(texts)
    
    model = train_linear_svc(X, labels)
    
    predictions = model.predict(X)
    
    
    assert len(predictions) == len(labels)
    assert set(predictions).issubset({0, 1})
    
    
    


def test_train_naive_bayes():
    texts = [
        "hello, how are you please call me when you arrived",
        "You can use our product for free",
        "I love you, and I am sorry for everything",
        "send number one to win 10 million dollar prise",
        "you just won a free ticket to china",
        "hello mom, when do you come back home"
    ]
    
    labels = [0, 1, 0, 1, 1, 0]
    
    vectorizer = TfidfVectorizer()
    
    X = vectorizer.fit_transform(texts)
    
    model = train_naive_bayes(X, labels)
    
    predictions = model.predict(X)
    
    assert len(predictions) == len(labels)
    assert set(predictions).issubset({0, 1})
    
    
    



def test_simple_rnn():
    vocabulary_size = 100
    sequence_length = 120

    model = build_simple_rnn(
        vocabulary_size,
        sequence_length,
    )

    assert model.output_shape == (None, 1)


def test_lstm():
    vocabulary_size = 100
    sequence_length = 120

    model = build_lstm(
        vocabulary_size,
        sequence_length,
    )

    assert model.output_shape == (None, 1)


def test_gru():
    vocabulary_size = 100
    sequence_length = 120

    model = build_gru(
        vocabulary_size,
        sequence_length,
    )

    assert model.output_shape == (None, 1)