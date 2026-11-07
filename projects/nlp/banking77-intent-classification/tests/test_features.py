from configs import load_config
from features.text import TextVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_feature_config():
    config = load_config("configs/model.yaml")
    return config["classical"]["features"]

def test_text_vectorizer():
    config = get_feature_config()

    vectorizer = TextVectorizer(method="bow", config=config)

    assert isinstance(vectorizer.vectorizer, CountVectorizer)


def test_text_tfidf():
    config = get_feature_config()

    vectorizer = TextVectorizer(method="tfidf", config=config)

    assert isinstance(vectorizer.vectorizer, TfidfVectorizer)
    
    
def test_fit_transform():
    config = get_feature_config()
    vectorizer = TextVectorizer(config = config, method=  "bow")
    texts = [
    "I want to check my bank account balance",
    "How can I check my bank account balance",
    "I need to transfer money to another account",
    "How do I transfer money to another bank account",
    "My credit card payment was declined",
    "Why was my credit card payment declined",
    "I forgot my banking PIN",
    "How can I reset my banking PIN",
    "I want to cancel a scheduled payment",
    "How do I cancel a scheduled payment",
]
    
    features = vectorizer.fit_transform(texts)
    
    assert features.shape[0] ==10
    assert features.shape[1] > 0
    
    

def test_transform():
    config = get_feature_config()

    vectorizer = TextVectorizer(config=config, method="bow")

    train_texts = ["hello world", "hello banking"]
    test_texts = ["hello world"]

    vectorizer.fit_transform(train_texts)
    features = vectorizer.transform(test_texts)

    assert features.shape[0] == 1
    assert features.shape[1] > 0
    
    
    
def test_config():
    path = r"configs/model.yaml"
    config = load_config(path)
    assert "classical" in config
    assert "neural" in config
    assert "transformer" in config