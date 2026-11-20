import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def create_tfidf_vectorizer() -> TfidfVectorizer:
    """Create a TF-IDF vectorizer."""

    return TfidfVectorizer()


def fit_tfidf_vectorizer(
    vectorizer: TfidfVectorizer,
    texts: pd.Series,
) -> None:
    """Fit the TF-IDF vectorizer on training text."""

    vectorizer.fit(texts)


def transform_tfidf(
    vectorizer: TfidfVectorizer,
    texts: pd.Series,
):
    """Transform text into TF-IDF features."""

    return vectorizer.transform(texts)


def create_bow_vectorizer() -> CountVectorizer:
    """Create a Bag-of-Words vectorizer."""

    return CountVectorizer()


def fit_bow_vectorizer(
    vectorizer: CountVectorizer,
    texts: pd.Series,
) -> None:
    """Fit the Bag-of-Words vectorizer on training text."""

    vectorizer.fit(texts)
    
    
def transform_bow(
    vectorizer: CountVectorizer,
    texts: pd.Series,
):
    """Transform text into Bag-of-Words features."""

    return vectorizer.transform(texts)