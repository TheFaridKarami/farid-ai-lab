import keras
import tensorflow as tf

from src.config import MODEL_DIR


def load_models():
    transformer = keras.models.load_model(
        MODEL_DIR / "transformer.keras"
    )

    vectorizer = keras.models.load_model(
        MODEL_DIR / "text_vectorizer.keras"
    )

    return transformer, vectorizer



def predict_messages(messages, transformer, vectorizer):
    
    """Return spam predictions and probabilities for SMS messages."""
    
    sequence = vectorizer(tf.constant(messages)).numpy()
    
    probabilities = transformer.predict(sequence, verbose= 0).ravel()
    predictions = ["spam" if p >= 0.5 else "ham" for p in probabilities]
    
    return  predictions, probabilities
    