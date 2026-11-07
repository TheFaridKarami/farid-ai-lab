import keras


class SequenceTokenizer:
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        output_mode: str = "int",
    ) -> None:
        self.vectorizer = keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode=output_mode,
            output_sequence_length=sequence_length,
        )

    def fit(self, text):
        self.vectorizer.adapt(text)

    def get_vocabulary(self):
        return self.vectorizer.get_vocabulary()

    def transform(self, text):
        return self.vectorizer(text)
