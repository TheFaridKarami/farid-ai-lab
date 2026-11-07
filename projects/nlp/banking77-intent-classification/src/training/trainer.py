from features.text import TextVectorizer
from models.classical import ClassicalModel


class Trainer:
    
    def __init__(self, config: dict) -> None:
        self.config = config["classical"]

    def extract_features(
        self,
        X_train,
        X_val,
        method: str,
    ):
        vectorizer = TextVectorizer(
            config=self.config["features"],
            method=method,
        )

        X_train_features = vectorizer.vectorizer.fit_transform(X_train)
        X_val_features = vectorizer.vectorizer.transform(X_val)

        return vectorizer, X_train_features, X_val_features

    def train_model(
        
        self,
        X_train_features,
        y_train,
        method: str,
        
    ):
        model = ClassicalModel(
            config=self.config["models"],
            method=method,
        )

        model.model.fit(X_train_features, y_train)

        return model
    



