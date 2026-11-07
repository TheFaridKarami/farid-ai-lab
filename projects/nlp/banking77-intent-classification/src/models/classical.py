from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "linear_svc": LinearSVC,
    "complement_nb": ComplementNB,
    "multinomial_nb": MultinomialNB,
    "sgd_classifier": SGDClassifier,
}


class ClassicalModel:
    def __init__(self, config: dict, method: str) -> None:
        if method not in MODEL_CLASSES:
            raise ValueError(
                f"Unsupported method: {method!r}. "
                f"Accepted methods: {sorted(MODEL_CLASSES)}."
            )
        self.method = method
        self.config = config
        self.model = self._create_model()

    def _create_model(self):
        model_class = MODEL_CLASSES[self.method]
        params = self.config[self.method]
        return model_class(**params)

    def fit(self, features, labels):
        self.model.fit(features, labels)
        return self

    def predict(self, features):
        return self.model.predict(features)