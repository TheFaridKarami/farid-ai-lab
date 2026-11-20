from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.config import DATASET_NAME, RAW_DATA_DIR
from src.data import load_data, remove_duplicates, split_data, validate_data
from src.features import (
    create_tfidf_vectorizer,
    fit_tfidf_vectorizer,
    transform_tfidf,
)
from src.models.classical import (
    train_linear_svc,
    train_logistic_regression,
    train_naive_bayes,
)


def evaluate_model(
    model,
    X,
    y,
    model_name: str,
) -> None:
    """Evaluate a trained classifier."""

    predictions = model.predict(X)

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy:  {accuracy_score(y, predictions):.4f}")
    print(
        f"Precision: "
        f"{precision_score(y, predictions, pos_label='spam'):.4f}"
    )
    print(
        f"Recall:    "
        f"{recall_score(y, predictions, pos_label='spam'):.4f}"
    )
    print(
        f"F1 Score:  "
        f"{f1_score(y, predictions, pos_label='spam'):.4f}"
    )

    print("\nClassification Report:")
    print(classification_report(y, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))





def main() -> None:
    """Run the classical SMS spam classification pipeline."""

    dataset_path = RAW_DATA_DIR / DATASET_NAME

    df = load_data(dataset_path)
    validate_data(df)

    print(f"Original dataset size: {len(df)}")

    df = remove_duplicates(df)

    print(f"Dataset size after duplicate removal: {len(df)}")

    splits = split_data(df)

    print("\nDataset splits:")
    print(f"Train:      {len(splits.X_train)}")
    print(f"Validation: {len(splits.X_val)}")
    print(f"Test:       {len(splits.X_test)}")

    vectorizer = create_tfidf_vectorizer()

    fit_tfidf_vectorizer(
        vectorizer,
        splits.X_train,
    )

    X_train = transform_tfidf(
        vectorizer,
        splits.X_train,
    )
    
    X_val =transform_tfidf(
        vectorizer,
        splits.X_val
    )

    X_test = transform_tfidf(
        vectorizer,
        splits.X_test,
    )

    print(f"\nTraining TF-IDF shape: {X_train.shape}")
    print(f"Test TF-IDF shape:     {X_test.shape}")
    print(f"X validation TF-IDF shape: {X_val.shape}")

    logistic_model = train_logistic_regression(
        X_train,
        splits.y_train,
    )

    svc_model = train_linear_svc(
        X_train,
        splits.y_train,
    )

    naive_bayes_model = train_naive_bayes(
        X_train,
        splits.y_train,
    )

    evaluate_model(
        logistic_model,
        X_val,
        splits.y_val,
        "Logistic Regression",
    )

    evaluate_model(
        svc_model,
        X_val,
        splits.y_val,
        "Linear SVC",
    )

    evaluate_model(
        naive_bayes_model,
        X_val,
        splits.y_val,
        "Naive Bayes",
    )
    evaluate_model(
        svc_model,
        X_test,
        splits.y_test,
        "Linear SVC — Final Test Evaluation",
    )

if __name__ == "__main__":
    main()