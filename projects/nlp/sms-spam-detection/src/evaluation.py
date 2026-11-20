import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classical_model(
    model,
    X_test,
    y_test,
    model_name: str,
):
    prediction = model.predict(X_test)

    positive_label = "spam" if "spam" in np.unique(y_test) else 1
    
    accuracy = accuracy_score(y_test, prediction)
    
    
    precision = precision_score(
        y_test,
        prediction,
        pos_label=positive_label,
    )
    recall = recall_score(
        y_test,
        prediction,
        pos_label=positive_label,
    )
    f1 = f1_score(
        y_test,
        prediction,
        pos_label=positive_label,
    )

    report = classification_report(y_test, prediction)
    report_cm = confusion_matrix(y_test, prediction)

    print(f"{'=' * 20} {model_name} {'=' * 20}")
    print(f"Classification report:\n{report}")
    print(f"Confusion Matrix:\n{report_cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"Recall score: {recall:.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        report_cm,
        annot=True,
        fmt="d",
        cbar=True,
        cmap="Blues",
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_nlp_model(
    model,
    X_test,
    y_test,
    model_name: str,
):
    probabilities = model.predict(X_test)

    prediction = (
        probabilities.ravel() >= 0.5
    ).astype(int)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(
        y_test,
        prediction,
        pos_label=1,
    )
    recall = recall_score(
        y_test,
        prediction,
        pos_label=1,
    )
    f1 = f1_score(
        y_test,
        prediction,
        pos_label=1,
    )

    report = classification_report(y_test, prediction)
    report_cm = confusion_matrix(y_test, prediction)

    print(f"{'=' * 20} {model_name} {'=' * 20}")
    print(f"Classification report:\n{report}")
    print(f"Confusion Matrix:\n{report_cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"Recall score: {recall:.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        report_cm,
        annot=True,
        fmt="d",
        cbar=True,
        cmap="Blues",
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_loss(
    history,
    model_name: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )
    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )
    plt.title(
        f"{model_name} - Training and Validation Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.7,
    )
    plt.show()

