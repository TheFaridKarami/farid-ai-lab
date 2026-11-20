import argparse

from src.inference import load_models, predict_messages


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether an SMS message is spam."
    )

    parser.add_argument(
        "message",
        type=str,
        help="The SMS message to classify.",
    )

    args = parser.parse_args()

    transformer, vectorizer = load_models()

    predictions, probabilities = predict_messages(
        [args.message],
        transformer,
        vectorizer,
    )

    print(f"Prediction: {predictions[0]}")
    print(f"Spam probability: {probabilities[0]:.2%}")


if __name__ == "__main__":
    main()