from configs import load_config
from data.data import prepare_data, split_data
from evaluation.evaluator import ClassificationEvaluator
from training.trainer import Trainer


def main():
    config = load_config(r"configs/model.yaml")

    train_df, _ = prepare_data()
    X_train, X_val, y_train, y_val = split_data(train_df)

    features = ["bow", "tfidf"]
    
    methods = [
        "logistic_regression",
        "linear_svc",
        "multinomial_nb",
        "complement_nb",
        "sgd_classifier",
    ]

    trainer = Trainer(config)
    evaluator = ClassificationEvaluator()

    final_report = {}

    for feature in features:
        
        final_report[feature] = {}

        _, X_train_features, X_val_features = trainer.extract_features(
            X_train,
            X_val,
            method=feature,
        )

        for method in methods:
            model = trainer.train_model(
                X_train_features,
                y_train,
                method=method,
            )

            predictions = model.predict(X_val_features)

            metrics = evaluator.compute_metrics(y_val, predictions)

            final_report[feature][method] = metrics

    print("\n" + "=" * 75)
    print("EXPERIMENT SUMMARY")
    print("=" * 75)

    print(
        f"{'Feature':<10}"
        f"{'Model':<25}"
        f"{'Accuracy':>12}"
        f"{'Macro F1':>12}"
        f"{'Weighted F1':>16}"
    )

    print("-" * 75)

    for feature, models in final_report.items():
        
        for method, metrics in models.items():
            
            print(
                f"{feature:<10}"
                f"{method:<25}"
                f"{metrics['accuracy']:>12.4f}"
                f"{metrics['macro_f1']:>12.4f}"
                f"{metrics['weighted_f1']:>16.4f}"
            )


    return final_report




if __name__ == "__main__":
    main()
    
    




