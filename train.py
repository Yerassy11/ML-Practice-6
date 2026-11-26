
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


MODEL_PATH = Path(__file__).with_name("model.joblib")


def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("wine-classification")

    params = {"n_estimators": 100, "random_state": 42, "test_size": 0.2}
    wine = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        wine.data,
        wine.target,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=wine.target,
    )

    with mlflow.start_run(run_name="random-forest") as run:
        mlflow.log_params(params)

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            random_state=params["random_state"],
        )
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_macro": f1_score(y_test, predictions, average="macro"),
        }
        mlflow.log_metrics(metrics)

        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=x_test[:1],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        registered_model = mlflow.register_model(
            model_info.model_uri, "wine-classifier"
        )

        print(f"Test accuracy: {metrics['accuracy']:.3f}")
        print(f"Test F1 (macro): {metrics['f1_macro']:.3f}")
        print(f"Saved model to {MODEL_PATH}")
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Registered model: {registered_model.name}, version: {registered_model.version}")


if __name__ == "__main__":
    main()
