
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


MODEL_PATH = Path(__file__).with_name("model.joblib")


def main() -> None:
    wine = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        wine.data,
        wine.target,
        test_size=0.2,
        random_state=42,
        stratify=wine.target,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
