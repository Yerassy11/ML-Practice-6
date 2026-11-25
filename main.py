from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Wine ML API")

MODEL_PATH = Path(__file__).with_name("model.joblib")
model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        min_length=13,
        max_length=13,
        examples=[
            [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]
        ],
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(data: PredictionRequest) -> dict[str, int]:
    # The model expects a list of samples, so wrap the features in another list.
    prediction = model.predict([data.features])[0]
    return {"prediction": int(prediction)}
