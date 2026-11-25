# Wine ML API

An educational project for training a classification model and serving it with FastAPI and Docker.
The `RandomForestClassifier` model predicts a wine class (`0`, `1`, or `2`) from 13 numerical features.

The project uses scikit-learn's built-in Wine dataset: 178 samples and 3 classes.
No separate dataset download is needed. The data is split into 80% for training and 20% for testing.

## Project Files

- `train.py` — trains, evaluates, and saves the model.
- `model.joblib` — the saved model.
- `main.py` — the API with `GET /` and `POST /predict` endpoints.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — instructions for building the image and running the container.
- `.dockerignore` — files excluded from the Docker build context.

## Local Setup and Run

Python 3.12 or later is required. Run the following commands in PowerShell
from the project directory. To create the environment, the `python` command
must point to an installed Python interpreter.

If the `.venv` environment has not been created yet:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train the model and save it to `model.joblib`:

```powershell
.\.venv\Scripts\python.exe train.py
```

The script prints `Test accuracy`, the proportion of correct predictions on the test set.
If `model.joblib` is missing, train the model before starting the API.

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Swagger: http://127.0.0.1:8000/docs

Press `Ctrl+C` to stop the server.

## Run with Docker

Start Docker Desktop. Make sure `model.joblib` exists in the project directory.

Build the image:

```powershell
docker build -t wine-ml-api .
```

Run the container:

```powershell
docker run --rm -p 8001:8000 --name wine-api wine-ml-api
```

Swagger in Docker: http://127.0.0.1:8001/docs

Port `8001` on your computer is mapped to port `8000` inside the container.
To stop the container, press `Ctrl+C` or run this command in another terminal:

```powershell
docker stop wine-api
```

## Test the API with Swagger

1. Open the `/docs` page for your chosen setup (local or Docker).
2. Expand `GET /`, click **Try it out**, then **Execute**.
   Expect status code `200` and the response `{"message": "ML API is running"}`.
3. Expand `POST /predict`, click **Try it out**, keep the example request,
   and click **Execute**. Expect status code `200` and a `prediction` field containing the wine class.

Example request body for `POST /predict`:

```json
{
  "features": [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]
}
```

Example response:

```json
{"prediction": 0}
```

The feature order matches `load_wine().feature_names`:

1. alcohol
2. malic_acid
3. ash
4. alcalinity_of_ash
5. magnesium
6. total_phenols
7. flavanoids
8. nonflavanoid_phenols
9. proanthocyanins
10. color_intensity
11. hue
12. od280/od315_of_diluted_wines
13. proline

The API accepts exactly 13 numbers. An incorrect number of features returns status code `422`.
Check `GET /`, `POST /predict`, and `/docs` both locally and in Docker.
