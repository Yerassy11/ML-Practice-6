

from fastapi import FastAPI

app = FastAPI(title="Wine ML API")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ML API is running"}
