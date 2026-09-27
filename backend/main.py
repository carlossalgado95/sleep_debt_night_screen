import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.recommend import (
    RecommendationError,
    build_recommendations,
    load_dotenv,
    recommendation_context,
)
from backend.service import service

app = FastAPI(
    title="The Midnight Scroll",
    description="Prevê a categoria de dívida de sono a partir dos hábitos do notebook.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    load_dotenv()
    service.load()


@app.get("/")
def routes() -> dict:
    return {
        "name": "The Midnight Scroll",
        "routes": [
            {"method": "GET", "path": "/health"},
            {"method": "GET", "path": "/schema"},
            {"method": "GET", "path": "/metrics"},
            {"method": "POST", "path": "/predict"},
            {"method": "POST", "path": "/recommend"},
        ],
    }


@app.get("/health")
def health() -> dict:
    metrics = service.metrics()
    return {
        "status": "ok",
        "data_source": metrics["data_source"],
        "n_rows": metrics["n_rows"],
        "accuracy": metrics["accuracy"],
    }


@app.get("/schema")
def schema() -> dict:
    return service.schema()


@app.get("/metrics")
def metrics() -> dict:
    return service.metrics()


@app.post("/predict")
def predict(payload: dict) -> dict:
    try:
        return service.predict(payload)
    except ValueError as exc:
        detail = json.loads(str(exc))
        raise HTTPException(status_code=422, detail=detail) from exc


@app.post("/recommend")
def recommend(payload: dict) -> dict:
    try:
        prediction = service.predict(payload)
        context = recommendation_context(payload, prediction, service.bundle)
        advice = build_recommendations(context)
    except ValueError as exc:
        detail = json.loads(str(exc))
        raise HTTPException(status_code=422, detail=detail) from exc
    except RecommendationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"message": str(exc)},
        ) from exc
    return {"prediction": prediction, "recommendations": advice}
