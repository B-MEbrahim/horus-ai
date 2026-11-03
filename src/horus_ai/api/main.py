import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
import time

import numpy as np
from fastapi import FastAPI, Request

from horus_ai.api.placeholder import PlaceholderDetector
from horus_ai.api.schemas import TimingSample, Verdict
from horus_ai.api.service import DetectionService
from horus_ai.models.base import BaseDetector  # adjust import

logger = logging.getLogger("horus_ai.api")
logging.basicConfig(level=logging.INFO)


def build_service() -> DetectionService:
    window_size = int(os.getenv("HORUS_WINDOW_SIZE", "30"))
    model_path = os.getenv("HORUS_MODEL_PATH")

    if model_path and Path(model_path).exists():
        detector = BaseDetector.load(model_path)
        logger.info("Loaded model %s from %s", detector.name, model_path)
    else:
        logger.warning("No trained model found, using PlaceholderDetector.")
        detector = PlaceholderDetector().fit(np.zeros((10, 6)))

    return DetectionService(detector, window_size)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = build_service()  # runs once, when the app starts
    yield


app = FastAPI(title="HORUS AI", lifespan=lifespan)

@app.middleware("http")
async def log_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %d in %.2f ms", request.method, request.url.path, response.status_code, ms)
    response.headers["X-Process-Time-Ms"] = f"{ms:.2f}"
    return response


@app.get("/health")
def health(request: Request):
    svc = request.app.state.service
    return {"status": "ok", "detector": svc.detector.name, "window_size": svc.window_size}


@app.post("/predict", response_model=Verdict)
def predict(sample: TimingSample, request: Request):
    return request.app.state.service.process(sample)