import base64
import io
import time
from typing import List, Dict

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO


class Input(BaseModel):
    model: str
    image: str


class Detection(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    class_name: str
    confidence: float


class Result(BaseModel):
    detections: List[Detection] = []
    time: float = 0.0
    model: str


def parse_prediction(box, class_name: str, confidence: float) -> Detection:
    x0, y0, x1, y1 = box
    detection = Detection(
        x_min=int(x0),
        y_min=int(y0),
        x_max=int(x1),
        y_max=int(y1),
        confidence=round(float(confidence), 3),
        class_name=class_name,
    )
    return detection


def load_model(model_name: str) -> YOLO:
    model = YOLO(f"{model_name}.pt")
    return model


app = FastAPI(
    title="YOLO v11 WebApp created with FastAPI",
    description="""
                Wraps 3 different YOLO v11 models under the same RESTful API
                """,
    version="2.0",
)

MODEL_NAMES = ["yolo11n", "yolo11s", "yolo11m"]
MODELS: Dict[str, YOLO] = {}


@app.get("/", description="return the title", response_description="title", response_model=str)
def root() -> str:
    return app.title


@app.get("/describe", description="return the description", response_description="description", response_model=str)
def describe() -> str:
    return app.description


@app.get("/version", description="return the version", response_description="version", response_model=str)
def get_version() -> str:
    return app.version


@app.get("/health", description="return whether it's alive", response_description="alive", response_model=str)
def health() -> str:
    return "HEALTH OK"


@app.get(
    "/models",
    description="Query the list of models",
    response_description="A list of available models",
    response_model=List[str],
)
def models() -> List[str]:
    return MODEL_NAMES


@app.post(
    "/predict",
    description="Send a base64 encoded image + the model name, get detections",
    response_description="Detections + Processing time",
    response_model=Result,
)
def predict(inputs: Input) -> Result:
    global MODELS

    model_name = inputs.model

    if model_name not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail="wrong model name, choose between {}".format(MODEL_NAMES))

    if MODELS.get(model_name) is None:
        MODELS[model_name] = load_model(model_name)

    model = MODELS.get(model_name)

    try:
        image = inputs.image.encode("utf-8")
        image = base64.b64decode(image)
        image = Image.open(io.BytesIO(image))
    except:
        raise HTTPException(status_code=400, detail="File is not an image")

    if image.mode == "RGBA":
        image = image.convert("RGB")

    t0 = time.time()
    results = model(image, imgsz=640)
    t1 = time.time()

    detections = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            cls = int(boxes.cls[i].cpu().numpy())
            class_name = result.names[cls]
            detections.append(parse_prediction(box=box, class_name=class_name, confidence=conf))

    result = Result(detections=detections, time=round(t1 - t0, 3), model=model_name)

    return result
