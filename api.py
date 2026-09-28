import hmac
import json
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "modelo_emnist_balanced.keras"
LABEL_MAP_FILE = BASE_DIR / "label_map.json"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_FILE.is_file() or not LABEL_MAP_FILE.is_file():
        raise RuntimeError(
            "Model files are missing. Run treino.py first or copy both "
            "modelo_emnist_balanced.keras and label_map.json into the project."
        )
    app.state.model = load_model(MODEL_FILE)
    with LABEL_MAP_FILE.open(encoding="utf-8") as label_file:
        app.state.label_map = json.load(label_file)
    yield


app = FastAPI(
    title="EMNIST Character Recognition API",
    version="1.0.0",
    lifespan=lifespan,
)


def preprocess_image_for_model(image: Image.Image) -> np.ndarray:
    """Mirror the exact training preprocessing as closely as possible."""
    grayscale = image.convert("L")

    array = np.asarray(grayscale)
    mask = array < 200
    if mask.any():
        ys, xs = np.where(mask)
        grayscale = grayscale.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    width, height = grayscale.size
    size = max(width, height)
    padded = Image.new("L", (size, size), 255)
    padded.paste(grayscale, ((size - width) // 2, (size - height) // 2))

    pixels = np.asarray(padded.resize((28, 28), Image.Resampling.LANCZOS), dtype=np.uint8)
   

    # Match the training data exactly: the model was trained on black strokes over a
    # white background. Inverting dark inputs here makes the network see the wrong
    # foreground/background polarity and produces garbage predictions.
    return pixels.astype(np.float32)[None, ..., None] / np.float32(255.0)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        raise HTTPException(status_code=503, detail="API_KEY is not configured")
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(file: UploadFile = File(...)) -> dict[str, int | float | str]:
    image_bytes = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions are too large")
            model_input = preprocess_image_for_model(image)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="File is not a valid image") from error

    probabilities = app.state.model.predict(model_input, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    character_code = app.state.label_map.get(str(class_index))
    if character_code is None:
        raise HTTPException(status_code=500, detail="Predicted class has no label mapping")

    return {
        "class_index": class_index,
        "character": chr(int(character_code)),
        "confidence": float(probabilities[class_index]),
    }
