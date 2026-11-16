"""E1 localization on CPU with ONNX Runtime. No torch or torchvision imports."""
import base64
import io
import logging
import math
import os
import secrets
import threading
import time
import warnings
from contextlib import asynccontextmanager
from typing import Annotated

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from download_model import ensure_model

MAX_UPLOAD_BYTES = 3 * 1024 * 1024
MAX_PIXELS = 4_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
THRESHOLD = float(os.environ.get("MFD_THRESHOLD", "0.5"))
if not math.isfinite(THRESHOLD) or not 0 <= THRESHOLD <= 1:
    raise ValueError("MFD_THRESHOLD must be finite and within [0, 1]")
SESSION = None
LOCK = threading.Lock()
AUTH = HTTPBearer(auto_error=False)
LOGGER = logging.getLogger("mfd-e1")


def load_session():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.add_session_config_entry("session.disable_prepacking", "1")
    options.log_severity_level = 3
    return ort.InferenceSession(str(ensure_model()), sess_options=options,
                                providers=["CPUExecutionProvider"])


@asynccontextmanager
async def lifespan(app):
    global SESSION
    SESSION = load_session()
    yield
    SESSION = None


app = FastAPI(title="MFD-DL E1 ONNX backend", lifespan=lifespan)


def authenticate(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(AUTH)]):
    token = os.environ.get("MFD_BACKEND_TOKEN", "")
    if not token:
        raise HTTPException(503, "Configure MFD_BACKEND_TOKEN")
    if credentials is None or not secrets.compare_digest(credentials.credentials, token):
        raise HTTPException(401, "Invalid backend token", headers={"WWW-Authenticate": "Bearer"})


@app.get("/")
@app.get("/healthz")
def health():
    return {"service": "MFD-DL E1 ONNX", "model_loaded": SESSION is not None,
            "configured": bool(os.environ.get("MFD_BACKEND_TOKEN")), "docs": "/docs"}


@app.post("/analyze", dependencies=[Depends(authenticate)])
def analyze(image: Annotated[UploadFile, File()]):
    if SESSION is None:
        raise HTTPException(503, "Model unavailable")
    if not LOCK.acquire(blocking=False):
        raise HTTPException(503, "Model busy; retry shortly", headers={"Retry-After": "5"})
    try:
        started = time.perf_counter()
        raw = image.file.read(MAX_UPLOAD_BYTES + 1)
        if not raw or len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Use a non-empty image under 3 MB")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as opened:
                    if opened.format not in {"JPEG", "PNG", "WEBP"} or getattr(opened, "n_frames", 1) != 1:
                        raise HTTPException(415, "Use a still JPG, PNG, or WebP")
                    if opened.width * opened.height > MAX_PIXELS:
                        raise HTTPException(413, "Resize the image to 4 megapixels or fewer")
                    original = ImageOps.exif_transpose(opened).convert("RGB")
        except HTTPException:
            raise
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombWarning, Image.DecompressionBombError):
            raise HTTPException(400, "Invalid image; use JPG/PNG/WebP up to 4 megapixels") from None
        resized = original.resize((512, 512), Image.Resampling.BILINEAR)
        x = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        x = (x - np.array([.485, .456, .406], np.float32)[:, None, None]) / np.array(
            [.229, .224, .225], np.float32)[:, None, None]
        probabilities = SESSION.run(["segmentation"], {"image": x[None].copy()})[0]
        if probabilities.shape != (1, 1, 512, 512) or not np.isfinite(probabilities).all() or probabilities.min() < 0 or probabilities.max() > 1:
            raise RuntimeError("Invalid model probabilities")
        # Sigmoid is already part of the exported graph. Keep the full photograph's
        # aspect ratio and map probabilities to a bounded display grid, not a benchmark grid.
        original.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        probability_image = Image.fromarray(probabilities[0, 0])
        mask = np.asarray(probability_image.resize(original.size, Image.Resampling.BILINEAR)) >= THRESHOLD
        flagged = bool(mask.any())
        rgb = np.asarray(original, dtype=np.float32).copy()
        rgb[mask] = .65 * rgb[mask] + .35 * np.array([0, 210, 184], np.float32)
        binary = Image.fromarray(mask.astype(np.uint8) * 255)
        outline = (np.asarray(binary.filter(ImageFilter.MaxFilter(3))) > 0) & ~(
            np.asarray(binary.filter(ImageFilter.MinFilter(3))) > 0)
        rgb[outline] = [0, 174, 151]
        output_image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        output = io.BytesIO()
        output_image.save(output, "JPEG", quality=85)
        if output.tell() > 2_500_000:
            raise RuntimeError("Output too large")
        LOGGER.warning("ONNX E1 inference completed in %.3fs", time.perf_counter() - started)
        return {"schema_version": 1, "localization_jpeg_base64": base64.b64encode(output.getvalue()).decode("ascii"),
                "forgery_type": "unknown" if flagged else "not_applicable",
                "verdict": "suspected_manipulation" if flagged else "no_regions_detected"}
    except HTTPException:
        raise
    except Exception:
        LOGGER.exception("ONNX E1 inference failed")
        raise HTTPException(503, "Inference failed; check backend logs") from None
    finally:
        image.file.close()
        LOCK.release()
