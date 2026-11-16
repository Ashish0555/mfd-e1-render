"""Fetch the verified ONNX export at build time; no PyTorch installation needed."""
import hashlib
import os
import tempfile
import time
import urllib.request
from pathlib import Path

MODEL_URL = "https://huggingface.co/spaces/Ashish0567/mfd-e1/resolve/main/e1.onnx"
MODEL_BYTES = 120159686
MODEL_SHA256 = "c9a34d54bb406deffba1c2e7b2ce8d73568d48f881d1695baf0c6b0501b9e154"
MODEL_PATH = Path(__file__).resolve().parent / "model" / "e1.onnx"


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_model():
    local = os.environ.get("MFD_ONNX_PATH")
    target = Path(local).expanduser().resolve() if local else MODEL_PATH
    if target.is_file() and target.stat().st_size == MODEL_BYTES and checksum(target) == MODEL_SHA256:
        return target
    if local:
        raise RuntimeError("MFD_ONNX_PATH is missing or does not match the verified export")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(dir=target.parent, suffix=".download")
    temporary = Path(filename)
    try:
        digest, count, started = hashlib.sha256(), 0, time.monotonic()
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > MODEL_BYTES or time.monotonic() - started > 180:
                        raise RuntimeError("Model download exceeded its size or time limit")
                    digest.update(block)
                    output.write(block)
        if count != MODEL_BYTES or digest.hexdigest() != MODEL_SHA256:
            raise RuntimeError("ONNX model checksum or size mismatch")
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    print(f"Verified ONNX model: {ensure_model()}")
