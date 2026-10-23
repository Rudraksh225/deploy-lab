import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
from fastapi import HTTPException
#
import tritonclient.http as triton_http

_original_client_init = triton_http.InferenceServerClient.__init__

def _client_init_with_longer_timeout(self, *args, **kwargs):
    kwargs.setdefault("connection_timeout", 300.0)
    kwargs.setdefault("network_timeout", 300.0)
    return _original_client_init(self, *args, **kwargs)

triton_http.InferenceServerClient.__init__ = _client_init_with_longer_timeout
#

app = FastAPI(title="deploy-lab gateway")
model = YOLO(os.getenv("TRITON_URL", "http://localhost:8000/yolo_onnx_trt"), task="detect")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile):
    img = cv2.imdecode(
        np.frombuffer(await file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image file"
        )
    
    r = model.predict(img, verbose=False)[0]

    return {
        "detections": [
            {"label": r.names[int(b.cls)], "conf": round(float(b.conf), 3), "xyxy": [round(v, 1) for v in b.xyxy[0].tolist()]}
            for b in r.boxes
        ],
        "speed_ms": r.speed,
    }