"""Usage: python src/bench.py <model_path> <device> [label]
   device: 0 (GPU) | cpu  (for OpenVINO devices see the Ultralytics OpenVINO docs)
"""

import json
import os
import sys
import numpy as np
from ultralytics import YOLO

IMG = "assets/test.jpg"

def size_mb(path):
    if os.path.isdir(path):
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
    return os.path.getsize(path) / 1e6

def main(path, device, label=None, warmup=30, iters=300, imgsz=640):
    model = YOLO(path, task="detect")
    for _ in range(warmup):
        model.predict(IMG, imgsz=imgsz, device=device, verbose=False)

    infer, total  = [], []
    for _ in range(iters):
        s = model.predict(IMG, imgsz=imgsz, device=device, verbose=False)[0].speed
        infer.append(s["inference"])
        total.append(sum(s.values()))

    row = dict(
        model = label or os.path.basename(path), 
        device = str(device),
        infer_p50_ms = round(float(np.percentile(infer, 50)), 2),
        infer_p90_ms = round(float(np.percentile(infer, 95)), 2),
        e2e_fps = round(1000 / float(np.percentile(total, 50)), 1), 
        size_mb = round(size_mb(path), 1)
    )
    print(json.dumps(row))
    os.makedirs("results", exist_ok=True)
    with open("results/bench.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    main(*sys.argv[1:4])
