"""Usage: python src/val.py <model_path> <device>"""
import sys

from ultralytics import YOLO

if __name__ == "__main__":
    path, device = sys.argv[1], sys.argv[2]
    m = YOLO(path, task="detect").val(data="coco128.yaml", imgsz=640, batch=1, device=device, workers=0, verbose=False)
    print(f"{path}: mAP50-95 = {m.box.map:.4f}")
