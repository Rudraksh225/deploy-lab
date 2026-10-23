import shutil
from pathlib import Path
from ultralytics import YOLO

ART = Path("artifacts")
ART.mkdir(exist_ok=True)
CALIB = "coco128.yaml"

def run(tag, **kw):
    """Export, then rename so FP16 / INT8 outputs do not overwrite each other."""
    p = Path(YOLO("yolo26n.pt").export(**kw))
    name = p.name
    if tag == "int8":
        name = name.replace("_int8_openvino_model", "_openvino_model")
    dst = ART / name.replace("yolo26n", f"yolo26n-{tag}", 1)
    if dst.is_dir(): 
        shutil.rmtree(dst)
    elif dst.exists():
        dst.unlink()
    shutil.move(str(p), str(dst))
    print("->", dst)

if __name__ == "__main__":
    # run("fp32", format="onnx", dynamic=True, simplify=True)
    # run("fp16", format="engine", half=True, device=0)                    # TensorRT FP16
    # run("int8", format="engine", int8=True, data=CALIB, device=0)
    # run("fp32", format="openvino")    
    run("int8", format="openvino", int8=True, data=CALIB)    