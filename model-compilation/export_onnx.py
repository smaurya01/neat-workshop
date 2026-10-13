#!/usr/bin/env python3
"""Step 1 - export an Ultralytics YOLO26 checkpoint to a static ONNX graph.

    python export_onnx.py --weights yolo26n.pt
    python export_onnx.py --weights yolo26n-pose.pt

The .pt file is downloaded automatically by Ultralytics if it is not present.
Output: <name>.onnx next to the weights, input "images" [1, 3, 640, 640].
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="yolo26n.pt, yolo26n-pose.pt, ...")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    # Static shape, opset 17, NO NMS: graph surgery cuts the decode tail off next,
    # and Neat runs box decode + NMS on the device.
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, opset=17,
                             dynamic=False, simplify=True, nms=False)
    print(f"[export] wrote {Path(onnx_path).resolve()}")


if __name__ == "__main__":
    main()
