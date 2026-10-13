#!/usr/bin/env python3
"""Step 2 - graph surgery: make a YOLO26 ONNX graph run 100% on the MLA.

    python surgery.py --onnx yolo26n.onnx            # -> yolo26n.compile_ready.onnx
    python surgery.py --onnx yolo26n-pose.onnx       # -> yolo26n-pose.compile_ready.onnx

What it does:
  1. Rewrites the two attention MatMuls per block into Einsum (MLA-friendly).
  2. Cuts off the CPU-style decode tail (concat / transpose / top-k) and exposes the
     raw per-scale heads as graph outputs, in the order Neat's box decoder expects:
        detection : bbox_0..2 (4 ch), class_logit_0..2 (80 ch)
        pose      : bbox_0..2 (4 ch), class_logit_0..2 (1 ch), kpt_0..2 (51 ch padded to 64)
  3. Simplifies and checks the result.

Works for every YOLO26 size (n/s/m/l/x): the head node names are the same.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxsim import simplify

INPUT_NAME = "images"
INPUT_SHAPE = [1, 3, 640, 640]
SCALES = [(0, 80, 80), (1, 40, 40), (2, 20, 20)]          # stride 8 / 16 / 32
ATTENTION_BLOCKS = ["/model.10/m/m.0/attn", "/model.22/m.0/m.0.1/attn"]
HEAD = "/model.23"
BBOX_SRC = HEAD + "/one2one_cv2.{i}/one2one_cv2.{i}.2/Conv_output_0"
CLASS_SRC = HEAD + "/one2one_cv3.{i}/one2one_cv3.{i}.2/Conv_output_0"
KPT_SRC = HEAD + "/one2one_cv4_kpts.{i}/Conv_output_0"
KPT_CHANNELS, KPT_PADDED = 51, 64   # 17 keypoints x (x, y, visibility); see note below


def find_node(model, name):
    return next((n for n in model.graph.node if n.name == name), None)


def attention_to_einsum(model):
    for prefix in ATTENTION_BLOCKS:
        for suffix, eq in (("MatMul", "bhnc,bhck->bhnk"), ("MatMul_1", "bhcn,bhnm->bhcm")):
            node = find_node(model, f"{prefix}/{suffix}")
            if node is None:
                continue
            node.op_type = "Einsum"
            node.attribute.append(helper.make_attribute("equation", eq))
            print(f"[surgery] {node.name}: MatMul -> Einsum")


def tensor_channels(model, name):
    inferred = onnx.shape_inference.infer_shapes(model)
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output):
        if vi.name == name:
            return vi.type.tensor_type.shape.dim[1].dim_value
    raise ValueError(f"cannot infer shape of {name}")


def expose_heads(model, is_pose):
    nodes, inits, outputs = [], [], []

    def expose(src, name, channels, h, w):
        nodes.append(helper.make_node("Identity", [src], [name], name=f"/heads/{name}"))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, channels, h, w]))

    num_classes = tensor_channels(model, CLASS_SRC.format(i=0))
    for i, h, w in SCALES:
        expose(BBOX_SRC.format(i=i), f"bbox_{i}", 4, h, w)
    for i, h, w in SCALES:
        expose(CLASS_SRC.format(i=i), f"class_logit_{i}", num_classes, h, w)

    if is_pose:
        # Zero-pad the 51-channel keypoint head to 64 channels. Same weights, same
        # information, but the compiler can then fuse the output tail: measured
        # 8.5 ms/frame padded vs 1782 ms/frame unpadded on yolo26s-pose.
        # Neat's YoloV26Pose decoder reads the first 51 channels and ignores the rest.
        for i, h, w in SCALES:
            pad = f"kpt_pad_{i}"
            inits.append(numpy_helper.from_array(
                np.zeros((1, KPT_PADDED - KPT_CHANNELS, h, w), np.float32), pad))
            nodes.append(helper.make_node("Concat", [KPT_SRC.format(i=i), pad], [f"kpt_{i}"],
                                          name=f"/heads/kpt_{i}", axis=1))
            outputs.append(helper.make_tensor_value_info(
                f"kpt_{i}", TensorProto.FLOAT, [1, KPT_PADDED, h, w]))

    model.graph.node.extend(nodes)
    model.graph.initializer.extend(inits)
    del model.graph.output[:]
    model.graph.output.extend(outputs)
    return [o.name for o in outputs], num_classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, type=Path)
    args = ap.parse_args()

    model = onnx.load(args.onnx)
    is_pose = find_node(model, f"{HEAD}/one2one_cv4_kpts.0/Conv") is not None
    print(f"[surgery] {args.onnx.name}: task = {'pose' if is_pose else 'detection'}")

    attention_to_einsum(model)
    names, num_classes = expose_heads(model, is_pose)

    model, ok = simplify(model, overwrite_input_shapes={INPUT_NAME: INPUT_SHAPE})
    if not ok:
        raise SystemExit("[surgery] onnxsim check failed")
    onnx.checker.check_model(model)

    out = args.onnx.with_suffix(".compile_ready.onnx")
    onnx.save(model, out)
    print(f"[surgery] classes = {num_classes}")
    print(f"[surgery] outputs = {names}")
    print(f"[surgery] wrote {out}")


if __name__ == "__main__":
    main()
