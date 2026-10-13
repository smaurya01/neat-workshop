#!/usr/bin/env python3
"""Step 3 - quantize (INT8 or BF16) and compile an ONNX model into a Modalix model archive.

    python compile.py --onnx yolo26n.compile_ready.onnx      --precision int8 --calib-dir calib
    python compile.py --onnx yolo26n-pose.compile_ready.onnx --precision bf16 --calib-dir calib

INT8 : weights + activations in 8 bit. Smallest and fastest. Activation ranges are learned
       from the calibration images, so they MUST be real images from your domain.
BF16 : weights + activations in bfloat16. Close to FP32 accuracy, larger and slower.
       (The calibration pass still runs, but there are no 8-bit ranges to get wrong.)

Output: <out-dir>/<onnx-name>/<onnx-name>_mpk.tar.gz   (the archive Neat loads)
"""
import argparse
import logging
import tarfile
from pathlib import Path

import numpy as np
import onnx
from PIL import Image

from afe.apis.defines import (CalibrationMethod, InputName, RequantizationMode,
                              bfloat16_scheme, default_quantization, gen2_target,
                              quantization_scheme)
from afe.apis.loaded_net import load_model
from afe.core.utils import convert_data_generator_to_iterable
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import ImporterParams, ModelFormat
from sima_utils.data.data_generator import DataGenerator


def letterbox(path: Path, size: int) -> np.ndarray:
    """Same preprocessing Neat applies at runtime: letterbox, pad 114, RGB, x/255, NHWC."""
    img = Image.open(path).convert("RGB")
    scale = min(size / img.width, size / img.height)
    w, h = round(img.width * scale), round(img.height * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img.resize((w, h), Image.BILINEAR), ((size - w) // 2, (size - h) // 2))
    return np.asarray(canvas, dtype=np.float32)[None] / 255.0        # [1, H, W, 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--precision", choices=["int8", "bf16"], required=True)
    ap.add_argument("--calib-dir", required=True, type=Path, help="folder of REAL images")
    ap.add_argument("--num-calib", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: compiled_<precision>/ next to the ONNX file")
    args = ap.parse_args()

    graph = onnx.load(args.onnx)
    inp = graph.graph.input[0]
    shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)      # (1, 3, 640, 640)
    outputs = [o.name for o in graph.graph.output]
    name = args.onnx.name.removesuffix(".onnx")
    out_dir = (args.out_dir or args.onnx.parent / f"compiled_{args.precision}") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compile] {name}: input {inp.name}{shape}, {len(outputs)} outputs, {args.precision}")

    # 1. Load the ONNX graph for the Modalix target (gen2).
    params = ImporterParams(format=ModelFormat.onnx, file_paths=[str(args.onnx)],
                            input_names=[inp.name], input_shapes=[shape],
                            input_types=[ScalarType.float32], layout="NCHW",
                            output_names=outputs)
    net = load_model(params, target=gen2_target)

    # 2. Calibration data from real images.
    images = sorted(p for p in args.calib_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[:args.num_calib]
    if len(images) < 8:
        raise SystemExit(f"[compile] need >= 8 real calibration images in {args.calib_dir}")
    batch = np.concatenate([letterbox(p, shape[2]) for p in images])
    calib = convert_data_generator_to_iterable(DataGenerator({InputName(inp.name): batch}))
    print(f"[compile] calibrating on {len(images)} images")

    # 3. Quantization config: the only difference between INT8 and BF16 is the scheme.
    if args.precision == "bf16":
        act_scheme, weight_scheme = bfloat16_scheme(), bfloat16_scheme()
    else:
        act_scheme = quantization_scheme(True, False, 8)      # asymmetric, per-tensor
        weight_scheme = quantization_scheme(False, True, 8)   # symmetric, per-channel
    config = (default_quantization.with_activation_quantization(act_scheme)
              .with_weight_quantization(weight_scheme)
              .with_requantization_mode(RequantizationMode.sima)
              .with_calibration(CalibrationMethod.from_str("mse")))

    quant = net.quantize(calibration_data=calib, quantization_config=config,
                         model_name=name, log_level=logging.WARNING)
    quant.save(model_name=name, output_directory=str(out_dir))

    # 4. Compile to the archive Neat loads.
    quant.compile(output_path=str(out_dir), batch_size=1, log_level=logging.WARNING)

    # 5. Contract check: exactly one .elf and no .so = the whole graph runs on the MLA.
    archive = out_dir / f"{name}_mpk.tar.gz"
    members = tarfile.open(archive).getnames()
    elf = [m for m in members if m.endswith(".elf")]
    so = [m for m in members if m.endswith(".so")]
    print(f"[compile] archive : {archive}")
    print(f"[compile] contents: {len(elf)} .elf, {len(so)} .so")
    if len(elf) != 1 or so:
        raise SystemExit("[compile] FAIL: graph is fragmented or fell back to the CPU")
    print("[compile] PASS: single MLA program")


if __name__ == "__main__":
    main()
