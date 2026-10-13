# Session 2 — Model Compilation: INT8 and BF16 (30 min)

[← Session 1: C++ API](01-neat-cpp-api.md) · [Agenda](00-agenda.md) · [Next: App 1 →](03-app1-single-stream.md)

**Goal:** turn two public PyTorch checkpoints into Modalix model archives yourself:

| Model | Task | Precision | Used in |
|---|---|---|---|
| `yolo26n.pt` | Object detection (80 COCO classes) | **INT8** | App 1, App 2 stream 0 |
| `yolo26n-pose.pt` | Human pose (17 keypoints) | **BF16** | App 2 stream 1 |

| Time | Step |
|---|---|
| 3 min | Set up the environment and calibration images |
| 4 min | Step 1: export to ONNX · Step 2: graph surgery |
| 2 min | Step 3: start both compiles (they run about 16 min in total, in the background) |
| 16 min | While they run: how the flow works, INT8 vs BF16, what surgery does |
| 5 min | Step 4: verify and benchmark the archives |

> ⚠️ **Compile one model at a time.** The compiler needs several GB of RAM. Two compiles in parallel
> on a 16 GB host can run out of memory.

---

## The flow

```text
 yolo26n.pt ──► export_onnx.py ──► yolo26n.onnx ──► surgery.py ──► yolo26n.compile_ready.onnx
 (PyTorch)        Step 1            (has a CPU-style      Step 2       (only raw heads,
                                     decode tail)                       100% MLA-placeable)
                                                                                │
                                         calib/ (real images) ──► compile.py ◄──┘   Step 3
                                                                       │
                                                  yolo26n.compile_ready_mpk.tar.gz  ◄── what Neat loads
```

The three scripts below are all you need. They live in `workshop/model-compilation/`.

---

## Step 0 — Environment and calibration images (3 min)

```bash
mkdir -p /workspace/workshop/model-compilation/work
cd /workspace/workshop/model-compilation/work
source /sdk-extensions/model-compiler/bin/activate      # the Model Compiler (afe) + onnx + torch
python -c "import afe, ultralytics; print('ready')"      # see the agenda, step 6, if this fails
```

INT8 quantization learns the value range of every tensor from **calibration images**. Use real
images from your domain. COCO128 (128 everyday images, 7 MB) is a good fit for COCO models:

```bash
curl -L -o coco128.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
unzip -qo coco128.zip
mkdir -p calib && cp coco128/images/train2017/*.jpg calib/
ls calib | wc -l      # 128
```

> **Never calibrate on random or synthetic images.** The compile still succeeds and the archive
> looks perfect, but the INT8 ranges are meaningless and the model is quietly wrong.

---

## Step 1 — Export to ONNX (2 min)

Create `../export_onnx.py`:

<!-- file: model-compilation/export_onnx.py -->
```python
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
    # and Neat decodes the boxes on the device.
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, opset=17,
                             dynamic=False, simplify=True, nms=False)
    print(f"[export] wrote {Path(onnx_path).resolve()}")


if __name__ == "__main__":
    main()
```

Run it for both models (Ultralytics downloads the `.pt` files automatically):

```bash
python ../export_onnx.py --weights yolo26n.pt
python ../export_onnx.py --weights yolo26n-pose.pt
```

---

## Step 2 — Graph surgery (2 min)

### Why surgery?

A stock YOLO export ends in a **decode tail**: concat, transpose, anchor maths, top-k. These are
CPU-style operations. If you compile the graph as is, the compiler splits it, and parts fall back to
the ARM CPU (`.so` files in the archive), which is slow.

The fix is to **cut the tail off and expose the raw heads**. Neat already decodes the boxes on the EV74
(`decode_type = YoloV26` / `YoloV26Pose`; YOLO26's one-to-one heads need no NMS), so the network
only needs to deliver raw head tensors:

```text
stock export : backbone → heads → [concat · transpose · decode · top-k] → 1 output
                                   └──────── not MLA-friendly ────────┘
compile_ready: backbone → heads                                         → 6 or 9 raw outputs
                                                                          (all on the MLA)
```

Two further fixes are in the script:

- **Attention `MatMul` → `Einsum`**: YOLO26 has two attention blocks. `Einsum` with these
  equations maps cleanly onto the MLA.
- **Pose keypoint head 51 → 64 channels**: 17 keypoints × (x, y, visibility) = 51 channels. Padding
  with zeros to 64 lets the compiler fuse the output stage. Measured on yolo26s-pose: **8.5 ms
  instead of 1782 ms** per frame. Neat's `YoloV26Pose` decoder reads the first 51 channels and
  ignores the padding.

Create `../surgery.py`:

<!-- file: model-compilation/surgery.py -->
```python
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
```

Run it:

```bash
python ../surgery.py --onnx yolo26n.onnx
python ../surgery.py --onnx yolo26n-pose.onnx
```

Expected:

```text
[surgery] yolo26n.onnx: task = detection
[surgery] /model.10/m/m.0/attn/MatMul: MatMul -> Einsum
...
[surgery] classes = 80
[surgery] outputs = ['bbox_0', 'bbox_1', 'bbox_2', 'class_logit_0', 'class_logit_1', 'class_logit_2']
[surgery] wrote yolo26n.compile_ready.onnx
[surgery] yolo26n-pose.onnx: task = pose
...
[surgery] classes = 1
[surgery] outputs = ['bbox_0', 'bbox_1', 'bbox_2', 'class_logit_0', 'class_logit_1', 'class_logit_2', 'kpt_0', 'kpt_1', 'kpt_2']
[surgery] wrote yolo26n-pose.compile_ready.onnx
```

---

## Step 3 — Quantize and compile (start now, about 16 min in the background)

Create `../compile.py`:

<!-- file: model-compilation/compile.py -->
```python
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
```

Start both compiles, **one after the other**, in the background:

```bash
( python -u ../compile.py --onnx yolo26n.compile_ready.onnx      --precision int8 --calib-dir calib \
  && python -u ../compile.py --onnx yolo26n-pose.compile_ready.onnx --precision bf16 --calib-dir calib \
) > compile.log 2>&1 &

tail -f compile.log | grep --line-buffered -E '^\[compile\]|Error|Traceback'   # Ctrl+C stops watching, not the compile
```

Expected, after about 7 + 9 minutes:

```text
[compile] yolo26n.compile_ready: input images(1, 3, 640, 640), 6 outputs, int8
[compile] calibrating on 50 images
[compile] archive : compiled_int8/yolo26n.compile_ready/yolo26n.compile_ready_mpk.tar.gz
[compile] contents: 1 .elf, 0 .so
[compile] PASS: single MLA program
[compile] yolo26n-pose.compile_ready: input images(1, 3, 640, 640), 9 outputs, bf16
[compile] calibrating on 50 images
[compile] archive : compiled_bf16/yolo26n-pose.compile_ready/yolo26n-pose.compile_ready_mpk.tar.gz
[compile] contents: 1 .elf, 0 .so
[compile] PASS: single MLA program
```

### While it compiles: what is happening

`compile.py` makes four Model Compiler (`afe`) calls:

| Call | What it does |
|---|---|
| `load_model(params, target=gen2_target)` | Imports the ONNX graph for **Modalix** (`gen2_target`; MLSoC is `gen1_target`) |
| `net.quantize(calibration_data, config)` | Runs the calibration images through the network, records each tensor's range, and converts weights and activations to the chosen precision |
| `quant.save(...)` | Saves the quantized network (`.sima` files) for inspection or reuse |
| `quant.compile(output_path, batch_size=1)` | Generates the MLA program (`.elf`) and packages it with the pipeline description into `*_mpk.tar.gz` |

### INT8 or BF16?

The **only** difference in the script is the quantization scheme:

```python
if args.precision == "bf16":
    act_scheme, weight_scheme = bfloat16_scheme(), bfloat16_scheme()
else:
    act_scheme = quantization_scheme(True, False, 8)      # asymmetric, per-tensor
    weight_scheme = quantization_scheme(False, True, 8)   # symmetric, per-channel
```

| | **INT8** | **BF16** |
|---|---|---|
| Number format | 8-bit integers + a scale per tensor/channel | 16-bit floats (8-bit exponent, like FP32) |
| Accuracy | Small loss. Depends on good calibration data | Very close to FP32 |
| Speed and size | Fastest, smallest | Slower, larger |
| Risk | Some architectures do not survive 8-bit quantization | No ranges to get wrong |
| Typical use | Production detection at high frame rates | Pose, segmentation, or precision-sensitive models, and models that fail INT8 |

Reference numbers for YOLO26s on COCO val: INT8 lost 2.67 mAP50-95 points against FP32; BF16 lost 0.07.

### The archive contract: 1 `.elf`, 0 `.so`

The last check in `compile.py` opens the archive and counts files **by extension**:

| Archive contains | Meaning |
|---|---|
| exactly **1 `.elf`**, **0 `.so`** | The whole network runs on the MLA. ✅ |
| a `.so` | Part of the graph fell back to the ARM CPU. Surgery is incomplete |
| several `.elf` | The graph was split into pieces |

An exit code of 0 alone does not prove a good archive. Always check the contents, and run the model on real
images. A model can compile cleanly and still be numerically wrong.

---

## Step 4 — Verify and benchmark (5 min)

Use the `model_info` program from session 1 on both archives:

```bash
cd /workspace/workshop/model-info
W=/workspace/workshop/model-compilation/work
dk ./build/model_info $W/compiled_int8/yolo26n.compile_ready/yolo26n.compile_ready_mpk.tar.gz
dk ./build/model_info $W/compiled_bf16/yolo26n-pose.compile_ready/yolo26n-pose.compile_ready_mpk.tar.gz
```

The pose model shows nine outputs. The keypoint heads are 64 channels wide, because of the padding:

```text
outputs:
  [80, 80, 4]          ← boxes
  [40, 40, 4]
  [20, 20, 4]
  [80, 80, 1]          ← person score
  [40, 40, 1]
  [20, 20, 1]
  [80, 80, 64]         ← 17 keypoints x 3 (+13 zero padding)
  [40, 40, 64]
  [20, 20, 64]
```

(The INT8 detection archive prints its shapes with a leading batch dimension, e.g. `[1, 80, 80, 4]`.)

Results measured on the Modalix DevKit (`Model::benchmark`, 200 samples):

| Archive | Precision | Compile time | Latency | Throughput |
|---|---|---|---|---|
| yolo26n detection (this flow) | INT8 | 6 min 28 s | **5.45 ms** | **550 fps** |
| yolo26n detection (`--precision bf16`, or the prebuilt `yolo26n-bf16…`) | BF16 | 8 min 0 s | 9.79 ms | 224 fps |
| yolo26n-pose (this flow) | BF16 | 8 min 47 s | 10.27 ms | 199 fps |

Same network, same calibration data: **INT8 runs about 2.5× faster than BF16** (550 vs 224 fps).
For detection at video rates, INT8 is the default. BF16 is the choice when accuracy matters more than raw speed.

Copy the archives into the app folders for the next sessions:

```bash
W=/workspace/workshop/model-compilation/work
mkdir -p /workspace/workshop/app1-single-stream/models /workspace/workshop/app2-dual-stream/models
cp $W/compiled_int8/yolo26n.compile_ready/yolo26n.compile_ready_mpk.tar.gz       /workspace/workshop/app1-single-stream/models/
cp $W/compiled_int8/yolo26n.compile_ready/yolo26n.compile_ready_mpk.tar.gz       /workspace/workshop/app2-dual-stream/models/
cp $W/compiled_bf16/yolo26n-pose.compile_ready/yolo26n-pose.compile_ready_mpk.tar.gz /workspace/workshop/app2-dual-stream/models/
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `libLLVM.so.18.1: cannot open shared object file` | `sudo apt-get install -y libllvm18 libopenblas0` (agenda, step 6) |
| `No module named 'ultralytics'` | `pip install "ultralytics==8.4.175"` inside the model-compiler venv (agenda, step 6) |
| Compile killed / out of memory | Run one compile at a time, close other memory-heavy programs, and give Docker more RAM if needed |
| `cannot infer shape of /model.23/…`, or an onnxsim / checker error in surgery | Not a YOLO26 export, or a different Ultralytics version renamed nodes. Open the ONNX in <https://netron.app> and check `/model.23/one2one_cv*` |
| `FAIL: graph is fragmented` | Something was left that the MLA cannot place. Read the compile log for the operator that fell back |
| Detections are wrong or missing at runtime | Check the calibration images (real, from your domain), and that the app's `decode_type` matches the head (`YoloV26` / `YoloV26Pose`) |

[← Session 1: C++ API](01-neat-cpp-api.md) · [Agenda](00-agenda.md) · [Next: App 1 →](03-app1-single-stream.md)
