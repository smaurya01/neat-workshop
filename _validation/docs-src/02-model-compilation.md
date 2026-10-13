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
unzip -q coco128.zip
mkdir -p calib && cp coco128/images/train2017/*.jpg calib/
ls calib | wc -l      # 128
```

> **Never calibrate on random or synthetic images.** The compile still succeeds and the archive
> looks perfect, but the INT8 ranges are meaningless and the model is quietly wrong.

---

## Step 1 — Export to ONNX (2 min)

Create `../export_onnx.py`:

{{include:model-compilation/export_onnx.py}}

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

The fix is to **cut the tail off and expose the raw heads**. Neat already decodes boxes and runs NMS on the EV74
(`decode_type = YoloV26` / `YoloV26Pose`), so the network only needs to deliver raw head tensors:

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

{{include:model-compilation/surgery.py}}

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

{{include:model-compilation/compile.py}}

Start both compiles, **one after the other**, in the background:

```bash
( python -u ../compile.py --onnx yolo26n.compile_ready.onnx      --precision int8 --calib-dir calib \
  && python -u ../compile.py --onnx yolo26n-pose.compile_ready.onnx --precision bf16 --calib-dir calib \
) > compile.log 2>&1 &

tail -f compile.log | grep --line-buffered '^\[compile\]'      # Ctrl+C stops watching, not the compile
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
  [80, 80, 4]     [40, 40, 4]     [20, 20, 4]       ← boxes
  [80, 80, 1]     [40, 40, 1]     [20, 20, 1]       ← person score
  [80, 80, 64]    [40, 40, 64]    [20, 20, 64]      ← 17 keypoints x 3 (+13 zero padding)
```

Results measured on the Modalix DevKit (`Model::benchmark`, 200 samples):

| Archive | Precision | Compile time | Latency | Throughput |
|---|---|---|---|---|
| yolo26n detection (this flow) | INT8 | 6 min 28 s | **5.45 ms** | **550 fps** |
| yolo26n detection (this flow) | BF16 | 8 min 0 s | 9.79 ms | 224 fps |
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
| `No module named 'ultralytics'` | `pip install ultralytics` inside the model-compiler venv |
| Compile killed / out of memory | Run one compile at a time; `export SIMA_MLA_SIM_PARALLEL=4` (or lower) and retry |
| `missing head tensors` / node not found in surgery | Not a YOLO26 export, or a different Ultralytics version renamed nodes. Open the ONNX in <https://netron.app> and check `/model.23/one2one_cv*` |
| `FAIL: graph is fragmented` | Something was left that the MLA cannot place. Read the compile log for the operator that fell back |
| Detections are wrong or missing at runtime | Check the calibration images (real, from your domain), and that the app's `decode_type` matches the head (`YoloV26` / `YoloV26Pose`) |

## Checkpoint

- [ ] Both compiles print `PASS: single MLA program`
- [ ] `model_info` shows 6 outputs for detection and 9 for pose
- [ ] You can explain why we cut the decode tail off, and when you would choose BF16 over INT8
- [ ] The archives are copied into `app1-single-stream/models` and `app2-dual-stream/models`

[← Session 1: C++ API](01-neat-cpp-api.md) · [Agenda](00-agenda.md) · [Next: App 1 →](03-app1-single-stream.md)
