# Session 3 — App 1: One Source, One Model, One Stream (20 min)

[← Session 2: Model compilation](02-model-compilation.md) · [Agenda](00-agenda.md) · [Next: App 2 →](04-app2-dual-stream.md)

**Goal:** build a real-time object detection app. It reads **one** RTSP video stream, runs the **YOLO26n INT8** model you compiled, and shows live video
with detection boxes in Neat Insight.

| Time | Topic |
|---|---|
| 7 min | Code walkthrough: the five parts of the app |
| 5 min | Create the files, build |
| 8 min | Run, view in Insight, exercises |

---

## What we are building

```text
                          ┌──────────────────────────── one Neat Graph, one Run ──────────────────────────┐
                          │                                                                                │
                          │                         ┌─► "video"  VideoSender: HW H.264 encode → RTP/UDP ───┼──► Insight video  udp 9000+ch
 rtsp://…/src1  ───► source: HW H.264 decode ─► Branch                                                    │
                          │      (NV12 frames)      └─► "model"  YOLO26n: EV74 pre → MLA → EV74 decode+NMS │
                          │                                         │                                     │
                          └─────────────────────────────────────────┼─────────────────────────────────────┘
                                                                    ▼  Output "detections"
                                         app (A65): run.pull() → boxes → JSON → MetadataSender ──► Insight boxes  udp 9100+ch
```

Video frames never touch the CPU: decode, encode, preprocessing, inference and box decoding all
run on dedicated hardware. The app only handles a short list of boxes per frame.

---

## Code walkthrough

`main.cpp` has five numbered parts. Read them top to bottom.

### Part 1 — Configuration

A plain `key=value` file (`app1.conf`) holds the RTSP URL, the model path, the Insight host and the
frame size. If a value still contains a `<placeholder>`, the app stops with a clear message.

### Part 2 — The model

`model_options()` describes what happens around the MLA program:

- **Preprocess** (EV74): NV12 from the decoder → RGB, **letterbox** to 640×640 (keeps the aspect
  ratio, pads with grey), `COCO_YOLO` normalization (pixel / 255). This must match how the model was
  calibrated in session 2, and it does: `compile.py` letterboxes and divides by 255 too.
- `input_max_width/height` tells the preprocessor the largest frame it will receive.
- **Postprocess** (EV74): `decode_type = YoloV26` turns the six raw heads into boxes and runs NMS.
  `score_threshold`, `nms_iou_threshold` and `top_k` tune it.
- `pre_run_target / post_run_target = "EV74"` pins both steps on the vision DSP instead of letting
  Neat choose.

### Part 3 — The graph

`make_source()` builds the input side with `groups::RtspDecodedInput`: one ready-made group for
RTSP → depayload → parse → HW decode. `auto_caps_from_stream` reads the resolution from the stream.
`output_caps` fixes the format to NV12 at `width×height`.

`make_graph()` wires everything:

```cpp
graph.connect(make_source(cfg), branch);   // source → Branch
graph.connect(branch, video, live);        // Branch → VideoSender   (to Insight)
graph.connect(branch, infer, live);        // Branch → Model → Output("detections")
```

`live.policy = RealtimeLatestByStream` means: if a branch is busy, drop the frame instead of
queuing it. Latency stays low and the pipeline never backs up.

### Part 4 — Results → JSON

`decode_bbox_tensor()` turns the decoder output into `neat::Box` values (`x1, y1, x2, y2, score,
class_id`) in **frame pixels**. The letterbox is already undone. `to_json()` converts them to Insight's
`object-detection` format, with `bbox = [x, y, w, h]`.

### Part 5 — Main loop

```text
load config → load Model → build Graph → create MetadataSender
run = graph.build()                             ← start the pipeline
loop: run.pull("detections") → boxes → JSON → metadata.send_metadata(..., frame_id)
      stop on Ctrl+C, or when pull() says Closed (the stream ended)
run.close()
```

`frame_id` travels with every frame through decode, branch and model. Sending it with the metadata
lets Insight draw each box set on the exact video frame it belongs to.

---

## Create the project

```bash
mkdir -p /workspace/workshop/app1-single-stream/models
cd /workspace/workshop/app1-single-stream
```

### `main.cpp`

{{include:app1-single-stream/main.cpp}}

### `CMakeLists.txt`

{{include:app1-single-stream/CMakeLists.txt}}

### `app1.conf`

{{include:app1-single-stream/app1.conf}}

Replace **every** `<host-ip>` with the IP of the machine running the SDK and Insight. This command prints it:

```bash
curl -sk https://127.0.0.1:9900/api/server-ip
```

Check that the model from session 2 is in place:

```bash
ls models/yolo26n.compile_ready_mpk.tar.gz
```

---

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr
cmake --build build
```

Expected: `[100%] Built target app1_single_stream`.

---

## Run

Make sure `src1` is playing in Insight (**RTSP Source** tab), then:

```bash
dk ./build/app1_single_stream app1.conf
```

Expected output (Neat prints its pipelines first):

```text
source   : rtsp://<host-ip>:8554/src1
model    : ./models/yolo26n.compile_ready_mpk.tar.gz
insight  : <host-ip> channel 0 (video udp 9000, metadata udp 9100)
frames=100 fps=14.3 boxes=6          ← the first window includes the start-up time
frames=200 fps=20.0 boxes=6          ← steady state = the source frame rate (20 fps here)
frames=300 fps=20.0 boxes=5
```

Open Insight → **Video Viewer** → channel **0**. You should see the video with labelled boxes.
`Ctrl+C` stops the app cleanly.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `edit the config first, replace: …` | A `<host-ip>` placeholder is still in `app1.conf` |
| `[warn] no result in 5 s` | The stream is not playing in Insight, or the URL uses `127.0.0.1` instead of the host IP |
| Video in Insight, but no boxes | Same channel for video and metadata? Wrong `decode_type` for the model? Try `score_threshold=0.25` |
| Boxes are shifted or scaled | `width`/`height` in the config do not match the real source resolution |
| `Exec format error` | You ran the binary in the container. Use `dk` to run it on the DevKit |

---

## Exercises (if time allows)

1. Publish only people and cars: filter on `b.class_id` (0 = person, 2 = car) in `to_json()`.
2. Count objects per class each frame and print the counts every 100 frames.
3. Point `model_path` at the zoo BF16 archive from session 1 and compare the reported fps.

## Checkpoint

- [ ] Live video with boxes on Insight channel 0 from the RTSP stream
- [ ] You can name the hardware that runs each step: decode, preprocess, inference, box decode, encode

[← Session 2: Model compilation](02-model-compilation.md) · [Agenda](00-agenda.md) · [Next: App 2 →](04-app2-dual-stream.md)
