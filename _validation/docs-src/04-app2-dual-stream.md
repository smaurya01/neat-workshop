# Session 4 — App 2: Two Sources, Two Models, Two Streams (30 min)

[← Session 3: App 1](03-app1-single-stream.md) · [Agenda](00-agenda.md)

**Goal:** scale App 1 to two independent pipelines on one DevKit, running at the same time:

| Stream | Source | Model | Insight output |
|---|---|---|---|
| 0 | `src1` (traffic) | YOLO26n **detection**, INT8 | channel 0: boxes |
| 1 | `src2` (people) | YOLO26n **pose**, BF16 | channel 1: skeletons |

| Time | Topic |
|---|---|
| 10 min | What changes from App 1: one graph for N streams, a decoder per task, a thread per stream |
| 5 min | Create the files, build |
| 10 min | Run, view both channels in Insight |
| 5 min | Exercises / discussion |

---

## What we are building

```text
 ┌────────────────────────────────── one Graph, one Run ───────────────────────────────────┐
 │                                                                                         │
 │ source0 ─► decode ─► Branch "source0" ─┬─► "video0" VideoSender (channel 0) ────────────┼─► Insight ch 0 video
 │                                        └─► "model0" YOLO26n det  INT8 ─► Output "results0"
 │                                                                                         │
 │ source1 ─► decode ─► Branch "source1" ─┬─► "video1" VideoSender (channel 1) ────────────┼─► Insight ch 1 video
 │                                        └─► "model1" YOLO26n pose BF16 ─► Output "results1"
 └─────────────────────────────────────────────────────────────────┬───────────────┬───────┘
                                                                   │               │
                       thread 0: pull("results0") → boxes JSON ────┘               │  ─► Insight ch 0 boxes
                       thread 1: pull("results1") → pose JSON  ────────────────────┘  ─► Insight ch 1 skeletons
```

Both models share the one MLA. Neat schedules their inferences, and you do not manage the hardware.

---

## What changes from App 1

The source, branch, video and model code is **the same as App 1**. Five things change:

### 1. Config: one block per stream

```ini
insight_host=<host-ip>
stream0.source=...      stream0.model_path=...      stream0.task=detection
stream1.source=...      stream1.model_path=...      stream1.task=pose
```

`load_config()` reads `stream0.*`, `stream1.*`, … until there are no more. The stream number is also
its Insight channel. To add a third stream, add a `stream2.*` block. No code changes are needed.

### 2. One decoder setting per task (`model_options`)

Preprocessing is identical. Only the postprocess differs:

```cpp
if (s.task == Task::Pose) {
  opt.decode_type = neat::BoxDecodeType::YoloV26Pose;   // boxes + 17 keypoints
  opt.num_classes = 1;                                   // "person" only
} else {
  opt.decode_type = neat::BoxDecodeType::YoloV26;        // boxes
  opt.num_classes = 80;                                  // COCO
}
```

### 3. One graph for all streams, with unique names (`add_stream`)

`add_stream()` is App 1's `make_graph()`, called once per stream on a **shared** graph. Every
endpoint name gets the stream number (`source0`, `video0`, `model0`, `results0`, …), so the two
pipelines never collide. One graph and one `graph.build()` mean one `Run` that drives
both pipelines. Neat can then plan both hardware decoders together.

### 4. Pose results → JSON (`pose_json`)

```cpp
auto pose  = neat::decode_pose_tensor(tensor, width, height, 100, false);
// pose.boxes     : [N, 6]      x1 y1 x2 y2 score class
// pose.keypoints : [N, 17, 3]  x y visibility  (nose, left_eye, ..., right_ankle)
```

The keypoints are already in frame pixels, and the padding channels from session 2 are already removed.
`pose_json()` writes Insight's `pose-estimation` format: one entry per person, with a `bbox` and 17
named keypoints. Insight draws the skeleton from the keypoint names.

### 5. One thread per stream (`stream_worker`)

Each stream has its own worker thread. The thread owns its own `MetadataSender` (on its own channel)
and pulls only its own endpoint:

```cpp
run.pull(results_endpoint(s.id), 1000, sample, &perr);    // "results0" or "results1"
```

The two streams are fully independent. A slow stream never blocks the other one. `Ctrl+C`
sets `g_stop`, both workers exit, and `main` calls `run.close()`.

---

## Create the project

```bash
mkdir -p /workspace/workshop/app2-dual-stream/models
cd /workspace/workshop/app2-dual-stream
```

### `main.cpp`

{{include:app2-dual-stream/main.cpp}}

### `CMakeLists.txt`

{{include:app2-dual-stream/CMakeLists.txt}}

### `app2.conf`

{{include:app2-dual-stream/app2.conf}}

Replace every `<host-ip>`. Make sure `src1` (traffic) and `src2` (people) are playing in Insight, and that both models are in place:

```bash
ls models/
# yolo26n-pose.compile_ready_mpk.tar.gz  yolo26n.compile_ready_mpk.tar.gz
```

---

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr
cmake --build build
dk ./build/app2_dual_stream app2.conf
```

Expected:

```text
stream 0: rtsp://<host-ip>:8554/src1 -> ./models/yolo26n.compile_ready_mpk.tar.gz -> Insight channel 0
stream 1: rtsp://<host-ip>:8554/src2 -> ./models/yolo26n-pose.compile_ready_mpk.tar.gz -> Insight channel 1
[stream 1 pose] frames=100 fps=14.3 results=2
[stream 0 det ] frames=100 fps=14.2 results=4
[stream 1 pose] frames=200 fps=20.0 results=2
[stream 0 det ] frames=200 fps=20.0 results=4
...
^C
[stream 1] stopped after 467 frames
[stream 0] stopped after 468 frames
```

Both streams run at the full source rate. `results` is the number of objects (stream 0) or people (stream 1) in the last frame.

In Insight → **Video Viewer**: channel **0** shows traffic with boxes, and channel **1** shows people
with skeletons.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Channel 1 shows video but no skeletons | Is `stream1.task=pose` set? Is the pose model on stream 1? Are there people in the video? |
| Skeletons are offset from people | `stream1.width/height` do not match the source resolution |
| `pull: ...` error on one stream | Read the message. The model path or source of that stream is usually wrong. The other stream stops too, by design |
| Lower fps on both streams | Expected only when the MLA is saturated. At 2 × 20 fps with these models it is far below its limit (`model_info`: 550 + 199 fps) |

---

## Exercises and discussion

1. **Add a third stream**: add a `stream2.*` block (for example `src1` again, with the pose model) and
   watch channel 2. No code changes are needed.
2. **Swap precisions**: compile the pose model in INT8 (`--precision int8`) and compare keypoint
   stability and fps with BF16.
3. **Trigger logic**: in `stream_worker`, print a message when a person's `left_wrist` is above the `nose`
   (a raised hand). This is the start of a real application.
4. **Discussion**: why does one graph with N streams scale better than N separate processes? Think
   about decoder planning and the shared MLA.

## Checkpoint

- [ ] Two live channels in Insight: boxes on channel 0, skeletons on channel 1
- [ ] You can explain how endpoint names keep the two pipelines apart in one graph

[← Session 3: App 1](03-app1-single-stream.md) · [Agenda](00-agenda.md)
