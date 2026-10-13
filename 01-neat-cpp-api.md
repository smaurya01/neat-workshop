# Session 1 — Neat C++ API Introduction (20 min)

[← Agenda](00-agenda.md) · [Next: Model compilation →](02-model-compilation.md)

**Goal:** understand the five objects every Neat application is built from, and build and run your
first Neat program on the DevKit.

| Time | Topic |
|---|---|
| 5 min | What Neat is and where things run on Modalix |
| 8 min | The core objects: `Model`, `Graph`, nodes, `Run`, results |
| 7 min | Hands-on: `model_info`, which loads a model, prints its tensors and benchmarks it |

---

## 1. What Neat is

**Neat** is SiMa's application library for Modalix. You describe *what* the pipeline should do
(decode this stream, run this model, send the result there), and Neat builds a hardware
pipeline that places each step on the right engine:

| Engine on Modalix | What Neat runs there |
|---|---|
| **Video codec** (hardware) | H.264 / H.265 decode of RTSP input, H.264 encode of output video |
| **EV74** vision DSP | Preprocessing (resize, letterbox, colour convert, normalize) and postprocessing (box decode, NMS) |
| **MLA** (ML accelerator) | The neural network itself, from the compiled `.tar.gz` model archive |
| **A65** ARM CPU | Your application code: reading results, business logic, sending metadata |

You write C++ against one header:

```cpp
#include <neat.h>
namespace neat = simaai::neat;
```

The same concepts are available in Python as `pyneat`.

---

## 2. The core objects

```text
   Model ─────────► a compiled network + its pre/post processing, loaded from *_mpk.tar.gz
   Node / Group ──► one pipeline step (SimaDecode, ...) or a ready-made chain (RtspDecodedInput, VideoSender)
   Graph ─────────► how the steps are wired: add() for a chain, connect() for branches
   Run ───────────► the live, running pipeline built from a Graph: pull() results, close() it
   Sample/Tensor ─► the data that comes out: tensors plus frame_id and timestamps
```

### 2.1 `Model`: load a compiled network

```cpp
neat::Model::Options opt;
opt.preprocess.resize.width  = 640;                                   // model input size
opt.preprocess.resize.height = 640;
opt.preprocess.resize.mode   = neat::ResizeMode::Letterbox;
opt.preprocess.color_convert.input_format  = neat::PreprocessColorFormat::NV12;  // from decoder
opt.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;   // what the net wants
opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;             // pixel / 255

opt.decode_type     = neat::BoxDecodeType::YoloV26;                   // box decode on the device
opt.num_classes     = 80;
opt.score_threshold = 0.4f;

neat::Model model("models/yolo26n.compile_ready_mpk.tar.gz", opt);
```

- The **archive** holds the MLA program. The **options** describe the pre- and postprocessing
  around it. Neat places both on the EV74, so frames never take a round trip through the CPU.
- `decode_type` picks the decoder that matches the network's raw outputs: `YoloV26` for
  YOLO26 detection (its one-to-one heads need no NMS), `YoloV26Pose` for YOLO26 pose, `YoloV8`, `YoloX` and so on.
- `model.input_specs()` / `model.output_specs()` describe the tensors. `model.benchmark()` measures
  latency and throughput on the device.

### 2.2 Nodes and groups: the pipeline steps

| Building block | What it does |
|---|---|
| `nodes::Input("name")` / `nodes::Output("name")` | Named entry and exit points of a graph |
| `nodes::H264ParseAu`, `SimaDecode` | Single steps: parse H.264, hardware decode |
| `groups::RtspDecodedInput(opts)` | Ready-made chain: RTSP → depayload → parse → hardware decode → NV12 frames |
| `groups::VideoSender(opts)` | Ready-made chain: frames → hardware H.264 encode → RTP/UDP (to Insight) |
| `graphs::Branch("src", {"a", "b"})` | Fan one stream out to several paths |
| `nodes::Custom("<gst fragment>")` | Escape hatch: insert any GStreamer element |

### 2.3 `Graph`: wire the steps together

```cpp
neat::Graph graph("my_app");
graph.add(step1);              // add(): linear chain, step1 -> step2 -> ...
graph.add(step2);

graph.connect(source, branch); // connect(): explicit edges between sub-graphs, for branches
graph.connect(branch, video_path);
graph.connect(branch, model_path);
```

A `Model` is a graph fragment too, so it drops straight into a graph:

```cpp
neat::Graph infer("infer");
infer.connect(neat::nodes::Input("model"), model);                   // frames in
infer.add(neat::nodes::Output("detections"));                         // results out, by name
```

### 2.4 `Run`: execute and read results

```cpp
neat::Run run = graph.build();                     // start the hardware pipeline

neat::Sample sample;
neat::PullError err;
auto status = run.pull("detections", 5000, sample, &err);   // wait up to 5 s for a result
if (status == neat::PullStatus::Ok) { /* use sample */ }
if (status == neat::PullStatus::Closed) { /* end of stream */ }

run.close();                                        // stop and release the hardware
```

### 2.5 Results → boxes → Insight

```cpp
auto tensors = neat::tensors_from_sample(sample, false);
auto boxes   = neat::decode_bbox_tensor(tensors.front(), width, height, 100, false).boxes;
for (const neat::Box& b : boxes) { /* b.x1 b.y1 b.x2 b.y2 b.score b.class_id */ }

// pose models: neat::decode_pose_tensor(...) -> boxes [N,6] + keypoints [N,17,3]

neat::MetadataSender metadata(meta_opt, &err_text);          // JSON over UDP to Insight
metadata.send_metadata("object-detection", json, -1, std::to_string(sample.frame_id), nullptr);
```

A timestamp of `-1` tells Insight to draw each set of boxes on the newest video frame when it
arrives; `frame_id` identifies the frame in logs and diagnostics.

### 2.6 The development loop

```text
 edit main.cpp ──► cmake --build build ──► dk ./build/app config ──► watch Insight
 (SDK container)    (cross-compiles ARM64)   (runs on the DevKit)     (browser)
```

Every app in this workshop uses the same `CMakeLists.txt` pattern: `find_package(SimaNeat)` plus
one `target_link_libraries`.

---

## 3. Hands-on: `model_info`

A 40-line program that loads any compiled model archive, prints its input and output tensors, and
benchmarks it on the MLA. You will run it again in session 2 to compare your INT8 and BF16 models.

### 3.1 Create the project

```bash
mkdir -p /workspace/workshop/model-info && cd /workspace/workshop/model-info
```

Create `main.cpp`:

<!-- file: model-info/main.cpp -->
```cpp
// model_info: load a compiled model archive, print its inputs/outputs, and benchmark it.
//
// Usage:  ./model_info <model_mpk.tar.gz>
#include <neat.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace neat = simaai::neat;

std::string shape_str(const std::vector<int64_t>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
  return s + "]";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <model_mpk.tar.gz>\n";
    return 1;
  }
  try {
    neat::Model model(argv[1]);                    // load + validate the archive

    std::cout << "inputs:\n";
    for (const auto& spec : model.input_specs()) std::cout << "  " << shape_str(spec.shape) << "\n";
    std::cout << "outputs:\n";
    for (const auto& spec : model.output_specs()) std::cout << "  " << shape_str(spec.shape) << "\n";

    // Synthetic inputs, warm-up, then single-frame latency and pipelined throughput.
    const neat::BenchmarkReport report = model.benchmark(200);
    std::cout << std::fixed << std::setprecision(2) << "latency_ms=" << report.latency_ms
              << "  fps=" << std::setprecision(1) << report.fps << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 1;
  }
}
```

Create `CMakeLists.txt`:

<!-- file: model-info/CMakeLists.txt -->
```cmake
cmake_minimum_required(VERSION 3.16)
project(model_info LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(model_info main.cpp)
target_link_libraries(model_info PRIVATE SimaNeat::sima_neat)
```

### 3.2 Get a model to inspect

For now, use the YOLO26n detection model from the SiMa Model Zoo (your own compiled models come in
session 2):

```bash
mkdir -p models && cd models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.3/models/modalix/yolo26-detection/yolo26n-det-bf16-mla_tess-b1.tar.gz
cd ..
```

(No internet or login? Ask the instructor for the prebuilt archive.)

### 3.3 Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr
cmake --build build
dk ./build/model_info models/yolo26n-det-bf16-mla_tess-b1.tar.gz
```

Expected output (`dk` prefixes each line with `[DEVKIT][STDOUT]`; Neat also prints its own start-up
lines, an advisory `[WARN]` you can ignore, and a benchmark summary):

```text
inputs:
  [640, 640, 3]
outputs:
  [80, 80, 4]
  [40, 40, 4]
  [20, 20, 4]
  [80, 80, 80]
  [40, 40, 80]
  [20, 20, 80]
...
latency_ms=8.10  fps=243.6
```

### 3.4 How to read it

- **Input `[640, 640, 3]`**: one RGB image, channels last (HWC), the layout the MLA wants.
- **Six outputs**: YOLO26 has three detection scales (80×80, 40×40 and 20×20 grid cells, at strides 8,
  16 and 32). Each scale has a **box head** (4 channels: distances to left, top, right and bottom)
  and a **class head** (80 channels, one score per COCO class). These are *raw* tensors. In the
  apps, `decode_type = YoloV26` turns them into boxes on the EV74.
- **latency / fps**: latency is one frame in, one result out. fps is the pipelined throughput,
  with several frames in flight on the MLA at once.

---

[← Agenda](00-agenda.md) · [Next: Model compilation →](02-model-compilation.md)
