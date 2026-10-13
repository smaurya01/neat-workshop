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
                          │      (NV12 frames)      └─► "model"  YOLO26n: EV74 pre → MLA → EV74 box decode │
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
- **Postprocess** (EV74): `decode_type = YoloV26` turns the six raw heads into boxes. YOLO26's
  one-to-one heads give one box per object, so no NMS runs. `score_threshold` and `top_k` tune it
  (`nms_iou_threshold` only matters for NMS-based decoders such as `YoloV8`).
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

The metadata goes out with a timestamp of `-1`, so Insight draws each box set on the newest video
frame when it arrives. `frame_id` travels with every frame through decode, branch and model, and
identifies the frame in logs and diagnostics.

---

## Create the project

```bash
mkdir -p /workspace/workshop/app1-single-stream/models
cd /workspace/workshop/app1-single-stream
```

### `main.cpp`

<!-- file: app1-single-stream/main.cpp -->
```cpp
// Workshop App 1 — one source, one model, one output stream.
//
//   RTSP source -> H.264 HW decode -+-> VideoSender ---------> Insight video
//                                    |
//                                    +-> YOLO26n (MLA) -> "detections"
//                                                           |
//                                     app: pull -> boxes -> JSON -> MetadataSender -> Insight boxes
//
// Usage:  ./app1_single_stream app1.conf
#include <neat.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

// ---------------------------------------------------------------------------
// 1. Configuration: a plain key=value file
// ---------------------------------------------------------------------------
struct Config {
  std::string source;          // rtsp://<host-ip>:8554/<path>
  std::string model_path;      // compiled model archive (*_mpk.tar.gz)
  std::string insight_host;    // IP of the machine running Neat Insight
  int channel = 0;             // Insight channel: video 9000+ch, metadata 9100+ch
  int width = 1280;            // decoded frame size
  int height = 720;
  int fps = 30;
  float score_threshold = 0.4f;
};

Config load_config(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open config: " + path);
  std::map<std::string, std::string> kv;
  std::string line;
  while (std::getline(file, line)) {
    line = line.substr(0, line.find('#'));                     // strip comments
    const auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    auto trim = [](std::string s) {
      s.erase(0, s.find_first_not_of(" \t\r"));
      s.erase(s.find_last_not_of(" \t\r") + 1);
      return s;
    };
    kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    if (line.find('<') != std::string::npos)
      throw std::runtime_error("edit the config first, replace: " + trim(line));
  }
  Config c;
  c.source = kv["source"];
  c.model_path = kv["model_path"];
  c.insight_host = kv["insight_host"];
  if (kv.count("channel")) c.channel = std::stoi(kv["channel"]);
  if (kv.count("width")) c.width = std::stoi(kv["width"]);
  if (kv.count("height")) c.height = std::stoi(kv["height"]);
  if (kv.count("fps")) c.fps = std::stoi(kv["fps"]);
  if (kv.count("score_threshold")) c.score_threshold = std::stof(kv["score_threshold"]);
  if (c.source.empty() || c.model_path.empty() || c.insight_host.empty())
    throw std::runtime_error("config needs source, model_path and insight_host");
  return c;
}

// ---------------------------------------------------------------------------
// 2. The model: preprocess + MLA inference + box decode, all on the device
// ---------------------------------------------------------------------------
neat::Model::Options model_options(const Config& cfg) {
  neat::Model::Options opt;
  // Preprocess runs on the EV74 vision processor: NV12 -> RGB, letterbox to 640x640, x/255.
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.enable = neat::AutoFlag::On;
  opt.preprocess.input_max_width = cfg.width;
  opt.preprocess.input_max_height = cfg.height;
  opt.preprocess.input_max_depth = 3;
  opt.preprocess.resize.width = 640;
  opt.preprocess.resize.height = 640;
  opt.preprocess.resize.mode = neat::ResizeMode::Letterbox;
  opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  opt.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  opt.processcvu.pre_run_target = "EV74";
  opt.processcvu.post_run_target = "EV74";
  // Postprocess: decode the raw YOLO26 heads on the device (one-to-one heads: no NMS needed).
  opt.decode_type = neat::BoxDecodeType::YoloV26;
  opt.num_classes = 80;
  opt.score_threshold = cfg.score_threshold;
  opt.nms_iou_threshold = 0.5f;
  opt.top_k = 100;
  return opt;
}

// ---------------------------------------------------------------------------
// 3. The graph
// ---------------------------------------------------------------------------
neat::Graph make_source(const Config& cfg) {
  groups::RtspDecodedInputOptions o;                // RTSP -> depay -> parse -> HW decode
  o.url = cfg.source;
  o.tcp = true;
  o.latency_ms = 200;
  o.out_format = neat::FormatTag::NV12;
  o.auto_caps_from_stream = true;
  o.fallback_h264_width = cfg.width;
  o.fallback_h264_height = cfg.height;
  o.fallback_h264_fps = cfg.fps;
  o.output_caps.enable = true;
  o.output_caps.format = neat::FormatTag::NV12;
  o.output_caps.width = cfg.width;
  o.output_caps.height = cfg.height;
  o.output_caps.fps = cfg.fps;
  o.output_caps.memory = neat::CapsMemory::Any;
  return groups::RtspDecodedInput(o);
}

neat::Graph make_graph(const Config& cfg, const neat::Model& model) {
  // Video branch: re-encode the clean frames to H.264 and send RTP/UDP to Insight.
  auto video_opt = groups::VideoSenderOptions::H264RtpUdpFromRaw(cfg.width, cfg.height, cfg.fps);
  video_opt.host = cfg.insight_host;
  video_opt.channel = cfg.channel;
  neat::Graph video("video");
  video.connect(neat::nodes::Input("video"), groups::VideoSender(video_opt));

  // Model branch: the Model is itself a graph fragment; its result lands on "detections".
  neat::Graph infer("infer");
  infer.connect(neat::nodes::Input("model"), model);
  infer.add(neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(4)));

  // Wire it: source -> Branch fans out every frame to both paths.
  neat::GraphLinkOptions live;
  live.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;  // drop, never queue up, if slow
  neat::Graph graph("app1");
  auto branch = neat::graphs::Branch("source", {"video", "model"});
  graph.connect(make_source(cfg), branch);
  graph.connect(branch, video, live);
  graph.connect(branch, infer, live);
  return graph;
}

// ---------------------------------------------------------------------------
// 4. Results -> Insight metadata JSON
// ---------------------------------------------------------------------------
constexpr std::array<const char*, 80> kCoco = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

// Insight "object-detection" format: bbox = [x, y, w, h] in frame pixels.
std::string to_json(const std::vector<neat::Box>& boxes, const Config& cfg) {
  std::ostringstream js;
  js << "{\"objects\":[";
  int n = 0;
  for (const auto& b : boxes) {
    if (b.score < cfg.score_threshold) continue;
    const int x = std::max(0, static_cast<int>(b.x1));
    const int y = std::max(0, static_cast<int>(b.y1));
    const int w = std::min(cfg.width - x, static_cast<int>(b.x2 - b.x1));
    const int h = std::min(cfg.height - y, static_cast<int>(b.y2 - b.y1));
    if (w <= 0 || h <= 0) continue;
    const char* label = (b.class_id >= 0 && b.class_id < 80) ? kCoco[b.class_id] : "object";
    if (n++) js << ",";
    js << "{\"id\":\"obj_" << n << "\",\"label\":\"" << label
       << "\",\"confidence\":" << b.score << ",\"bbox\":[" << x << "," << y << "," << w << ","
       << h << "]}";
  }
  js << "]}";
  return js.str();
}

// ---------------------------------------------------------------------------
// 5. Main loop
// ---------------------------------------------------------------------------
std::atomic<bool> g_stop{false};

int main(int argc, char** argv) {
  std::signal(SIGINT, [](int) { g_stop = true; });
  std::signal(SIGTERM, [](int) { g_stop = true; });
  try {
    std::cout << std::fixed << std::setprecision(1);   // fps=30.0
    const Config cfg = load_config(argc > 1 ? argv[1] : "app1.conf");
    neat::Model model(cfg.model_path, model_options(cfg));
    neat::Graph graph = make_graph(cfg, model);

    neat::MetadataSenderOptions meta_opt;
    meta_opt.host = cfg.insight_host;
    meta_opt.channel = cfg.channel;
    std::string err;
    neat::MetadataSender metadata(meta_opt, &err);
    if (!metadata.ok()) throw std::runtime_error("metadata sender: " + err);

    std::cout << "source   : " << cfg.source
              << "\nmodel    : " << cfg.model_path << "\ninsight  : " << cfg.insight_host
              << " channel " << cfg.channel << " (video udp " << 9000 + cfg.channel
              << ", metadata udp " << 9100 + cfg.channel << ")" << std::endl;

    int frames = 0;
    auto t0 = std::chrono::steady_clock::now();
    neat::Run run = graph.build();            // start the pipeline
    while (!g_stop) {
      neat::Sample sample;
      neat::PullError perr;
      const auto status = run.pull("detections", 5000, sample, &perr);
      if (status == neat::PullStatus::Closed) break;                // stream ended
      if (status == neat::PullStatus::Timeout) { std::cerr << "[warn] no result in 5 s\n"; continue; }
      if (status != neat::PullStatus::Ok) throw std::runtime_error("pull: " + perr.message);

      const auto tensors = neat::tensors_from_sample(sample, false);
      if (tensors.empty()) continue;
      const auto boxes =
          neat::decode_bbox_tensor(tensors.front(), cfg.width, cfg.height, 100, false).boxes;
      // timestamp -1: Insight draws the boxes on the newest frame; frame_id is for diagnostics.
      metadata.send_metadata("object-detection", to_json(boxes, cfg), -1,
                             std::to_string(sample.frame_id), nullptr);

      if (++frames % 100 == 0) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "frames=" << frames << " fps=" << 100.0 / s << " boxes=" << boxes.size() << std::endl;
        t0 = std::chrono::steady_clock::now();
      }
    }
    run.close();
    std::cout << "done, " << frames << " frames" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 1;
  }
}
```

### `CMakeLists.txt`

<!-- file: app1-single-stream/CMakeLists.txt -->
```cmake
cmake_minimum_required(VERSION 3.16)
project(app1_single_stream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(app1_single_stream main.cpp)
target_link_libraries(app1_single_stream PRIVATE SimaNeat::sima_neat)
```

### `app1.conf`

<!-- file: app1-single-stream/app1.conf -->
```ini
# Source: RTSP URL of the video stream
source=rtsp://<host-ip>:8554/src1

# Compiled model archive from the model-compilation session
model_path=./models/yolo26n.compile_ready_mpk.tar.gz

# Neat Insight: host IP and channel (video udp 9000+channel, metadata udp 9100+channel)
insight_host=<host-ip>
channel=0

# Decoded frame size and rate. Must match the stream exactly (the 720p30 catalog videos are 30 fps)
width=1280
height=720
fps=30

score_threshold=0.4
```

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

Make sure `src1` is playing in Insight (**Streaming** tab), then:

```bash
dk ./build/app1_single_stream app1.conf
```

Expected output (`dk` prefixes each line with `[DEVKIT][STDOUT]`; Neat prints its start-up lines and
pipelines first):

```text
source   : rtsp://<host-ip>:8554/src1
model    : ./models/yolo26n.compile_ready_mpk.tar.gz
insight  : <host-ip> channel 0 (video udp 9000, metadata udp 9100)
frames=100 fps=17.6 boxes=9          ← the first window includes the start-up time
frames=200 fps=30.2 boxes=8          ← steady state = the source frame rate (30 fps here)
frames=300 fps=29.8 boxes=7
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
| `[misconfig.media_caps] Two connected pipeline stages require incompatible media caps` | `fps` (or `width`/`height`) in the config does not match the stream. Neat 0.5.0 requires an exact match: use the 720p30 catalog variant, or set `fps` to your stream's rate |
| `Exec format error` | You ran the binary in the container. Use `dk` to run it on the DevKit |

---

## Exercises (if time allows)

1. Publish only people and cars: filter on `b.class_id` (0 = person, 2 = car) in `to_json()`.
2. Count objects per class each frame and print the counts every 100 frames.
3. Point `model_path` at a BF16 detection archive (the zoo archive from session 1, or the prebuilt
   `yolo26n-bf16…`) and compare the boxes. The app's fps stays at the source rate, so compare speed
   with `model_info` instead.

## Checkpoint

- [ ] Live video with boxes on Insight channel 0 from the RTSP stream
- [ ] You can name the hardware that runs each step: decode, preprocess, inference, box decode, encode

[← Session 2: Model compilation](02-model-compilation.md) · [Agenda](00-agenda.md) · [Next: App 2 →](04-app2-dual-stream.md)
