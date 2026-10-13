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

<!-- file: app2-dual-stream/main.cpp -->
```cpp
// Workshop App 2 — two sources, two models, two output streams, ONE graph.
//
//   stream 0: source0 -> decode -+-> VideoSender (Insight channel 0)
//                                +-> YOLO26n detection (INT8) -> "results0" -> thread 0 -> boxes JSON
//
//   stream 1: source1 -> decode -+-> VideoSender (Insight channel 1)
//                                +-> YOLO26n-pose      (BF16) -> "results1" -> thread 1 -> pose JSON
//
// Both chains live in one Graph and one Run. Each stream gets its own pull thread.
//
// Usage:  ./app2_dual_stream app2.conf
#include <neat.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

// ---------------------------------------------------------------------------
// 1. Configuration: shared keys + one "streamN." block per stream
// ---------------------------------------------------------------------------
enum class Task { Detection, Pose };

struct StreamConfig {
  int id = 0;                  // also the Insight channel: video 9000+id, metadata 9100+id
  std::string source;          // rtsp://<host-ip>:8554/<path>
  std::string model_path;      // compiled model archive
  Task task = Task::Detection;
  int width = 1280;
  int height = 720;
  int fps = 30;
};

struct Config {
  std::string insight_host;
  float score_threshold = 0.4f;
  std::vector<StreamConfig> streams;
};

Config load_config(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open config: " + path);
  std::map<std::string, std::string> kv;
  std::string line;
  while (std::getline(file, line)) {
    line = line.substr(0, line.find('#'));
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
  c.insight_host = kv["insight_host"];
  if (kv.count("score_threshold")) c.score_threshold = std::stof(kv["score_threshold"]);
  for (int i = 0; kv.count("stream" + std::to_string(i) + ".source"); ++i) {
    const std::string p = "stream" + std::to_string(i) + ".";
    StreamConfig s;
    s.id = i;
    s.source = kv[p + "source"];
    s.model_path = kv[p + "model_path"];
    s.task = kv[p + "task"] == "pose" ? Task::Pose : Task::Detection;
    if (kv.count(p + "width")) s.width = std::stoi(kv[p + "width"]);
    if (kv.count(p + "height")) s.height = std::stoi(kv[p + "height"]);
    if (kv.count(p + "fps")) s.fps = std::stoi(kv[p + "fps"]);
    c.streams.push_back(s);
  }
  if (c.insight_host.empty() || c.streams.empty())
    throw std::runtime_error("config needs insight_host and at least stream0.source");
  return c;
}

// ---------------------------------------------------------------------------
// 2. Models: same preprocess, different decoder per task
// ---------------------------------------------------------------------------
neat::Model::Options model_options(const StreamConfig& s, float score_threshold) {
  neat::Model::Options opt;
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.enable = neat::AutoFlag::On;
  opt.preprocess.input_max_width = s.width;
  opt.preprocess.input_max_height = s.height;
  opt.preprocess.input_max_depth = 3;
  opt.preprocess.resize.width = 640;
  opt.preprocess.resize.height = 640;
  opt.preprocess.resize.mode = neat::ResizeMode::Letterbox;
  opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  opt.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  opt.processcvu.pre_run_target = "EV74";
  opt.processcvu.post_run_target = "EV74";
  opt.score_threshold = score_threshold;
  opt.nms_iou_threshold = 0.5f;
  opt.top_k = 100;
  if (s.task == Task::Pose) {
    opt.decode_type = neat::BoxDecodeType::YoloV26Pose;   // boxes + 17 keypoints
    opt.num_classes = 1;                                   // "person" only
  } else {
    opt.decode_type = neat::BoxDecodeType::YoloV26;        // boxes
    opt.num_classes = 80;                                  // COCO
  }
  return opt;
}

// ---------------------------------------------------------------------------
// 3. Graph: the App 1 chain, built once per stream with unique endpoint names
// ---------------------------------------------------------------------------
neat::Graph make_source(const StreamConfig& s) {
  groups::RtspDecodedInputOptions o;
  o.url = s.source;
  o.tcp = true;
  o.latency_ms = 200;
  o.out_format = neat::FormatTag::NV12;
  o.auto_caps_from_stream = true;
  o.fallback_h264_width = s.width;
  o.fallback_h264_height = s.height;
  o.fallback_h264_fps = s.fps;
  o.output_caps.enable = true;
  o.output_caps.format = neat::FormatTag::NV12;
  o.output_caps.width = s.width;
  o.output_caps.height = s.height;
  o.output_caps.fps = s.fps;
  o.output_caps.memory = neat::CapsMemory::Any;
  return groups::RtspDecodedInput(o);
}

std::string results_endpoint(int id) {
  return "results" + std::to_string(id);
}

// Adds one complete stream (source -> branch -> video + model) to the shared graph.
void add_stream(neat::Graph& graph, const StreamConfig& s, const neat::Model& model,
                const std::string& insight_host) {
  const std::string n = std::to_string(s.id);

  auto video_opt = groups::VideoSenderOptions::H264RtpUdpFromRaw(s.width, s.height, s.fps);
  video_opt.host = insight_host;
  video_opt.channel = s.id;
  neat::Graph video("video" + n);
  video.connect(neat::nodes::Input("video" + n), groups::VideoSender(video_opt));

  neat::Graph infer("infer" + n);
  infer.connect(neat::nodes::Input("model" + n), model);
  infer.add(neat::nodes::Output(results_endpoint(s.id), neat::OutputOptions::EveryFrame(4)));

  neat::GraphLinkOptions live;
  live.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  auto branch = neat::graphs::Branch("source" + n, {"video" + n, "model" + n});
  graph.connect(make_source(s), branch);
  graph.connect(branch, video, live);
  graph.connect(branch, infer, live);
}

// ---------------------------------------------------------------------------
// 4. Results -> Insight metadata JSON (one format per task)
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

constexpr std::array<const char*, 17> kKeypoints = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"};

// Detection -> {"objects":[{label, confidence, bbox:[x,y,w,h]}]}
std::string detection_json(const neat::Tensor& t, const StreamConfig& s, float min_score,
                           int& count) {
  const auto boxes = neat::decode_bbox_tensor(t, s.width, s.height, 100, false).boxes;
  std::ostringstream js;
  js << "{\"objects\":[";
  count = 0;
  for (const auto& b : boxes) {
    if (b.score < min_score) continue;
    const int x = std::max(0, static_cast<int>(b.x1));
    const int y = std::max(0, static_cast<int>(b.y1));
    const int w = std::min(s.width - x, static_cast<int>(b.x2 - b.x1));
    const int h = std::min(s.height - y, static_cast<int>(b.y2 - b.y1));
    if (w <= 0 || h <= 0) continue;
    const char* label = (b.class_id >= 0 && b.class_id < 80) ? kCoco[b.class_id] : "object";
    if (count++) js << ",";
    js << "{\"id\":\"obj_" << count << "\",\"label\":\"" << label
       << "\",\"confidence\":" << b.score << ",\"bbox\":[" << x << "," << y << "," << w << ","
       << h << "]}";
  }
  js << "]}";
  return js.str();
}

// Copy a small decoded CPU tensor into floats (a frame with nobody in it has 0 rows).
std::vector<float> to_floats(const neat::Tensor& t) {
  if (t.shape.empty() || t.shape[0] == 0) return {};
  const auto bytes = t.contiguous().copy_payload_bytes();
  std::vector<float> v(bytes.size() / sizeof(float));
  std::memcpy(v.data(), bytes.data(), v.size() * sizeof(float));
  return v;
}

// Pose -> {"poses":[{label, confidence, bbox, keypoints:[{name,x,y,confidence} x17]}]}
std::string pose_json(const neat::Tensor& t, const StreamConfig& s, int& count) {
  const auto pose = neat::decode_pose_tensor(t, s.width, s.height, 100, false);
  const auto boxes = to_floats(pose.boxes);       // [N, 6]  x1 y1 x2 y2 score class
  const auto kpts = to_floats(pose.keypoints);    // [N, 17, 3]  x y visibility
  count = static_cast<int>(boxes.size() / 6);
  std::ostringstream js;
  js << "{\"poses\":[";
  for (int i = 0; i < count; ++i) {
    const float* b = &boxes[i * 6];
    js << (i ? "," : "") << "{\"id\":\"pose_" << i + 1 << "\",\"label\":\"person\""
       << ",\"confidence\":" << b[4] << ",\"bbox\":[" << std::lround(b[0]) << ","
       << std::lround(b[1]) << "," << std::lround(b[2] - b[0]) << ","
       << std::lround(b[3] - b[1]) << "],\"keypoints\":[";
    for (int k = 0; k < 17; ++k) {
      const float* p = &kpts[(i * 17 + k) * 3];
      js << (k ? "," : "") << "{\"name\":\"" << kKeypoints[k] << "\",\"x\":" << std::lround(p[0])
         << ",\"y\":" << std::lround(p[1]) << ",\"confidence\":" << p[2] << "}";
    }
    js << "]}";
  }
  js << "]}";
  return js.str();
}

// ---------------------------------------------------------------------------
// 5. One worker thread per stream: pull -> JSON -> MetadataSender
// ---------------------------------------------------------------------------
std::atomic<bool> g_stop{false};
std::mutex print_mutex;

void stream_worker(neat::Run& run, const StreamConfig& s, const Config& cfg) {
  neat::MetadataSenderOptions meta_opt;
  meta_opt.host = cfg.insight_host;
  meta_opt.channel = s.id;
  std::string err;
  neat::MetadataSender metadata(meta_opt, &err);
  if (!metadata.ok()) throw std::runtime_error("metadata sender: " + err);

  const char* kind = s.task == Task::Pose ? "pose-estimation" : "object-detection";
  int frames = 0;
  auto t0 = std::chrono::steady_clock::now();
  while (!g_stop) {
    neat::Sample sample;
    neat::PullError perr;
    const auto status = run.pull(results_endpoint(s.id), 1000, sample, &perr);
    if (status == neat::PullStatus::Closed) break;
    if (status == neat::PullStatus::Timeout) continue;
    if (status != neat::PullStatus::Ok) throw std::runtime_error("pull: " + perr.message);

    const auto tensors = neat::tensors_from_sample(sample, false);
    if (tensors.empty()) continue;
    int count = 0;
    const std::string json = s.task == Task::Pose
                                 ? pose_json(tensors.front(), s, count)
                                 : detection_json(tensors.front(), s, cfg.score_threshold, count);
    metadata.send_metadata(kind, json, -1, std::to_string(sample.frame_id), nullptr);

    if (++frames % 100 == 0) {
      const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::lock_guard<std::mutex> lock(print_mutex);
      std::cout << "[stream " << s.id << " " << (s.task == Task::Pose ? "pose" : "det ")
                << "] frames=" << frames << " fps=" << 100.0 / sec << " results=" << count << std::endl;
      t0 = std::chrono::steady_clock::now();
    }
  }
  std::lock_guard<std::mutex> lock(print_mutex);
  std::cout << "[stream " << s.id << "] stopped after " << frames << " frames" << std::endl;
}

int main(int argc, char** argv) {
  std::signal(SIGINT, [](int) { g_stop = true; });
  std::signal(SIGTERM, [](int) { g_stop = true; });
  try {
    std::cout << std::fixed << std::setprecision(1);   // fps=30.0
    const Config cfg = load_config(argc > 1 ? argv[1] : "app2.conf");

    // Load every model first, then add every stream to ONE graph.
    std::vector<std::unique_ptr<neat::Model>> models;
    neat::Graph graph("app2");
    for (const auto& s : cfg.streams) {
      models.push_back(std::make_unique<neat::Model>(s.model_path,
                                                     model_options(s, cfg.score_threshold)));
      add_stream(graph, s, *models.back(), cfg.insight_host);
      std::cout << "stream " << s.id << ": " << s.source << " -> " << s.model_path
                << " -> Insight channel " << s.id << std::endl;
    }

    neat::Run run = graph.build();               // one Run drives every stream

    std::vector<std::thread> workers;
    std::atomic<bool> failed{false};
    for (const auto& s : cfg.streams) {
      workers.emplace_back([&run, &cfg, &failed, s] {
        try {
          stream_worker(run, s, cfg);
        } catch (const std::exception& e) {
          std::cerr << "[error] stream " << s.id << ": " << e.what() << "\n";
          failed = true;
          g_stop = true;
        }
      });
    }
    for (auto& w : workers) w.join();
    run.close();
    return failed ? 1 : 0;
  } catch (const std::exception& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 1;
  }
}
```

### `CMakeLists.txt`

<!-- file: app2-dual-stream/CMakeLists.txt -->
```cmake
cmake_minimum_required(VERSION 3.16)
project(app2_dual_stream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(app2_dual_stream main.cpp)
target_link_libraries(app2_dual_stream PRIVATE SimaNeat::sima_neat)
```

### `app2.conf`

<!-- file: app2-dual-stream/app2.conf -->
```ini
# Neat Insight host. Stream N is shown on Insight channel N
# (video udp 9000+N, metadata udp 9100+N)
insight_host=<host-ip>
score_threshold=0.4

# Stream 0: object detection (YOLO26n INT8)
stream0.source=rtsp://<host-ip>:8554/src1
stream0.model_path=./models/yolo26n.compile_ready_mpk.tar.gz
stream0.task=detection
stream0.width=1280
stream0.height=720
stream0.fps=30                  # must match the stream's frame rate

# Stream 1: pose estimation (YOLO26n-pose BF16)
stream1.source=rtsp://<host-ip>:8554/src2
stream1.model_path=./models/yolo26n-pose.compile_ready_mpk.tar.gz
stream1.task=pose
stream1.width=1280
stream1.height=720
stream1.fps=30
```

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

Expected (`dk` prefixes each line with `[DEVKIT][STDOUT]`; Neat's start-up lines come first):

```text
stream 0: rtsp://<host-ip>:8554/src1 -> ./models/yolo26n.compile_ready_mpk.tar.gz -> Insight channel 0
stream 1: rtsp://<host-ip>:8554/src2 -> ./models/yolo26n-pose.compile_ready_mpk.tar.gz -> Insight channel 1
[stream 0 det ] frames=100 fps=18.0 results=7
[stream 1 pose] frames=100 fps=17.8 results=1
[stream 0 det ] frames=200 fps=31.6 results=7
[stream 1 pose] frames=200 fps=30.2 results=1
...
^C
[stream 0] stopped after 744 frames
[stream 1] stopped after 729 frames
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
| `[misconfig.media_caps] … incompatible media caps` | A `streamN.fps` (or `width`/`height`) does not match that stream. Neat 0.5.0 requires an exact match |
| `pull: ...` error on one stream | Read the message. The model path or source of that stream is usually wrong. The other stream stops too, by design |
| Lower fps on both streams | Expected only when the MLA is saturated. At 2 × 30 fps with these models it is far below its limit (`model_info`: 550 + 199 fps) |

---

## Exercises and discussion

1. **Add a third stream**: copy the whole `stream1.*` block as `stream2.*` (keep `stream2.task=pose`
   and `stream2.fps`), point `stream2.source` at `src1`, and watch channel 2. No code changes are needed.
2. **Swap precisions**: compile the pose model in INT8 (`--precision int8`) and compare keypoint
   stability with BF16 (the app runs at the source rate, so compare speed with `model_info`).
3. **Trigger logic**: in `stream_worker`, print a message when a person's `left_wrist` is above the `nose`
   (a raised hand). This is the start of a real application.
4. **Discussion**: why does one graph with N streams scale better than N separate processes? Think
   about decoder planning and the shared MLA.

## Checkpoint

- [ ] Two live channels in Insight: boxes on channel 0, skeletons on channel 1
- [ ] You can explain how endpoint names keep the two pipelines apart in one graph

[← Session 3: App 1](03-app1-single-stream.md) · [Agenda](00-agenda.md)
