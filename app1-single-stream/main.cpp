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
  // Postprocess: decode the raw YOLO26 heads + NMS on the device.
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
    js << (n++ ? "," : "") << "{\"id\":\"obj_" << n << "\",\"label\":\"" << label
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
      // frame_id ties these boxes to the matching video frame in Insight.
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
