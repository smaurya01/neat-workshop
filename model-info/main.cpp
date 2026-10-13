// model_info: load a compiled model archive, print its inputs/outputs, and benchmark it.
//
// Usage:  ./model_info <model_mpk.tar.gz>
#include <neat.h>

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
    std::cout << "latency_ms=" << report.latency_ms << "  fps=" << report.fps << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 1;
  }
}
