#pragma once
// Shared by hs_lex_bench and bench_lex_*: latency statistics and the .meta.json sidecar that
// every timing result carries (docs/ARCHITECTURE.md "Hardware labels").

#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace hs::lexical::bench {

inline std::string sh(const std::string& cmd) {
  std::string out;
  if (std::FILE* p = popen(cmd.c_str(), "r")) {
    char buf[512];
    while (std::fgets(buf, sizeof buf, p)) out += buf;
    pclose(p);
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return out;
}

inline std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\', o += c;
    else if (c == '\n') o += "\\n";
    else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
    else o += c;
  }
  return o;
}

// Exact percentile of a sorted sample (nearest-rank).
inline double percentile(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) return 0;
  size_t rank = size_t(std::ceil(p / 100.0 * double(sorted.size())));
  rank = std::clamp<size_t>(rank, 1, sorted.size());
  return sorted[rank - 1];
}

// The compile command CMake recorded for `source` (e.g. "tools/hs_lex_bench.cc"), found in
// compile_commands.json next to the executable.
inline std::string compile_command(const char* argv0, const std::string& source) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path exe = fs::canonical(fs::path(argv0), ec);
  if (ec) return "unknown";
  for (fs::path dir = exe.parent_path(); !dir.empty() && dir != dir.root_path(); dir = dir.parent_path()) {
    fs::path cc = dir / "compile_commands.json";
    if (!fs::exists(cc)) continue;
    std::ifstream in(cc);
    std::string all((std::istreambuf_iterator<char>(in)), {});
    size_t f = all.find(source + "\"");
    if (f == std::string::npos) return "unknown";
    size_t obj = all.rfind('{', f);
    size_t c = all.find("\"command\": \"", obj);
    if (c == std::string::npos || c > f + 4096) return "unknown";
    c += 12;
    size_t e = all.find("\",", c);
    return all.substr(c, e - c);
  }
  return "unknown";
}

inline std::string git_sha(const std::string& repo) {
  std::string s = sh("git -C '" + repo + "' rev-parse HEAD 2>/dev/null");
  std::string dirty = sh("git -C '" + repo + "' status --porcelain 2>/dev/null | head -1");
  return s.empty() ? "unknown" : s + (dirty.empty() ? "" : "+dirty");
}

// Writes <result_path minus .json>.meta.json.
inline void write_meta(const std::string& result_path, int argc, char** argv, const std::string& source,
                       const std::string& cache_state, const std::string& extra_json = "") {
  std::string meta = result_path;
  if (meta.size() > 5 && meta.substr(meta.size() - 5) == ".json") meta.resize(meta.size() - 5);
  meta += ".meta.json";
  std::string cmd;
  for (int i = 0; i < argc; ++i) cmd += (i ? " " : "") + std::string(argv[i]);
  struct utsname u {};
  uname(&u);
  std::string os = std::string(u.sysname) + " " + u.release + " " + u.machine;
  std::string cpu = sh("sysctl -n machdep.cpu.brand_string 2>/dev/null");
  if (cpu.empty()) cpu = sh("grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2");
  bool mac = std::string(u.sysname) == "Darwin";
  std::string cores_p = mac ? sh("sysctl -n hw.perflevel0.physicalcpu 2>/dev/null") : "";
  std::string cores_e = mac ? sh("sysctl -n hw.perflevel1.physicalcpu 2>/dev/null") : "";
  std::string mem = mac ? sh("sysctl -n hw.memsize") : sh("grep MemTotal /proc/meminfo | awk '{print $2*1024}'");
  std::string load = sh("uptime | sed 's/.*load average[s]*: //'");
  char when[64];
  std::time_t now = std::time(nullptr);
  std::strftime(when, sizeof when, "%Y-%m-%dT%H:%M:%S%z", std::localtime(&now));
  std::ofstream o(meta);
  o << "{\n"
    << "  \"hardware_label\": \"" << (mac ? "dev-signal-only" : "unlabelled-linux") << "\",\n"
    << "  \"cpu\": \"" << json_escape(cpu) << "\",\n"
    << "  \"logical_cores\": " << std::thread::hardware_concurrency() << ",\n"
    << "  \"performance_cores\": \"" << cores_p << "\",\n"
    << "  \"efficiency_cores\": \"" << cores_e << "\",\n"
    << "  \"memory_bytes\": \"" << mem << "\",\n"
    << "  \"os\": \"" << json_escape(os) << "\",\n"
    << "  \"pinned\": false,\n"
    << "  \"pinning_note\": \"macOS cannot pin threads; laptop timings are a development signal only\",\n"
    << "  \"cache_state\": \"" << cache_state << "\",\n"
    << "  \"load_average_at_end\": \"" << json_escape(load) << "\",\n"
#if defined(__VERSION__)
    << "  \"compiler\": \"" << json_escape(__VERSION__) << "\",\n"
#endif
    << "  \"compile_command\": \"" << json_escape(compile_command(argv[0], source)) << "\",\n"
    << "  \"git_sha\": \"" << json_escape(git_sha(".")) << "\",\n"
    << "  \"command\": \"" << json_escape(cmd) << "\",\n"
    << "  \"cwd\": \"" << json_escape(std::filesystem::current_path().string()) << "\",\n"
    << "  \"written_at\": \"" << when << "\"" << (extra_json.empty() ? "" : ",\n  " + extra_json) << "\n}\n";
}

}  // namespace hs::lexical::bench
