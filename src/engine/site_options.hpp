#pragma once
// Local RoCE device selection shared by serving, probes and loopback tests.
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dgpp::net {

inline std::vector<std::string> words(const std::string& value) {
  std::istringstream input(value);
  std::vector<std::string> result;
  for (std::string word; input >> word;) result.push_back(word);
  return result;
}

inline std::vector<std::string> configured_lane_devices() {
  if (const char* value = std::getenv("DGPP_ROCE_DEVICES"); value && *value)
    return words(value);
  // Sorted active Ethernet/RoCE devices. Explicit configuration is preferable
  // on hosts with multiple fabrics or unused-but-active ports.
  std::vector<std::string> devices;
  std::error_code error;
  for (const auto& device : std::filesystem::directory_iterator("/sys/class/infiniband", error)) {
    for (const auto& port : std::filesystem::directory_iterator(device.path() / "ports", error)) {
      if (port.path().filename() != "1") continue;
      std::string state, layer;
      std::ifstream(port.path() / "state") >> state;
      std::ifstream(port.path() / "link_layer") >> layer;
      if (state == "4:" && layer == "Ethernet") {
        devices.push_back(device.path().filename().string());
        break;
      }
    }
  }
  std::sort(devices.begin(), devices.end());
  return devices;
}

inline int configured_gid_index(const std::string& device) {
  const char* value = std::getenv("DGPP_ROCE_GID_INDICES");
  if (!value || !*value) return -1;
  const auto indices = words(value);
  const auto devices = configured_lane_devices();
  if (indices.size() != devices.size())
    throw std::invalid_argument("DGPP_ROCE_GID_INDICES must match DGPP_ROCE_DEVICES");
  for (size_t i = 0; i < devices.size(); ++i) {
    if (devices[i] != device) continue;
    size_t used = 0;
    const int index = std::stoi(indices[i], &used);
    if (used != indices[i].size() || index < 0 || index > 255)
      throw std::invalid_argument("RoCE GID index must be in [0, 255]");
    return index;
  }
  return -1;
}

}  // namespace dgpp::net
