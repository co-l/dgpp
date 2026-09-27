#pragma once

#include <cstdint>

namespace dgpp::serve {

// A byte limit on the serialized request, independent of the model's token
// budget. Leave room for large document prefills, JSON escaping and tool
// history. This is a ceiling, not a buffer allocated at startup.
inline constexpr int64_t kDefaultHttpMaxBodyBytes = 256ll * 1024 * 1024;

inline constexpr int kDefaultSsePingInterval = 30;
inline constexpr int64_t kMaxSsePingInterval = 2147483647;
inline constexpr bool valid_sse_ping_interval(int64_t seconds) {
  return seconds == -1 || (seconds >= 1 && seconds <= kMaxSsePingInterval);
}

}  // namespace dgpp::serve
