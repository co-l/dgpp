#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace dgpp::test {

// Decode scalar values independently of the serving helper's byte-range checks.
inline void require_utf8(std::string_view text) {
  for (size_t i = 0; i < text.size();) {
    const auto lead = static_cast<unsigned char>(text[i++]);
    if (lead < 0x80) continue;
    size_t trailing = 0;
    uint32_t cp = 0, minimum = 0;
    if ((lead & 0xE0) == 0xC0) {
      trailing = 1; cp = lead & 0x1F; minimum = 0x80;
    } else if ((lead & 0xF0) == 0xE0) {
      trailing = 2; cp = lead & 0x0F; minimum = 0x800;
    } else if ((lead & 0xF8) == 0xF0) {
      trailing = 3; cp = lead & 0x07; minimum = 0x10000;
    } else {
      throw std::runtime_error("invalid UTF-8 lead byte");
    }
    if (trailing > text.size() - i) throw std::runtime_error("incomplete UTF-8");
    while (trailing--) {
      const auto next = static_cast<unsigned char>(text[i++]);
      if ((next & 0xC0) != 0x80) throw std::runtime_error("invalid UTF-8 continuation");
      cp = (cp << 6) | (next & 0x3F);
    }
    if (cp < minimum || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
      throw std::runtime_error("invalid UTF-8 scalar value");
  }
}

}  // namespace dgpp::test
