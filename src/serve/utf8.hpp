#pragma once

#include <string>

namespace dgpp::serve {

// A field's token bytes may split a UTF-8 character across deltas. Keep only
// valid incomplete prefixes in carry; every emitted delta is valid UTF-8.
// Invalid subsequences become U+FFFD without consuming a following character.
inline void carry_utf8(std::string* text, std::string* carry) {
  if (!carry->empty()) {
    text->insert(0, *carry);
    carry->clear();
  }
  std::string out;
  out.reserve(text->size());
  for (size_t i = 0; i < text->size();) {
    const auto lead = static_cast<unsigned char>((*text)[i]);
    if (lead < 0x80) {
      out.push_back((*text)[i++]);
      continue;
    }
    const size_t len = lead >= 0xC2 && lead <= 0xDF ? 2 :
                       lead >= 0xE0 && lead <= 0xEF ? 3 :
                       lead >= 0xF0 && lead <= 0xF4 ? 4 : 0;
    if (len == 0) {
      out.append("\xEF\xBF\xBD");
      ++i;
      continue;
    }
    size_t have = 1;
    while (have < len && i + have < text->size()) {
      const auto next = static_cast<unsigned char>((*text)[i + have]);
      if (next < 0x80 || next > 0xBF) break;
      // RFC 3629: exclude overlong forms, surrogates and values > U+10FFFF.
      if (have == 1 && ((lead == 0xE0 && next < 0xA0) ||
                        (lead == 0xED && next >= 0xA0) ||
                        (lead == 0xF0 && next < 0x90) ||
                        (lead == 0xF4 && next >= 0x90))) break;
      ++have;
    }
    if (have == len) {
      out.append(*text, i, len);
    } else if (i + have == text->size()) {
      carry->assign(*text, i, have);
      break;
    } else {
      out.append("\xEF\xBF\xBD");
    }
    i += have;
  }
  text->swap(out);
}

inline std::string finish_utf8(std::string* carry) {
  if (carry->empty()) return {};
  carry->clear();
  return "\xEF\xBF\xBD";  // an incomplete character at the end of the field
}

inline void sanitize_utf8(std::string* text) {
  std::string carry;
  carry_utf8(text, &carry);
  text->append(finish_utf8(&carry));
}

}  // namespace dgpp::serve
