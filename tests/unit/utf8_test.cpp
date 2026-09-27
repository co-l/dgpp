#include "common/test.hpp"
#include "serve/utf8.hpp"
#include "../common/utf8.hpp"

#include <stdexcept>
#include <string>
#include <utility>

DGPP_TEST(utf8_every_chunk_partition_preserves_valid_text_and_replaces_invalid_bytes) {
  const std::pair<std::string, std::string> cases[] = {
      {"", ""}, {std::string("\0x", 2), std::string("\0x", 2)},
      {"\xC2\x80", "\xC2\x80"}, {"\xDF\xBF", "\xDF\xBF"},
      {"\xE0\xA0\x80", "\xE0\xA0\x80"}, {"\xED\x9F\xBF", "\xED\x9F\xBF"},
      {"\xEE\x80\x80", "\xEE\x80\x80"}, {"\xEF\xBF\xBF", "\xEF\xBF\xBF"},
      {"\xF0\x90\x80\x80", "\xF0\x90\x80\x80"},
      {"\xF4\x8F\xBF\xBF", "\xF4\x8F\xBF\xBF"}, {"é📭", "é📭"},
      {"\xED\xA0\x80", "���"}, {"\xED\xBF\xBF", "���"},
      {"\xC0\x80", "��"}, {"\xC1\xBF", "��"},
      {"\xE0\x80\x80", "���"}, {"\xF0\x80\x80\x80", "����"},
      {"\xF4\x90\x80\x80", "����"}, {"\xF5\x80\x80\x80", "����"},
      {"\xF7\xBF\xBF\xBF", "����"}, {"\x80\xBF\xFE\xFF", "����"},
      {"\xC2", "�"}, {"\xE2\x82", "�"}, {"\xF0\x9F\x93", "�"},
      {"\xE2\x82" "A", "�A"}, {"\xE2" "é", "�é"},
      {"\xED\xA0" "📭", "��📭"},
  };
  for (const auto& [bytes, expected] : cases) {
    for (const bool suffix : {false, true}) {
      const std::string input = "a" + bytes + (suffix ? "z" : "");
      const std::string want = "a" + expected + (suffix ? "z" : "");
      dgpp::test::require_utf8(want);
      std::string whole = input;
      dgpp::serve::sanitize_utf8(&whole);
      if (whole != want) throw std::runtime_error("one-shot UTF-8 replacement mismatch");
      // Enumerate every partition, including one-byte deltas and mixed sizes.
      for (size_t cuts = 0; cuts < (size_t{1} << (input.size() - 1)); ++cuts) {
        std::string carry, joined;
        size_t start = 0;
        for (size_t end = 1; end <= input.size(); ++end) {
          if (end != input.size() && !(cuts & (size_t{1} << (end - 1)))) continue;
          std::string delta = input.substr(start, end - start);
          dgpp::serve::carry_utf8(&delta, &carry);
          dgpp::test::require_utf8(delta);
          joined += delta;
          // Empty deltas must preserve a held prefix.
          delta.clear();
          dgpp::serve::carry_utf8(&delta, &carry);
          if (!delta.empty()) throw std::runtime_error("empty delta emitted a held prefix");
          start = end;
        }
        joined += dgpp::serve::finish_utf8(&carry);
        if (joined != want || !carry.empty() || !dgpp::serve::finish_utf8(&carry).empty())
          throw std::runtime_error("chunked UTF-8 replacement mismatch");
      }
    }
  }
}
