#pragma once
// The DeepSeek-V4 prompt renderer (docs/deepseek_v41_flash_plan.md G6):
// the checkpoint ships no chat_template.jinja — `encoding/encoding_dsv4.py`
// (its `encode_messages`, text-only) is the reference, and this is that
// function over the service's template globals. The prompt format
// (0731 encoder: no system marker, `system_msg_template` is bare content):
//   <｜begin▁of▁sentence｜>[effort prefix]{system}
//   <｜User｜>{user}<｜Assistant｜><think>{reasoning}</think>{content}[\n\n<｜DSML｜tool_calls>...]<｜end▁of▁sentence｜>
//   ... the generation header after a user message only (never after a
//   mid-conversation system): <｜Assistant｜> then <think> (thinking mode)
//   or </think> (chat mode).
// Tool results are <tool_result>...</tool_result> blocks inside the user
// turn (a `tool` message merges into the preceding user turn or opens
// one; consecutive user messages merge likewise), ordered by the previous
// assistant turn's call ids; tools are the "## Tools" block appended to
// the system message (the service's `tools` global attaches to the first
// system message, or to an empty one inserted at the front); earlier
// turns' reasoning is dropped in thinking mode unless the conversation
// carries tools (`clear_thinking`, default true); `enable_thinking` false
// is the reference's chat mode; `reasoning_effort` renders the reference's
// text prefix at the conversation's start in thinking mode (exactly
// low / high / max; the service's scale folds onto them — minimal/low,
// medium/high, xhigh/max; low and the absent default add nothing).
//
// Validation: tests/host/dsv41_prompt_test.cpp renders
// tests/data/dsv41_prompt_goldens.jsonl (tools/gen_dsv41_prompt_goldens.py:
// the snapshot's own encoder over the same globals) byte-exact, and the
// tokenizer ids of every render.
#include <cstdint>
#include <string>
#include <string_view>

#include "loaders/minijson.hpp"

namespace dgpp::text {

class Dsv41Prompt {
 public:
  // The service's globals: "messages" (the normalized OpenAI array; roles
  // system / user / assistant / tool), optional "tools" (the OpenAI
  // array), "reasoning_effort" (a string or an integer), "clear_thinking"
  // and "enable_thinking" (bools). Throws std::runtime_error naming the
  // offending field on anything the reference refuses (an unknown role,
  // image content, a namespace conflict).
  static std::string render(const minijson::Value& globals);
  // One OpenAI tool entry's name as the schema lists it and the model
  // writes it in a DSML invoke: the function's name, verbatim (the 0731
  // reference ignores the entry's namespace). Throws std::invalid_argument
  // on a malformed entry (the renderer's own refusals).
  static std::string qualified_tool_name(const minijson::Value& tool);
  // The knobs this renderer reads (the service's chat_template_kwargs gate).
  static bool reads(std::string_view name);
  // The renderer's revision (the prefix cache's key stands on it, as on a
  // Jinja template's source hash): bumped whenever the render changes.
  static uint64_t source_hash();

  static constexpr const char* kBos = "<｜begin▁of▁sentence｜>";
  static constexpr const char* kEos = "<｜end▁of▁sentence｜>";
  static constexpr const char* kUser = "<｜User｜>";
  static constexpr const char* kAssistant = "<｜Assistant｜>";
  static constexpr const char* kSystem = "<｜System｜>";
  static constexpr const char* kThinkOpen = "<think>";
  static constexpr const char* kThinkClose = "</think>";
  static constexpr const char* kDsml = "｜DSML｜";
};

}  // namespace dgpp::text
