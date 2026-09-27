#include "text/dsv41_prompt.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "text/chat_template.hpp"

namespace dgpp::text {
namespace {

using dgpp::minijson::Member;

[[noreturn]] void refuse(const std::string& what) { throw std::runtime_error("dsv41 prompt: " + what); }

// The 0731 encoder's DSML markers: the tag token is followed directly by the
// element name (no space), and the block is "tool_calls" (the reference's
// tool_call_template / tool_calls_template over dsml_token).
const std::string kCallsOpen = std::string("<") + Dsv41Prompt::kDsml + "tool_calls>";
const std::string kCallsClose = std::string("</") + Dsv41Prompt::kDsml + "tool_calls>";
const std::string kInvokeOpen = std::string("<") + Dsv41Prompt::kDsml + "invoke";
const std::string kInvokeClose = std::string("</") + Dsv41Prompt::kDsml + "invoke>";
const std::string kParamOpen = std::string("<") + Dsv41Prompt::kDsml + "parameter";
const std::string kParamClose = std::string("</") + Dsv41Prompt::kDsml + "parameter>";

struct Param {
  std::string key;
  std::string value;  // the raw string, or the JSON text
  bool is_string = false;
};

struct ToolCall {
  std::string id;
  std::string name;  // namespace-qualified for the render
  std::vector<Param> params;
};

struct Block {
  bool tool_result = false;
  std::string text;         // a text block, or the tool result's content
  std::string tool_use_id;  // tool_result
};

struct Msg {
  std::string role;
  std::string content;
  std::vector<Block> blocks;  // user turns after the merge
  bool has_blocks = false;
  std::string reasoning;
  bool has_reasoning = false;
  std::vector<ToolCall> calls;
  const minijson::Value* tools = nullptr;  // the system message's tools
};

std::string json_of(const minijson::Value& v) { return Value::from_minijson(v).to_json(/*ensure_ascii=*/false); }

// The text of a content field: a string, or the "\n\n"-joined text parts
// of an array (the reference's process_image_messages; an image part is
// refused — the API serves text).
std::string content_text(const minijson::Value* content, const std::string& where) {
  if (content == nullptr || content->is_null()) return "";
  if (content->is_string()) return std::string(content->as_string());
  if (!content->is_array()) refuse(where + ".content must be a string or an array of content parts");
  std::string out;
  bool first = true;
  for (const minijson::Value& part : content->items()) {
    if (!part.is_object()) refuse(where + ".content parts must be objects");
    const minijson::Value* type = part.find("type");
    const std::string t = type && type->is_string() ? std::string(type->as_string()) : "";
    if (t == "image" || t == "image_url") refuse(where + ": image content is not served (text only)");
    if (t != "text") continue;
    const minijson::Value* text = part.find("text");
    if (!first) out += "\n\n";
    first = false;
    if (text && text->is_string()) out += std::string(text->as_string());
  }
  return out;
}

// A tool result's content: a string, or a list of parts (the reference's
// tool_result branch of the user's content_blocks): text parts verbatim,
// any other part as "[Unsupported <type>]", the parts "\n\n"-joined.
std::string tool_result_text(const minijson::Value* content, const std::string& where) {
  if (content == nullptr || content->is_null()) return "";
  if (content->is_string()) return std::string(content->as_string());
  if (!content->is_array()) refuse(where + ".content must be a string or an array of content parts");
  std::string out;
  bool first = true;
  for (const minijson::Value& part : content->items()) {
    if (!part.is_object()) refuse(where + ".content parts must be objects");
    const minijson::Value* type = part.find("type");
    const std::string t = type && type->is_string() ? std::string(type->as_string()) : "None";
    if (!first) out += "\n\n";
    first = false;
    if (t == "text") {
      const minijson::Value* text = part.find("text");
      if (text && text->is_string()) out += std::string(text->as_string());
    } else {
      out += "[Unsupported " + t + "]";
    }
  }
  return out;
}

// The reference's tools_from_openai_format: the function object, as is
// (the entry's top-level "namespace" is ignored by the 0731 encoder). A
// flat definition (no "function" key) is the entry itself — the service
// accepts that shape.
const minijson::Value* function_def_of(const minijson::Value& tool, const std::string& where) {
  const minijson::Value* fn = tool.find("function");
  if (fn && fn->is_object()) {
    const minijson::Value* name = fn->find("name");
    if (!name || !name->is_string()) refuse(where + ".function.name must be a string");
    return fn;
  }
  if (!tool.is_object()) refuse(where + " must be an object");
  const minijson::Value* name = tool.find("name");
  if (!name || !name->is_string()) refuse(where + ".name must be a string");
  return &tool;
}

// One tool's schema line: the function object serialized verbatim (the
// reference's tools_from_openai_format + to_json).
std::string render_tool_schema(const minijson::Value& tool) {
  return Value::from_minijson(*function_def_of(tool, "tools[]")).to_json(false);
}

std::string render_tools(const minijson::Value& tools) {
  std::string schemas;
  bool first = true;
  for (const minijson::Value& t : tools.items()) {
    if (!t.is_object()) refuse("tools[] must be objects");
    if (!first) schemas += "\n";
    first = false;
    schemas += render_tool_schema(t);
  }
  std::string out;
  out += "## Tools\n\n";
  out += "You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"" +
         kCallsOpen + "\" block like the following:\n\n";
  out += kCallsOpen + "\n";
  out += kInvokeOpen + " name=\"$TOOL_NAME\">\n";
  out += kParamOpen + " name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE" + kParamClose + "\n";
  out += "...\n";
  out += kInvokeClose + "\n";
  out += kInvokeOpen + " name=\"$TOOL_NAME2\">\n";
  out += "...\n";
  out += kInvokeClose + "\n";
  out += kCallsClose + "\n\n";
  out += "String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, "
         "arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n";
  out += "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside "
         "<think>...</think> BEFORE any tool calls or final response.\n\n";
  out += "Otherwise, output directly after </think> with tool calls or final response.\n\n";
  out += "### Available Tool Schemas\n\n";
  out += schemas;
  out += "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";
  return out;
}

// An assistant tool call off the OpenAI form: the qualified name and the
// DSML parameter lines (a string argument raw, any other value as JSON;
// arguments that are not a JSON object become one under "arguments").
ToolCall parse_tool_call(const minijson::Value& tc, const std::string& where) {
  ToolCall call;
  if (const minijson::Value* id = tc.find("id"); id && id->is_string()) call.id = std::string(id->as_string());
  const minijson::Value* fn = tc.find("function");
  if (!fn || !fn->is_object()) refuse(where + ".function must be an object");
  if (call.id.empty())
    if (const minijson::Value* id = fn->find("id"); id && id->is_string()) call.id = std::string(id->as_string());
  const minijson::Value* name = fn->find("name");
  if (!name || !name->is_string()) refuse(where + ".function.name must be a string");
  call.name = std::string(name->as_string());
  const minijson::Value* args = fn->find("arguments");
  // The arguments: an object, or a JSON string of one (a double-encoded
  // string parses twice); anything else is {"arguments": <as given>}.
  minijson::ParseResult parsed;
  const minijson::Value* obj = nullptr;
  std::string owned;
  if (args && args->is_object()) {
    obj = args;
  } else if (args && args->is_string()) {
    owned = std::string(args->as_string());
    for (int pass = 0; pass < 2 && obj == nullptr; ++pass) {
      try {
        parsed = minijson::parse(owned);
        if (parsed.root.is_object()) obj = &parsed.root;
        else if (parsed.root.is_string()) owned = std::string(parsed.root.as_string());
        else break;
      } catch (const std::exception&) {
        break;
      }
    }
  }
  if (obj != nullptr) {
    for (const Member& m : obj->members()) {
      Param p;
      p.key = m.key;
      if (m.value.is_string()) {
        p.is_string = true;
        p.value = std::string(m.value.as_string());
      } else {
        p.value = json_of(m.value);
      }
      call.params.push_back(std::move(p));
    }
  } else {
    Param p;
    p.key = "arguments";
    if (args && args->is_string()) {
      p.is_string = true;
      p.value = std::string(args->as_string());
    } else if (args) {
      p.value = json_of(*args);
    } else {
      p.is_string = true;
    }
    call.params.push_back(std::move(p));
  }
  return call;
}

std::string render_call(const ToolCall& c) {
  std::string lines;
  for (size_t i = 0; i < c.params.size(); ++i) {
    if (i) lines += "\n";
    lines += kParamOpen + " name=\"" + c.params[i].key + "\" string=\"" + (c.params[i].is_string ? "true" : "false") +
             "\">" + c.params[i].value + kParamClose;
  }
  return kInvokeOpen + " name=\"" + c.name + "\">\n" + lines + "\n" + kInvokeClose;
}

// The reference's last user index: roles user / developer only — a
// mid-conversation system message does not move it.
int find_last_user_index(const std::vector<Msg>& msgs) {
  for (int i = static_cast<int>(msgs.size()) - 1; i >= 0; --i)
    if (msgs[static_cast<size_t>(i)].role == "user") return i;
  return -1;
}

// The reference's reasoning-effort levels (encoding_dsv4.py): exactly
// low / high / max, each realized as a text prefix prepended at the very
// start of the prompt (before the system message) in thinking mode only;
// "low" is the default and adds nothing. Any other value is refused.
std::string effort_prefix(const minijson::Value* v) {
  if (v == nullptr || v->is_null()) return "";  // the reference's default, "low"
  if (!v->is_string()) refuse("reasoning_effort must be a string");
  const std::string_view s = v->as_string();
  if (s == "low") return "";
  if (s == "high")
    return "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
           "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
           "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
  if (s == "max")
    return "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
           "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
           "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";
  refuse("reasoning_effort must be low, high or max");
}

}  // namespace

bool Dsv41Prompt::reads(std::string_view name) {
  // `thinking` (2026-09-14): the key the vLLM DeepSeek-V4.1 template and
  // its clients use for the same switch — an alias of enable_thinking, so a
  // request written for that stack turns thinking off here too.
  return name == "enable_thinking" || name == "thinking" || name == "clear_thinking" ||
         name == "reasoning_effort";
}

uint64_t Dsv41Prompt::source_hash() { return 0x64737634312d3031ull; }  // "dsv41-01": the renderer's first revision

std::string Dsv41Prompt::render(const minijson::Value& globals) {
  if (!globals.is_object()) refuse("globals must be an object");
  const minijson::Value* messages = globals.find("messages");
  if (!messages || !messages->is_array() || messages->items().empty()) refuse("messages is required and must be a non-empty array");
  const minijson::Value* tools = globals.find("tools");
  if (tools && (tools->is_null() || (tools->is_array() && tools->items().empty()))) tools = nullptr;
  if (tools && !tools->is_array()) refuse("tools must be an array");
  bool thinking = true, drop_thinking = true;
  if (const minijson::Value* v = globals.find("enable_thinking"); v && v->is_bool()) thinking = v->as_bool();
  if (const minijson::Value* v = globals.find("thinking"); v && v->is_bool()) thinking = v->as_bool();
  if (const minijson::Value* v = globals.find("clear_thinking"); v && v->is_bool()) drop_thinking = v->as_bool();

  // ---- the messages in the reference's shape --------------------------------
  std::vector<Msg> raw;
  for (size_t i = 0; i < messages->items().size(); ++i) {
    const minijson::Value& m = messages->items()[i];
    const std::string where = "messages[" + std::to_string(i) + "]";
    if (!m.is_object()) refuse(where + " must be an object");
    const minijson::Value* role = m.find("role");
    if (!role || !role->is_string()) refuse(where + ".role must be a string");
    Msg msg;
    msg.role = std::string(role->as_string());
    const minijson::Value* content = m.find("content");
    if (msg.role == "system") {
      msg.content = content_text(content, where);
    } else if (msg.role == "user") {
      msg.content = content_text(content, where);
      Block b;
      b.text = msg.content;
      msg.blocks.push_back(std::move(b));
      msg.has_blocks = true;
    } else if (msg.role == "assistant") {
      msg.content = content_text(content, where);
      if (const minijson::Value* rc = m.find("reasoning_content"); rc && rc->is_string()) {
        msg.has_reasoning = true;
        msg.reasoning = std::string(rc->as_string());
      }
      if (const minijson::Value* calls = m.find("tool_calls"); calls && calls->is_array())
        for (size_t k = 0; k < calls->items().size(); ++k)
          msg.calls.push_back(parse_tool_call(calls->items()[k], where + ".tool_calls[" + std::to_string(k) + "]"));
    } else if (msg.role == "tool") {
      Block b;
      b.tool_result = true;
      b.text = tool_result_text(content, where);
      if (const minijson::Value* id = m.find("tool_call_id"); id && id->is_string()) b.tool_use_id = std::string(id->as_string());
      msg.blocks.push_back(std::move(b));
      msg.has_blocks = true;
    } else {
      refuse(where + ": unknown role '" + msg.role + "' (system, user, assistant, tool)");
    }
    raw.push_back(std::move(msg));
  }
  // The tools ride the first system message, or an empty one in front.
  if (tools) {
    if (raw[0].role == "system") {
      raw[0].tools = tools;
    } else {
      Msg sys;
      sys.role = "system";
      sys.tools = tools;
      raw.insert(raw.begin(), std::move(sys));
    }
  }

  // ---- merge_tool_messages ------------------------------------------------------
  std::vector<Msg> merged;
  for (Msg& m : raw) {
    if (m.role == "tool") {
      if (!merged.empty() && merged.back().role == "user" && merged.back().has_blocks) {
        merged.back().blocks.push_back(std::move(m.blocks[0]));
      } else {
        Msg u;
        u.role = "user";
        u.blocks.push_back(std::move(m.blocks[0]));
        u.has_blocks = true;
        merged.push_back(std::move(u));
      }
    } else if (m.role == "user") {
      if (!merged.empty() && merged.back().role == "user" && merged.back().has_blocks) {
        for (Block& b : m.blocks) merged.back().blocks.push_back(std::move(b));
      } else {
        merged.push_back(std::move(m));
      }
    } else {
      merged.push_back(std::move(m));
    }
  }
  // ---- sort_tool_results_by_call_order --------------------------------------------
  {
    std::vector<std::pair<std::string, int>> order;  // the last assistant's call ids
    for (Msg& m : merged) {
      if (m.role == "assistant" && !m.calls.empty()) {
        order.clear();
        for (size_t i = 0; i < m.calls.size(); ++i)
          if (!m.calls[i].id.empty()) order.emplace_back(m.calls[i].id, static_cast<int>(i));
      } else if (m.role == "user" && m.has_blocks) {
        std::vector<Block*> tool_blocks;
        for (Block& b : m.blocks)
          if (b.tool_result) tool_blocks.push_back(&b);
        if (tool_blocks.size() > 1 && !order.empty()) {
          std::vector<Block> sorted;
          for (Block* b : tool_blocks) sorted.push_back(*b);
          const auto rank = [&](const Block& b) {
            for (const auto& [id, idx] : order)
              if (id == b.tool_use_id) return idx;
            return 0;
          };
          std::stable_sort(sorted.begin(), sorted.end(), [&](const Block& a, const Block& b) { return rank(a) < rank(b); });
          size_t k = 0;
          for (Block& b : m.blocks)
            if (b.tool_result) b = sorted[k++];
        }
      }
    }
  }
  // ---- drop_thinking --------------------------------------------------------------
  bool any_tools = false;
  for (const Msg& m : merged)
    if (m.tools) any_tools = true;
  const bool effective_drop = drop_thinking && !any_tools;
  std::vector<Msg> msgs;
  if (thinking && effective_drop) {
    const int last_user = find_last_user_index(merged);
    for (size_t i = 0; i < merged.size(); ++i) {
      Msg& m = merged[i];
      if (m.role == "user" || m.role == "system" || m.role == "tool" || static_cast<int>(i) >= last_user) {
        msgs.push_back(std::move(m));
      } else if (m.role == "assistant") {
        m.has_reasoning = false;
        m.reasoning.clear();
        msgs.push_back(std::move(m));
      }
    }
  } else {
    msgs = std::move(merged);
  }

  // ---- render ---------------------------------------------------------------------
  const int last_user = find_last_user_index(msgs);
  const std::string effort = effort_prefix(globals.find("reasoning_effort"));
  std::string prompt = kBos;
  for (size_t index = 0; index < msgs.size(); ++index) {
    const Msg& m = msgs[index];
    std::string out;
    // The reference's effort prefix sits at the very start of the prompt,
    // before the first message's content (encoding_dsv4.py render_message:
    // `prompt += REASONING_EFFORT_PROMPTS[...]` then the role block).
    if (index == 0 && thinking) out += effort;
    if (m.role == "system") {
      out += m.content;
      if (m.tools) out += "\n\n" + render_tools(*m.tools);
    } else if (m.role == "user") {
      out += kUser;
      bool first = true;
      for (const Block& b : m.blocks) {
        if (!first) out += "\n\n";
        first = false;
        out += b.tool_result ? "<tool_result>" + b.text + "</tool_result>" : b.text;
      }
    } else if (m.role == "assistant") {
      std::string tc;
      if (!m.calls.empty()) {
        std::string calls;
        for (size_t i = 0; i < m.calls.size(); ++i) {
          if (i) calls += "\n";
          calls += render_call(m.calls[i]);
        }
        tc = "\n\n" + kCallsOpen + "\n" + calls + "\n" + kCallsClose;
      }
      std::string thinking_part;
      if (thinking && (!effective_drop || static_cast<int>(index) > last_user))
        thinking_part = m.reasoning + kThinkClose;
      out += thinking_part + m.content + tc + kEos;
    } else {
      refuse("unknown role after preprocessing: " + m.role);
    }
    // The transition: the reference appends the generation header only after
    // a user message — when the next message is an assistant (that block opens
    // without a role marker) or this is the last message. A mid-conversation
    // system message gets none: the following block carries its own opener.
    const bool followed_by_other =
        index + 1 < msgs.size() && msgs[index + 1].role != "assistant";
    if (!followed_by_other && m.role == "user") {
      out += kAssistant;
      if (!effective_drop && thinking) out += kThinkOpen;
      else if (effective_drop && thinking && static_cast<int>(index) >= last_user) out += kThinkOpen;
      else out += kThinkClose;
    }
    prompt += out;
  }
  return prompt;
}

std::string Dsv41Prompt::qualified_tool_name(const minijson::Value& tool) {
  const minijson::Value* fn = function_def_of(tool, "tool");
  return std::string(fn->find("name")->as_string());
}

}  // namespace dgpp::text
