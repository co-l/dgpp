#include "models/dsv41/config.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("DeepSeek-V4.1 config.{}: {}", field, why));
}

const minijson::Value& require(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) reject(field, "missing");
  return *f;
}
int require_int(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  return static_cast<int>(f.as_int());
}
int64_t require_int64(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  return f.as_int();
}
int optional_int(const minijson::Value& v, std::string_view field, int dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  return static_cast<int>(f->as_int());
}
int64_t optional_int64(const minijson::Value& v, std::string_view field, int64_t dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  return f->as_int();
}
double require_double(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  const double d = f.as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}
double optional_double(const minijson::Value& v, std::string_view field, double dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  const double d = f->as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}
bool optional_bool(const minijson::Value& v, std::string_view field, bool dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_bool()) reject(field, "not a bool");
  return f->as_bool();
}
std::string optional_string(const minijson::Value& v, std::string_view field, const std::string& dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_string()) reject(field, "not a string");
  return std::string(f->as_string());
}
std::vector<int> require_int_list(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_array()) reject(field, "not an array");
  std::vector<int> out;
  for (const auto& item : f.items()) {
    if (!item.is_number()) reject(field, "non-numeric element");
    out.push_back(static_cast<int>(item.as_int()));
  }
  return out;
}
std::vector<int64_t> require_int64_list(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_array()) reject(field, "not an array");
  std::vector<int64_t> out;
  for (const auto& item : f.items()) {
    if (!item.is_number()) reject(field, "non-numeric element");
    out.push_back(item.as_int());
  }
  return out;
}
// A field that must be absent or null (a feature the engine does not
// implement and must not silently ignore).
void require_absent(const minijson::Value& v, std::string_view field, std::string_view why) {
  const minijson::Value* f = v.find(field);
  if (f && !f->is_null()) reject(field, why);
}

std::string read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    throw std::runtime_error(std::format("cannot open config {}: {}", path, std::strerror(errno)));
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  return text;
}

bool contains(const std::vector<int>& v, int x) {
  return std::find(v.begin(), v.end(), x) != v.end();
}
bool strictly_increasing(const std::vector<int>& v) {
  for (size_t i = 1; i < v.size(); ++i)
    if (v[i] <= v[i - 1]) return false;
  return true;
}

// --- the quantization contract --------------------------------------------

void parse_quantization(const minijson::Value& root, Dsv41TextConfig& c) {
  const minijson::Value* qc = root.find("quantization_config");
  if (!qc || qc->is_null())
    reject("quantization_config",
           "missing — the engine serves the release's own mixed fp8 / fp4 format "
           "(docs/deepseek_v41_flash_plan.md §0.1); a BF16 checkpoint has no expert path here");
  if (!qc->is_object()) reject("quantization_config", "not an object");
  if (const std::string m = optional_string(*qc, "quant_method", ""); m != "fp8")
    reject("quantization_config.quant_method", "only the release's fp8 method is implemented, got '" + m + "'");
  if (const std::string a = optional_string(*qc, "activation_scheme", "dynamic"); a != "dynamic")
    reject("quantization_config.activation_scheme", "only dynamic is implemented, got '" + a + "'");
  if (const std::string s = optional_string(*qc, "scale_fmt", ""); s != "ue8m0")
    reject("quantization_config.scale_fmt", "the loader reads e8m0 (power-of-two) scales, got '" + s + "'");
  // The V4.1 release declares expert_dtype inside quantization_config; the
  // V4-Flash-0731 config carries it at the root.
  const std::string e = optional_string(*qc, "expert_dtype", optional_string(root, "expert_dtype", ""));
  if (e != "fp4")
    reject("quantization_config.expert_dtype",
           "the routed experts must be the release's MXFP4 (e2m1 + e8m0 per 32), got '" + e + "'");
  const minijson::Value& wb = require(*qc, "weight_block_size");
  if (!wb.is_array() || wb.items().size() != 2 || !wb.items()[0].is_number() || !wb.items()[1].is_number())
    reject("quantization_config.weight_block_size", "must be [rows, cols]");
  const int br = static_cast<int>(wb.items()[0].as_int()), bc = static_cast<int>(wb.items()[1].as_int());
  if (c.variant == Dsv41Variant::V41 && (br != 32 || bc != 32))
    reject("quantization_config.weight_block_size",
           std::format("the V4.1 fp8 kernels take the release's 32x32 grid, got [{}, {}]", br, bc));
  if (c.variant == Dsv41Variant::V4 && (br != 128 || bc != 128))
    reject("quantization_config.weight_block_size",
           std::format("the V4-Flash-0731 dense projections use the release's 128x128 grid, got [{}, {}]", br, bc));
  c.fp8_block_size = br;
  c.fp4_block_size = 32;  // the release's MXFP4 block (convert.py fp4_block_size)
  // A re-packed checkpoint (the community NVFP4 casts) declares its
  // encoding under other keys; none of them is the format this loader reads.
  require_absent(*qc, "expert_block_size", "an NVFP4 expert re-pack is not the release format the loader reads");
  require_absent(*qc, "expert_scale_fmt", "an NVFP4 expert re-pack is not the release format the loader reads");
  require_absent(*qc, "engram_dtype", "a re-packed Engram table is not the release format the loader reads");
  require_absent(*qc, "moe_quant_algo", "a modelopt NVFP4 cast is not the release format the loader reads");
  require_absent(*qc, "config_groups", "a compressed-tensors layout is not the release format the loader reads");
}

void parse_vision(const minijson::Value& root, Dsv41TextConfig& c) {
  const minijson::Value* vc = root.find("vision_config");
  if (!vc || vc->is_null()) {
    c.vision_present = false;
    return;
  }
  if (!vc->is_object()) reject("vision_config", "not an object");
  c.vision_present = true;
  c.vision.num_hidden_layers = require_int(*vc, "num_hidden_layers");
  c.vision.hidden_size = require_int(*vc, "hidden_size");
  c.vision.num_attention_heads = require_int(*vc, "num_attention_heads");
  c.vision.intermediate_size = require_int(*vc, "intermediate_size");
  c.vision.patch_size = require_int(*vc, "patch_size");
  c.vision.downsample_ratio = require_int(*vc, "downsample_ratio");
  if (c.vision.num_hidden_layers <= 0 || c.vision.hidden_size <= 0 || c.vision.intermediate_size <= 0 ||
      c.vision.patch_size <= 0 || c.vision.downsample_ratio <= 0)
    reject("vision_config", "shape fields must be positive");
}

}  // namespace

Dsv41TextConfig Dsv41TextConfig::parse(const minijson::Value& root) {
  if (!root.is_object()) reject("", "root is not an object");
  Dsv41TextConfig c;
  const std::string model_type = optional_string(root, "model_type", "deepseek_v41");
  const minijson::Value* tcp = root.find("text_config");
  const minijson::Value* tptr = nullptr;
  if (model_type == "deepseek_v4") {
    // The V4-Flash-0731 release is flat: the text model fields live at the
    // root, no nested text_config, no vision tower (quantization still
    // root-level, as in V4.1).
    c.variant = Dsv41Variant::V4;
    tptr = &root;
    if (tcp && tcp->is_object())
      reject("model_type", "the DeepSeek-V4 release is flat — a nested text_config is not expected with model_type deepseek_v4");
  } else {
    if (model_type != "deepseek_v41") reject("model_type", "expected deepseek_v41 or deepseek_v4, got " + model_type);
    if (!tcp || !tcp->is_object())
      reject("text_config", "missing — the DeepSeek-V4.1 layout nests the text model under text_config");
    if (const std::string tt = optional_string(*tcp, "model_type", "deepseek_v41_text"); tt != "deepseek_v41_text")
      reject("text_config.model_type", "expected deepseek_v41_text, got " + tt);
    tptr = tcp;
  }
  const minijson::Value& t = *tptr;

  // --- tokens (root) ----------------------------------------------------------
  c.bos_token_id = optional_int64(root, "bos_token_id", 0);
  c.eos_token_id = require_int64(root, "eos_token_id");
  c.pad_token_id = optional_int64(root, "pad_token_id", -1);
  c.image_token_id = optional_int64(root, "image_token_id", -1);

  // --- shape --------------------------------------------------------------------
  c.hidden_size = require_int(t, "hidden_size");
  c.vocab_size = require_int(t, "vocab_size");
  c.num_hidden_layers = require_int(t, "num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(require_double(t, "rms_norm_eps"));
  c.tie_word_embeddings = optional_bool(t, "tie_word_embeddings", false);
  c.hidden_act = optional_string(t, "hidden_act", "silu");
  c.max_position_embeddings = require_int(t, "max_position_embeddings");
  if (c.hidden_size <= 0 || c.hidden_size % 64 != 0)
    reject("text_config.hidden_size", "must be a positive multiple of 64");
  if (c.vocab_size <= 0) reject("text_config.vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("text_config.num_hidden_layers", "must be positive");
  if (!(c.rms_norm_eps >= 0.f)) reject("text_config.rms_norm_eps", "must be non-negative");
  if (c.hidden_act != "silu") reject("text_config.hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (c.tie_word_embeddings) reject("text_config.tie_word_embeddings", "tied embeddings are not implemented");
  if (c.max_position_embeddings <= 0) reject("text_config.max_position_embeddings", "must be positive");
  for (const int64_t id : {c.bos_token_id, c.eos_token_id})
    if (id < 0 || id >= c.vocab_size) reject("eos_token_id", "id outside [0, vocab_size)");
  if (c.pad_token_id >= c.vocab_size) reject("pad_token_id", "id outside [0, vocab_size)");

  // --- attention ------------------------------------------------------------
  c.num_attention_heads = require_int(t, "num_attention_heads");
  c.num_key_value_heads = optional_int(t, "num_key_value_heads", 1);
  c.head_dim = require_int(t, "head_dim");
  c.qk_rope_head_dim = require_int(t, "qk_rope_head_dim");
  c.q_lora_rank = require_int(t, "q_lora_rank");
  c.o_lora_rank = require_int(t, "o_lora_rank");
  c.o_groups = require_int(t, "o_groups");
  c.attention_bias = optional_bool(t, "attention_bias", false);
  c.sliding_window = require_int(t, "sliding_window");
  if (c.num_attention_heads <= 0) reject("text_config.num_attention_heads", "must be positive");
  if (c.num_key_value_heads != 1)
    reject("text_config.num_key_value_heads",
           "CSA2 keeps one latent per token that is both key and value: num_key_value_heads must be 1");
  if (c.head_dim != 512)
    reject("text_config.head_dim", "the attention kernels are compiled for the release's 512-wide latent");
  if (c.qk_rope_head_dim != 64)
    reject("text_config.qk_rope_head_dim", "the rope kernels rotate the release's 64-wide tail");
  if (c.q_lora_rank <= 0 || c.q_lora_rank % 32 != 0)
    reject("text_config.q_lora_rank", "must be a positive multiple of 32 (an fp8 block)");
  if (c.o_lora_rank <= 0 || c.o_lora_rank % 32 != 0)
    reject("text_config.o_lora_rank", "must be a positive multiple of 32 (an fp8 block)");
  if (c.o_groups <= 0 || c.num_attention_heads % c.o_groups != 0)
    reject("text_config.o_groups", "must divide num_attention_heads");
  if (c.attention_bias) reject("text_config.attention_bias", "biased attention projections are not implemented");
  if (c.sliding_window <= 0) reject("text_config.sliding_window", "must be positive");
  c.rope_theta = optional_double(t, "rope_theta", 10000.0);
  c.compress_rope_theta = optional_double(t, "compress_rope_theta", c.rope_theta);
  if (!(c.rope_theta > 0) || !(c.compress_rope_theta > 0)) reject("text_config.rope_theta", "must be positive");
  {
    const minijson::Value& rs = require(t, "rope_scaling");
    if (!rs.is_object()) reject("text_config.rope_scaling", "not an object");
    // The V4.1 release writes `rope_type`; the V4-Flash-0731 config writes
    // `type` for the same YaRN scaling.
    const std::string rt = optional_string(rs, "rope_type", optional_string(rs, "type", "default"));
    if (rt != "yarn")
      reject("text_config.rope_scaling.rope_type", "the compressed layers' rope is YaRN, got '" + rt + "'");
    c.rope_factor = require_double(rs, "factor");
    c.original_max_position_embeddings = require_int(rs, "original_max_position_embeddings");
    c.beta_fast = optional_double(rs, "beta_fast", 32.0);
    c.beta_slow = optional_double(rs, "beta_slow", 1.0);
    if (!(c.rope_factor >= 1.0)) reject("text_config.rope_scaling.factor", "must be >= 1");
    if (c.original_max_position_embeddings <= 0)
      reject("text_config.rope_scaling.original_max_position_embeddings", "must be positive");
    if (!(c.beta_fast > c.beta_slow) || !(c.beta_slow > 0)) reject("text_config.rope_scaling.beta_fast", "must exceed beta_slow > 0");
    require_absent(rs, "mscale", "a YaRN attention mscale is not part of the release (softmax scale is head_dim^-0.5)");
    require_absent(rs, "mscale_all_dim", "a YaRN attention mscale is not part of the release");
  }

  // --- DSpark (plan §1.7): the draft depth is the checkpoint's mtp block
  // count ----------------------------------------------------------------
  // The V4.1 release declares the MTP count in `num_nextn_predict_layers`;
  // the V4-Flash-0731 config carries `num_nextn_predict_layers: 1` while
  // shipping three mtp.S stages. The authoritative depth is the length of
  // `dspark_target_layer_ids` (one target per draft stage) — the file's
  // count is not the source of truth (criterion: derive from the mtp
  // blocks, not the file field).
  c.dspark_block_size = optional_int(t, "dspark_block_size", 0);
  c.dspark_noise_token_id = optional_int64(t, "dspark_noise_token_id", -1);
  c.dspark_target_layer_ids = t.find("dspark_target_layer_ids") ? require_int_list(t, "dspark_target_layer_ids") : std::vector<int>{};
  c.dspark_markov_rank = optional_int(t, "dspark_markov_rank", 0);
  // NOTE: dspark_n_routed_experts / dspark_num_experts_per_tok default to
  // the main MoE's counts and are read after the MoE section below (the
  // draft stages of V4-Flash-0731 carry the full expert set).
  if (!c.dspark_target_layer_ids.empty()) {
    c.num_nextn_predict_layers = static_cast<int>(c.dspark_target_layer_ids.size());
  } else {
    c.num_nextn_predict_layers = optional_int(t, "num_nextn_predict_layers", 0);
  }
  if (c.num_nextn_predict_layers < 0) reject("text_config.num_nextn_predict_layers", "must be >= 0");
  if (c.num_nextn_predict_layers > 0) {
    if (c.dspark_block_size <= 0) reject("text_config.dspark_block_size", "must be positive with draft stages");
    if (c.dspark_noise_token_id < 0 || c.dspark_noise_token_id >= c.vocab_size)
      reject("text_config.dspark_noise_token_id", "id outside [0, vocab_size)");
    if (c.dspark_target_layer_ids.empty()) reject("text_config.dspark_target_layer_ids", "DSpark needs target layers");
    if (!strictly_increasing(c.dspark_target_layer_ids)) reject("text_config.dspark_target_layer_ids", "must be strictly increasing");
    for (const int l : c.dspark_target_layer_ids)
      if (l < 0 || l >= c.num_hidden_layers) reject("text_config.dspark_target_layer_ids", std::format("layer {} outside the backbone", l));
    if (c.dspark_markov_rank <= 0 || c.dspark_markov_rank % 32 != 0)
      reject("text_config.dspark_markov_rank", "must be a positive multiple of 32");
  }

  // --- CSA2 schedule ------------------------------------------------------------
  c.compress_ratios = require_int_list(t, "compress_ratios");
  if (static_cast<int>(c.compress_ratios.size()) != c.max_layer())
    reject("text_config.compress_ratios",
           std::format("must list every layer and draft stage ({} entries), got {}", c.max_layer(),
                       c.compress_ratios.size()));
  for (int l = 0; l < c.max_layer(); ++l) {
    const int r = c.compress_ratios[static_cast<size_t>(l)];
    const bool ok = (c.variant == Dsv41Variant::V4) ? (r == 0 || r == 4 || r == 128)
                                                    : (r == 0 || r == 1 || r == 2);
    if (!ok)
      reject("text_config.compress_ratios",
             std::format("layer {}: the compressor implements ratios {} for this variant, got {}", l,
                         c.variant == Dsv41Variant::V4 ? "0, 4 and 128" : "0, 1 and 2", r));
    if (c.is_draft(l) && r != 0)
      reject("text_config.compress_ratios", std::format("draft stage {} must be window-only (ratio 0)", c.draft_stage(l)));
  }
  if (c.variant == Dsv41Variant::V4) {
    // The V4-Flash-0731 release has no kv_source / index_source lists: the
    // CSA2 caches are per-layer (each compressed layer owns its own cache
    // and its own index keys when it has an indexer). The C4A layers
    // (ratio 4) carry the indexer and score their own cache; the C128A
    // layers (ratio 128) have no indexer and select their compressed
    // blocks sequentially (reference Attention.forward / get_compress_topk_idxs).
    for (int l = 0; l < c.num_hidden_layers; ++l) {
      if (c.compress_ratio(l) > 0) c.kv_source_layer_ids.push_back(l);
      if (c.compress_ratio(l) == 4) c.index_source_layer_ids.push_back(l);
    }
    c.candidate_source_layer_id = -1;
    c.candidate_topk_blocks = 0;
    c.candidate_block_size = 0;
  } else {
    c.kv_source_layer_ids = require_int_list(t, "kv_source_layer_ids");
    c.index_source_layer_ids = require_int_list(t, "index_source_layer_ids");
    if (!strictly_increasing(c.kv_source_layer_ids)) reject("text_config.kv_source_layer_ids", "must be strictly increasing");
    if (!strictly_increasing(c.index_source_layer_ids)) reject("text_config.index_source_layer_ids", "must be strictly increasing");
    for (const int l : c.kv_source_layer_ids) {
      if (l < 0 || l >= c.num_hidden_layers) reject("text_config.kv_source_layer_ids", std::format("layer {} outside the backbone", l));
      if (c.compress_ratio(l) == 0) reject("text_config.kv_source_layer_ids", std::format("layer {} is window-only (ratio 0)", l));
      if (!contains(c.index_source_layer_ids, l))
        reject("text_config.kv_source_layer_ids", std::format("layer {} compresses its KV but is not an index source (it must own the index keys)", l));
    }
    for (const int l : c.index_source_layer_ids) {
      if (l < 0 || l >= c.num_hidden_layers) reject("text_config.index_source_layer_ids", std::format("layer {} outside the backbone", l));
      if (c.compress_ratio(l) == 0) reject("text_config.index_source_layer_ids", std::format("layer {} is window-only (ratio 0)", l));
    }
    for (int l = 0; l < c.num_hidden_layers; ++l) {
      if (c.compress_ratio(l) == 0) continue;
      const int kv = c.kv_source_of(l);
      if (kv < 0) reject("text_config.compress_ratios", std::format("layer {} compresses but no kv source precedes it", l));
      if (c.compress_ratio(kv) != c.compress_ratio(l))
        reject("text_config.compress_ratios",
               std::format("layer {} (ratio {}) reads kv source {} (ratio {}); a cache is read at its own ratio", l,
                           c.compress_ratio(l), kv, c.compress_ratio(kv)));
      const int is = c.index_source_of(l);
      if (is < 0 || c.kv_source_of(is) != kv)
        reject("text_config.index_source_layer_ids",
               std::format("layer {} attends with a selection made over a different cache than the one it reads", l));
    }
    c.candidate_source_layer_id = optional_int(t, "candidate_source_layer_id", -1);
    c.candidate_topk_blocks = optional_int(t, "candidate_topk_blocks", 0);
    c.candidate_block_size = optional_int(t, "candidate_block_size", 0);
    if (c.candidate_source_layer_id >= 0) {
      const int cs = c.candidate_source_layer_id;
      if (!contains(c.index_source_layer_ids, cs))
        reject("text_config.candidate_source_layer_id", "must be an index source");
      if (c.candidate_topk_blocks <= 0 || c.candidate_block_size <= 0)
        reject("text_config.candidate_topk_blocks", "must be positive with a candidate source");
      for (const int l : c.index_source_layer_ids)
        if (l > cs && c.kv_source_of(l) != c.kv_source_of(cs))
          reject("text_config.candidate_source_layer_id",
                 std::format("index source {} scores inside the candidate pool but reads a different cache", l));
    }
  }
  c.index_n_heads = require_int(t, "index_n_heads");
  c.index_head_dim = require_int(t, "index_head_dim");
  c.index_topk = require_int(t, "index_topk");
  if (c.index_n_heads <= 0) reject("text_config.index_n_heads", "must be positive");
  if (c.index_head_dim != 128) reject("text_config.index_head_dim", "the indexer implements 128 (the index cache row)");
  if (c.index_topk <= 0 || (c.index_topk & (c.index_topk - 1)) != 0 || c.index_topk > 2048)
    reject("text_config.index_topk", "must be a power of two <= 2048 (the select networks)");
  if (c.candidate_source_layer_id >= 0 && c.candidate_pool_entries() < c.index_topk)
    reject("text_config.candidate_topk_blocks", "the candidate pool must hold at least index_topk entries");
  if (c.qk_rope_head_dim > c.index_head_dim) reject("text_config.qk_rope_head_dim", "exceeds index_head_dim");

  // --- mHC ------------------------------------------------------------------------
  c.hc_mult = require_int(t, "hc_mult");
  c.hc_sinkhorn_iters = require_int(t, "hc_sinkhorn_iters");
  c.hc_eps = static_cast<float>(require_double(t, "hc_eps"));
  if (c.hc_mult != 4) reject("text_config.hc_mult", "the mHC kernel is pinned to 4 streams");
  if (c.hc_sinkhorn_iters < 1) reject("text_config.hc_sinkhorn_iters", "must be >= 1");
  if (!(c.hc_eps > 0)) reject("text_config.hc_eps", "must be positive");

  // --- MoE ------------------------------------------------------------------------
  c.moe_intermediate_size = require_int(t, "moe_intermediate_size");
  c.n_routed_experts = require_int(t, "n_routed_experts");
  c.n_shared_experts = optional_int(t, "n_shared_experts", 0);
  c.num_experts_per_tok = require_int(t, "num_experts_per_tok");
  // The DSpark draft's expert counts default to the main MoE's (the 0731
  // draft stages carry the full 256-expert set; V4.1 overrides them).
  c.dspark_n_routed_experts = optional_int(t, "dspark_n_routed_experts", c.n_routed_experts);
  c.dspark_num_experts_per_tok = optional_int(t, "dspark_num_experts_per_tok", c.num_experts_per_tok);
  if (c.dspark_n_routed_experts <= 0 || c.dspark_n_routed_experts > 4096)
    reject("text_config.dspark_n_routed_experts", "must be in [1, 4096]");
  if (c.dspark_num_experts_per_tok <= 0 || c.dspark_num_experts_per_tok > 16 ||
      c.dspark_num_experts_per_tok > c.dspark_n_routed_experts)
    reject("text_config.dspark_num_experts_per_tok", "must be in [1, min(dspark_n_routed_experts, 16)]");
  c.norm_topk_prob = optional_bool(t, "norm_topk_prob", true);
  c.routed_scaling_factor = static_cast<float>(optional_double(t, "routed_scaling_factor", 1.0));
  c.swiglu_limit = static_cast<float>(optional_double(t, "swiglu_limit", 0.0));
  c.scoring_func = optional_string(t, "scoring_func", "sigmoid");
  c.topk_method = optional_string(t, "topk_method", "noaux_tc");
  if (c.moe_intermediate_size <= 0 || c.moe_intermediate_size % 32 != 0)
    reject("text_config.moe_intermediate_size", "must be a positive multiple of 32 (an MXFP4 block)");
  if (c.n_routed_experts <= 0 || c.n_routed_experts > 4096) reject("text_config.n_routed_experts", "must be in [1, 4096]");
  if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > 16 || c.num_experts_per_tok > c.n_routed_experts)
    reject("text_config.num_experts_per_tok", "must be in [1, min(n_routed_experts, 16)]");
  if (c.n_shared_experts != 1) reject("text_config.n_shared_experts", "the MoE chain implements exactly one shared expert");
  if (!(c.routed_scaling_factor > 0)) reject("text_config.routed_scaling_factor", "must be positive");
  if (!(c.swiglu_limit > 0)) reject("text_config.swiglu_limit", "the release clamps its SwiGLU; a positive limit is required");
  if (c.scoring_func != "sqrtsoftplus" && c.scoring_func != "sigmoid")
    reject("text_config.scoring_func", "sqrtsoftplus and sigmoid are implemented, got '" + c.scoring_func + "'");
  if (c.topk_method != "noaux_tc") reject("text_config.topk_method", "only noaux_tc is implemented");
  if (const int ng = optional_int(t, "n_group", 1); ng != 1) reject("text_config.n_group", "group-limited routing is not implemented");
  if (const int tg = optional_int(t, "topk_group", 1); tg != 1) reject("text_config.topk_group", "group-limited routing is not implemented");
  // The 0731 hash-routed prefix: layers [0, num_hash_layers) pick their
  // expert indices from the tid2eid table; absent in V4.1.
  c.num_hash_layers = optional_int(t, "num_hash_layers", 0);
  if (c.num_hash_layers < 0 || c.num_hash_layers > c.num_hidden_layers)
    reject("text_config.num_hash_layers", "must be in [0, num_hidden_layers]");

  // --- Engram (optional: the V4-Flash-0731 release has no tables) -------------------
  c.engram_layer_ids = t.find("engram_layer_ids") ? require_int_list(t, "engram_layer_ids") : std::vector<int>{};
  c.engram_num_embeddings = t.find("engram_num_embeddings") ? require_int64_list(t, "engram_num_embeddings") : std::vector<int64_t>{};
  c.engram_max_ngram_size = optional_int(t, "engram_max_ngram_size", 4);
  c.engram_vocab_size = optional_int64(t, "engram_vocab_size", 16000000);
  c.engram_n_heads = optional_int(t, "engram_n_heads", 8);
  c.engram_head_dim = optional_int(t, "engram_head_dim", 256);
  c.engram_pad_token_id = optional_int64(t, "engram_pad_token_id", c.pad_token_id);
  c.engram_compressed_vocab_size = optional_int(t, "engram_compressed_vocab_size", 0);
  if (!strictly_increasing(c.engram_layer_ids)) reject("text_config.engram_layer_ids", "must be strictly increasing");
  if (c.engram_num_embeddings.size() != c.engram_layer_ids.size())
    reject("text_config.engram_num_embeddings", "one table size per Engram layer");
  for (const int l : c.engram_layer_ids)
    if (l < 0 || l >= c.num_hidden_layers) reject("text_config.engram_layer_ids", std::format("layer {} outside the backbone", l));
  for (const int64_t n : c.engram_num_embeddings)
    if (n <= 0) reject("text_config.engram_num_embeddings", "must be positive");
  if (!c.engram_layer_ids.empty()) {
    if (c.engram_max_ngram_size < 2) reject("text_config.engram_max_ngram_size", "must be >= 2 (the 2-gram at least)");
    if (c.engram_vocab_size <= 0) reject("text_config.engram_vocab_size", "must be positive");
    if (c.engram_n_heads <= 0) reject("text_config.engram_n_heads", "must be positive");
    if (c.engram_head_dim != 256) reject("text_config.engram_head_dim", "the table row kernels are pinned to 256");
    if (c.engram_pad_token_id < 0 || c.engram_pad_token_id >= c.vocab_size)
      reject("text_config.engram_pad_token_id", "id outside [0, vocab_size)");
    if (c.engram_compressed_vocab_size <= 0 || c.engram_compressed_vocab_size > c.vocab_size)
      reject("text_config.engram_compressed_vocab_size", "must be in [1, vocab_size]");
  }

  parse_quantization(root, c);
  parse_vision(root, c);
  return c;
}

Dsv41TextConfig Dsv41TextConfig::from_json_file(const std::string& path) {
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  return parse(parsed.root);
}

bool Dsv41TextConfig::is_kv_source(int l) const { return contains(kv_source_layer_ids, l); }
bool Dsv41TextConfig::is_index_source(int l) const { return contains(index_source_layer_ids, l); }

bool Dsv41TextConfig::single_pass_pre() const {
  return variant == Dsv41Variant::V41;
}

void Dsv41TextConfig::check_decoder_invariant() const {
  if (variant != Dsv41Variant::V41) return;  // V4: CSA2 throughout, no ratio-1 decoder
  const int dec0 = decoder_first_layer();
  for (int l = dec0; l < num_hidden_layers; ++l)
    if (compress_ratio(l) > 1 || (is_kv_source(l) && l != dec0))
      throw std::invalid_argument("Dsv41Model: the decoder (from the last kv source on) must read that source at ratio 1");
}

Dsv41LayerMode Dsv41TextConfig::layer_mode(int l) const {
  if (l < 0 || l >= max_layer()) throw std::out_of_range("Dsv41TextConfig::layer_mode: layer out of range");
  if (compress_ratio(l) == 0) return Dsv41LayerMode::Window;
  if (is_kv_source(l)) return Dsv41LayerMode::Full;
  if (is_index_source(l)) return Dsv41LayerMode::Reindex;
  return Dsv41LayerMode::Reuse;
}

int Dsv41TextConfig::kv_source_of(int l) const {
  if (l < 0 || l >= max_layer() || compress_ratio(l) == 0) return -1;
  int best = -1;
  for (const int s : kv_source_layer_ids)
    if (s <= l) best = s;
  return best;
}

int Dsv41TextConfig::index_source_of(int l) const {
  if (l < 0 || l >= max_layer() || compress_ratio(l) == 0) return -1;
  int best = -1;
  for (const int s : index_source_layer_ids)
    if (s <= l) best = s;
  return best;
}

bool Dsv41TextConfig::has_engram(int l) const { return contains(engram_layer_ids, l); }

int Dsv41TextConfig::engram_index(int l) const {
  for (size_t i = 0; i < engram_layer_ids.size(); ++i)
    if (engram_layer_ids[i] == l) return static_cast<int>(i);
  return -1;
}

bool Dsv41TextConfig::is_dspark_target(int l) const { return contains(dspark_target_layer_ids, l); }

GlmMoeConfig Dsv41TextConfig::moe_config(int local_inter, bool draft, int layer) const {
  GlmMoeConfig m;
  m.hidden = hidden_size;
  m.inter = local_inter;
  m.n_experts = draft ? dspark_n_routed_experts : n_routed_experts;
  m.top_k = draft ? dspark_num_experts_per_tok : num_experts_per_tok;
  m.n_shared_experts = n_shared_experts;
  m.routed_scaling_factor = routed_scaling_factor;
  m.norm_topk_prob = norm_topk_prob;
  m.swiglu_limit = swiglu_limit;
  m.router_mode = scoring_func == "sigmoid" ? MoeRouterMode::SigmoidBias : MoeRouterMode::SqrtSoftplusBias;
  m.hash_route = !draft && is_hash_layer(layer);
  GlmMoeConfig::validate_config(m);
  return m;
}

// The compressor tails, per kv source (the reference's kv_state and
// score_state, 2*ratio slots of coff*head_dim each): ratio 2 the pair's
// (kv, score) [2, 512]; ratio 4 the overlap window's 16 slots x 1024 (main)
// and 16 x 256 (the indexer compressor); ratio 128 the full window's 256
// slots x 512.
int Dsv41TextConfig::compressor_tail_count() const {
  int n = 0;
  for (const int l : kv_source_layer_ids) {
    const int r = compress_ratio(l);
    if (r == 2)
      ++n;
    else if (r == 4)
      n += 2;
    else if (r == 128) ++n;
  }
  return n;
}

std::vector<int> Dsv41TextConfig::compressor_tail_floats() const {
  // The CSA2 latent (kCsa2Latent) and index (kCsa2IndexDim) widths are the
  // config's head_dim and index_head_dim (512 and 128).
  std::vector<int> floats;
  for (const int l : kv_source_layer_ids) {
    const int r = compress_ratio(l);
    if (r == 2)
      floats.push_back(2 * head_dim);
    else if (r == 4) {
      floats.push_back(16 * 2 * head_dim);
      floats.push_back(16 * 2 * index_head_dim);
    } else if (r == 128)
      floats.push_back(256 * head_dim);
  }
  return floats;
}

}  // namespace dgpp
