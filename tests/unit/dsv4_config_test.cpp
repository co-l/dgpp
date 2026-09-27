// The DeepSeek-V4-Flash-0731 config parser: the flat release parses, the
// draft depth comes from the mtp blocks (dspark_target_layer_ids) and NOT
// from the file's num_nextn_predict_layers, the hash-routed prefix and the
// fp8 128x128 grid are read, the per-layer CSA2 schedule is derived from
// compress_ratios, and the unsupported shapes are refused by name. The
// architecture registry dispatches on the flat `deepseek_v4` type.
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "dsv4_config_json.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/dsv41/config.hpp"

namespace {

using dsv4_test::config_json;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Dsv41TextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Dsv41TextConfig::parse(t.root);
}

std::string refusal(const std::string& text) {
  try {
    (void)parse(text);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool has(const std::string& msg, const char* needle) { return msg.find(needle) != std::string::npos; }

}  // namespace

DGPP_TEST(dsv4_config_parses_the_release) {
  using dgpp::Dsv41LayerMode;
  using dgpp::Dsv41Variant;
  const dgpp::Dsv41TextConfig c = parse(config_json());
  require(c.variant == Dsv41Variant::V4, "variant");
  require(!c.single_pass_pre(), "the 0731 release collapses each site with its own pre (the GLM form)");
  require(c.hidden_size == 4096 && c.vocab_size == 129280 && c.num_hidden_layers == 43, "shape");
  // The draft depth is the mtp block count (three stages), not the file's
  // `num_nextn_predict_layers: 1`.
  require(c.num_nextn_predict_layers == 3 && c.max_layer() == 46, "draft stages from mtp blocks");
  require(c.rms_norm_eps == 1e-6f && c.max_position_embeddings == 1048576, "eps / positions");
  require(c.bos_token_id == 0 && c.eos_token_id == 1 && c.pad_token_id == -1, "tokens");
  require(c.num_attention_heads == 64 && c.num_key_value_heads == 1 && c.head_dim == 512 && c.qk_rope_head_dim == 64,
          "attention dims");
  require(c.q_lora_rank == 1024 && c.o_lora_rank == 1024 && c.o_groups == 8 && c.heads_per_group() == 8, "lora ranks");
  require(c.sliding_window == 128 && c.qk_nope_head_dim() == 448, "window / nope");
  require(c.rope_theta == 10000.0 && c.compress_rope_theta == 160000.0, "thetas");
  require(c.rope_factor == 16.0 && c.original_max_position_embeddings == 65536 && c.beta_fast == 32.0 && c.beta_slow == 1.0,
          "yarn");
  // The per-layer CSA2 schedule derived from compress_ratios.
  require(c.compress_ratios.size() == 46 && c.compress_ratio(0) == 0 && c.compress_ratio(1) == 0 &&
              c.compress_ratio(2) == 4 && c.compress_ratio(3) == 128 && c.compress_ratio(41) == 128 &&
              c.compress_ratio(42) == 4 && c.compress_ratio(43) == 0 && c.compress_ratio(45) == 0,
          "ratios");
  require(c.num_kv_sources() == 41 && c.num_index_sources() == 21, "source counts");
  require(c.layer_mode(0) == Dsv41LayerMode::Window && c.layer_mode(1) == Dsv41LayerMode::Window, "window layers");
  require(c.layer_mode(2) == Dsv41LayerMode::Full && c.layer_mode(3) == Dsv41LayerMode::Full &&
              c.layer_mode(42) == Dsv41LayerMode::Full,
          "every compressed layer is its own kv source");
  require(c.layer_mode(43) == Dsv41LayerMode::Window && c.is_draft(43) && c.draft_stage(45) == 2 && !c.is_draft(42),
          "draft modes");
  require(c.kv_source_of(2) == 2 && c.kv_source_of(3) == 3 && c.kv_source_of(42) == 42 && c.kv_source_of(0) == -1,
          "own cache per compressed layer");
  require(c.index_source_of(2) == 2 && c.index_source_of(4) == 4 && c.index_source_of(42) == 42, "C4A index sources");
  require(!c.is_candidate_source(0) && !c.uses_candidates(20) && c.candidate_source_layer_id == -1, "no candidate pool");
  require(c.index_n_heads == 64 && c.index_topk == 512 && c.index_head_dim == 128, "indexer");
  // The hash-routed prefix.
  require(c.num_hash_layers == 3 && c.has_tid2eid(), "hash layers");
  require(c.is_hash_layer(0) && c.is_hash_layer(2) && !c.is_hash_layer(3) && !c.is_hash_layer(-1), "hash prefix");
  require(c.hc_mult == 4 && c.hc_sinkhorn_iters == 20 && c.hc_eps == 1e-6f && c.hc_coeff_rows() == 24 &&
              c.hc_head_rows() == 4, "mhc");
  require(c.moe_intermediate_size == 2048 && c.n_routed_experts == 256 && c.num_experts_per_tok == 6 &&
              c.n_shared_experts == 1 && c.shared_expert_inter() == 2048,
          "moe");
  require(c.routed_scaling_factor == 1.5f && c.norm_topk_prob && c.swiglu_limit == 10.0f &&
              c.scoring_func == "sqrtsoftplus" && c.topk_method == "noaux_tc",
          "router");
  require(c.engram_layer_ids.empty() && c.engram_num_embeddings.empty(), "no engram in 0731");
  require(!c.vision_present, "no vision in 0731");
  require(c.dspark_block_size == 5 && c.dspark_noise_token_id == 128799 && c.dspark_markov_rank == 256, "dspark");
  require(c.dspark_target_layer_ids.size() == 3 && c.is_dspark_target(40) && c.is_dspark_target(42) && !c.is_dspark_target(39),
          "dspark targets");
  require(c.dspark_n_routed_experts == 256 && c.dspark_num_experts_per_tok == 6, "dspark experts default to the main MoE");
  require(c.fp8_block_size == 128 && c.fp4_block_size == 32, "blocks");
  const dgpp::GlmMoeConfig m = c.moe_config(512, false);
  require(m.inter == 512 && m.n_experts == 256 && m.top_k == 6 && m.n_shared_experts == 1 && m.swiglu_limit == 10.0f,
          "moe_config");
  require(m.router_mode == dgpp::MoeRouterMode::SqrtSoftplusBias && m.routed_scaling_factor == 1.5f, "moe_config router");
  // The 0731 hash-routed prefix: the first num_hash_layers backbone layers
  // select their experts from the tid2eid table; the rest, and the draft,
  // route from the scores.
  require(c.moe_config(512, false, 0).hash_route && c.moe_config(512, false, 2).hash_route &&
              !c.moe_config(512, false, 3).hash_route,
          "moe_config hash_route on the prefix");
  const dgpp::GlmMoeConfig d = c.moe_config(512, true);
  require(d.n_experts == 256 && d.top_k == 6, "draft moe_config");
  require(!m.hash_route && !d.hash_route, "moe_config hash_route off elsewhere");
}

DGPP_TEST(dsv4_config_scale_shift) {
  const dgpp::Dsv41TextConfig c = parse(config_json());
  require(c.fp8_block_size == 128, "the 0731 grid");
  require(c.scale_shift() == 7, "the 128 x 128 scale grid's shift");
}

DGPP_TEST(dsv4_config_refuses_unsupported_shapes) {
  require(has(refusal(config_json("\"model_type\": \"deepseek_v4\"", "\"model_type\": \"deepseek_v3\"")), "model_type"), "type");
  require(has(refusal(config_json("\"num_key_value_heads\": 1", "\"num_key_value_heads\": 2")), "num_key_value_heads"), "kv heads");
  require(has(refusal(config_json("\"head_dim\": 512", "\"head_dim\": 256")), "head_dim"), "head dim");
  require(has(refusal(config_json("\"qk_rope_head_dim\": 64", "\"qk_rope_head_dim\": 32")), "qk_rope_head_dim"), "rope dim");
  require(has(refusal(config_json("\"o_groups\": 8", "\"o_groups\": 7")), "o_groups"), "groups");
  require(has(refusal(config_json("\"hc_mult\": 4", "\"hc_mult\": 2")), "hc_mult"), "hc_mult");
  require(has(refusal(config_json("\"scoring_func\": \"sqrtsoftplus\"", "\"scoring_func\": \"softmax\"")), "scoring_func"), "scoring");
  require(has(refusal(config_json("\"index_head_dim\": 128", "\"index_head_dim\": 64")), "index_head_dim"), "index dim");
  require(has(refusal(config_json("\"index_topk\": 512", "\"index_topk\": 500")), "index_topk"), "topk pow2");
  require(has(refusal(config_json("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true")), "tie_word_embeddings"), "tie");
  require(has(refusal(config_json("\"num_hash_layers\": 3", "\"num_hash_layers\": 44")), "num_hash_layers"), "hash layers");
  // The 0731 release's own fp8 grid is 128x128; a 32x32 grid is not this variant.
  require(has(refusal(config_json("\"weight_block_size\": [128, 128]", "\"weight_block_size\": [32, 32]")), "weight_block_size"),
          "fp8 block");
  require(has(refusal(config_json("\"weight_block_size\": [128, 128]", "\"weight_block_size\": [128, 64]")), "weight_block_size"),
          "fp8 block cols");
  // The compressor implements ratios 0, 4 and 128 here (the v4.1 pair ratios
  // are not this variant).
  {
    const std::string anchor = "\"compress_ratios\": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0]";
    require(has(refusal(config_json(anchor, "\"compress_ratios\": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]")),
                "compress_ratios"), "v4.1 pair ratios");
    require(has(refusal(config_json(anchor, "\"compress_ratios\": [0, 0, 4, 128, 4, 128]")), "compress_ratios"), "ratio length");
    require(has(refusal(config_json(anchor, "\"compress_ratios\": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 4, 0, 0]")),
                "draft stage"), "draft ratio");
  }
  require(has(refusal(config_json("\"quant_method\": \"fp8\"", "\"quant_method\": \"modelopt\"")), "quant_method"), "method");
  require(has(refusal(config_json("\"scale_fmt\": \"ue8m0\"", "\"scale_fmt\": null")), "scale_fmt"), "scale fmt");
  require(has(refusal(config_json("\"expert_dtype\": \"fp4\"", "\"expert_dtype\": \"nvfp4\"")), "expert_dtype"), "nvfp4 cast");
}

DGPP_TEST(dsv4_config_decoder_invariant) {
  const dgpp::Dsv41TextConfig c = parse(config_json());
  // V4 has no ratio-1 decoder: the CSA2 compressors run at their 4/128
  // ratios throughout, so the V4.1 CED-split rule must not fire here.
  c.check_decoder_invariant();
}

DGPP_TEST(dsv4_architecture_registry_dispatches) {
  const auto d = dgpp::minijson::parse(R"({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"})");
  require(dgpp::detect_architecture(d.root) == dgpp::ModelArchitecture::DeepseekV4, "deepseek_v4");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::DeepseekV4)) == "deepseek_v4", "name");
  const auto t = dgpp::minijson::parse(R"({"model_type": "deepseek_v4"})");
  require(dgpp::detect_architecture(t.root) == dgpp::ModelArchitecture::DeepseekV4, "by model_type");
  // The V4.1 class name is a strict prefix of the V4 one — the longer name
  // must still route to DeepseekV41.
  const auto v41 = dgpp::minijson::parse(R"({"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41"})");
  require(dgpp::detect_architecture(v41.root) == dgpp::ModelArchitecture::DeepseekV41, "deepseek_v41 unchanged");
  bool refused = false;
  try {
    const auto v3 = dgpp::minijson::parse(R"({"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3"})");
    (void)dgpp::detect_architecture(v3.root);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  require(refused, "DeepSeek-V3 is not this family");
}
