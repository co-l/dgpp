#pragma once
// DeepSeek-V4.1-Flash (DeepseekV41ForCausalLM, model_type deepseek_v41)
// configuration, parsed from the checkpoint's config.json (2026-09-13,
// docs/deepseek_v41_flash_plan.md §1.1). The text model sits under
// `text_config`; the root carries the token ids, the vision config and the
// quantization contract. The same policy as the other four parsers: every
// field the assembly consumes is parsed into a known-supported value or
// rejected with a message naming the field, at load time. The reference is
// the checkpoint's own `inference/model.py` (ModelArgs / Transformer).
//
// The quantization contract (the release as DeepSeek ships it, plan §0.1):
// `quant_method` fp8 with `weight_block_size` [32, 32] and `scale_fmt`
// ue8m0 for every dense projection (`X.weight` F8_E4M3 [N, K] + `X.scale`
// F8_E8M0 [N/32, K/32]), `expert_dtype` fp4 for the routed and draft
// experts (`X.weight` I8 [N, K/2] — two e2m1 codes per byte, the low
// nibble first — + `X.scale` F8_E8M0 [N, K/32]), the Engram tables F8_E4M3
// [rows, 256] + F8_E8M0 [rows, 8], everything else BF16 / F32. Nothing is
// requantized: the engine consumes these bytes as they are (plan D2).
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm/moe.hpp"

namespace dgpp {

// The two DeepSeek-V4 checkpoints this config parser serves: the V4.1
// release (nested text_config, Engram, vision, fp8 32x32 grids, explicit
// kv/index source lists) and the V4-Flash-0731 release (flat config, no
// Engram, no vision, fp8 128x128 grids, per-layer CSA2 caches derived from
// compress_ratios, hash routing on the first layers).
enum class Dsv41Variant : uint8_t { V41, V4 };

// How a layer's CSA2 attention obtains its main KV, index keys and top-k
// selection (tech report §2.3.1; plan §1.3).
enum class Dsv41LayerMode : uint8_t {
  Window,   // compress_ratio 0: the sliding window only (layers 0, 1, the draft)
  Full,     // a kv source: compresses its own main KV, owns index keys, selects
  Reindex,  // an index source that reads the last kv source's cache and keys
  Reuse,    // attends with the last index source's selection
};

struct Dsv41VisionConfig {
  // Present in the file, never executed (plan D9): the shapes below only
  // let the binding list the vision tensors as present-and-skipped.
  int num_hidden_layers = 32;
  int hidden_size = 1024;
  int num_attention_heads = 16;
  int intermediate_size = 2816;
  int patch_size = 14;
  int downsample_ratio = 3;
};

struct Dsv41TextConfig {
  // Which release the config describes (drives the layout, the fp8 grid and
  // the CSA2 schedule derivation).
  Dsv41Variant variant = Dsv41Variant::V41;

  // --- model shape -------------------------------------------------------
  int hidden_size = 5120;
  int vocab_size = 129280;
  int num_hidden_layers = 40;
  float rms_norm_eps = 1e-20f;
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  int max_position_embeddings = 1048576;
  int64_t bos_token_id = 0;
  int64_t eos_token_id = 1;
  int64_t pad_token_id = 2;
  int64_t image_token_id = 129264;

  // --- attention (plan §1.3) -------------------------------------------
  int num_attention_heads = 64;
  int num_key_value_heads = 1;   // one 512-wide latent per token, K == V
  int head_dim = 512;            // the latent width, rope tail included
  int qk_rope_head_dim = 64;
  int q_lora_rank = 1280;
  int o_lora_rank = 1024;
  int o_groups = 8;
  bool attention_bias = false;
  int sliding_window = 128;
  double rope_theta = 10000.0;          // window-only layers, no scaling
  double compress_rope_theta = 160000.0; // every layer with compress_ratio > 0, with YaRN
  double rope_factor = 16.0;
  int original_max_position_embeddings = 65536;
  double beta_fast = 32.0;
  double beta_slow = 1.0;

  // --- CSA2 schedule ------------------------------------------------------
  // One entry per layer, the draft layers included: 0 = window only, r =
  // the main KV compressed r-to-1.
  std::vector<int> compress_ratios;
  std::vector<int> kv_source_layer_ids;     // Full layers
  std::vector<int> index_source_layer_ids;  // Full + Reindex layers
  int candidate_source_layer_id = -1;       // -1: no candidate pool
  int candidate_topk_blocks = 0;
  int candidate_block_size = 0;
  int index_n_heads = 32;
  int index_head_dim = 128;
  int index_topk = 512;

  // --- mHC (single-pass) ----------------------------------------------
  int hc_mult = 4;
  int hc_sinkhorn_iters = 20;
  float hc_eps = 1e-6f;

  // --- MoE ------------------------------------------------------------------
  int moe_intermediate_size = 2304;
  int n_routed_experts = 384;
  int n_shared_experts = 1;
  int num_experts_per_tok = 6;
  bool norm_topk_prob = true;
  float routed_scaling_factor = 1.5f;
  float swiglu_limit = 10.0f;
  std::string scoring_func = "sqrtsoftplus";  // or sigmoid
  std::string topk_method = "noaux_tc";
  // The hash-routed prefix (0731 only): the first `num_hash_layers` layers
  // pick their expert indices from the static tid2eid table (vocab x topk,
  // int32) instead of the top-k of the router scores; the scores still
  // weight the expert outputs (reference Gate.forward).
  int num_hash_layers = 0;

  // --- Engram (plan §1.6) -----------------------------------------------
  std::vector<int> engram_layer_ids;
  std::vector<int64_t> engram_num_embeddings;
  int engram_max_ngram_size = 4;
  int64_t engram_vocab_size = 16000000;
  int engram_n_heads = 8;
  int engram_head_dim = 256;
  int64_t engram_pad_token_id = 2;
  int engram_compressed_vocab_size = 99092;

  // --- DSpark (plan §1.7) -----------------------------------------------
  int num_nextn_predict_layers = 3;
  int dspark_block_size = 5;
  int64_t dspark_noise_token_id = 128799;
  std::vector<int> dspark_target_layer_ids;
  int dspark_markov_rank = 256;
  int dspark_n_routed_experts = 128;
  int dspark_num_experts_per_tok = 3;

  // --- weight formats ----------------------------------------------------
  int fp8_block_size = 32;   // one e8m0 scale per 32x32 block of a dense fp8 matrix
  int fp4_block_size = 32;   // one e8m0 scale per 32 codes along K of an fp4 matrix
  bool vision_present = false;
  Dsv41VisionConfig vision;

  // Parses config.json's root object. Throws std::runtime_error naming the
  // offending field on anything unsupported.
  static Dsv41TextConfig parse(const minijson::Value& root);
  static Dsv41TextConfig from_json_file(const std::string& path);

  // --- derived layer facts ------------------------------------------------
  // Layers are indexed 0..num_hidden_layers-1 (the backbone) and
  // num_hidden_layers..max_layer()-1 (the DSpark draft stages, checkpoint
  // prefix `mtp.S.`).
  int max_layer() const { return num_hidden_layers + num_nextn_predict_layers; }
  bool is_draft(int l) const { return l >= num_hidden_layers && l < max_layer(); }
  int draft_stage(int l) const { return is_draft(l) ? l - num_hidden_layers : -1; }
  int compress_ratio(int l) const { return compress_ratios[static_cast<size_t>(l)]; }
  bool is_kv_source(int l) const;
  bool is_index_source(int l) const;
  Dsv41LayerMode layer_mode(int l) const;
  // The kv source whose cache and index keys layer `l` reads (-1 for a
  // window-only layer); the index source whose selection it attends with.
  int kv_source_of(int l) const;
  int index_source_of(int l) const;
  bool is_candidate_source(int l) const { return l == candidate_source_layer_id; }
  // Reindex layers after the candidate source score only its candidate pool.
  bool uses_candidates(int l) const {
    return candidate_source_layer_id >= 0 && candidate_source_layer_id < l && is_index_source(l);
  }
  int candidate_pool_entries() const { return candidate_topk_blocks * candidate_block_size; }
  int num_kv_sources() const { return static_cast<int>(kv_source_layer_ids.size()); }
  // The CED split (plan §1.8): the decoder begins at the last kv source
  // (layer 20: its cache is the decoder's global KV, a projection of the
  // encoder output); every layer from it on reads that cache at ratio 1.
  int decoder_first_layer() const { return kv_source_layer_ids.empty() ? num_hidden_layers : kv_source_layer_ids.back(); }
  // The model-side backstop on the decoder structure: the V4.1 CED split
  // reads the last kv source's cache at ratio 1 from that source on; V4 has
  // no ratio-1 decoder — its CSA2 compressors run at their ratios throughout.
  void check_decoder_invariant() const;
  // The V4.1 release collapses each site with the previous site's pre
  // coefficients (the single-pass form); the 0731 release (and the GLM
  // form) collapses with the site's own pre.
  bool single_pass_pre() const;
  int num_index_sources() const { return static_cast<int>(index_source_layer_ids.size()); }
  bool has_engram(int l) const;
  int engram_index(int l) const;  // ordinal among the Engram layers, -1 otherwise
  int engram_rows_per_layer() const { return (engram_max_ngram_size - 1) * engram_n_heads; }
  int engram_width() const { return engram_rows_per_layer() * engram_head_dim; }
  bool is_dspark_target(int l) const;
  // The 0731 hash-routed prefix: layer l picks experts from the tid2eid table.
  bool is_hash_layer(int l) const { return l >= 0 && l < num_hash_layers; }
  bool has_tid2eid() const { return num_hash_layers > 0; }
  int shared_expert_inter() const { return n_shared_experts * moe_intermediate_size; }
  int qk_nope_head_dim() const { return head_dim - qk_rope_head_dim; }
  int heads_per_group() const { return num_attention_heads / o_groups; }
  int hc_coeff_rows() const { return (2 + hc_mult) * hc_mult; }
  // The V4 head collapse is a single hc_mult-row projection (reference
  // Transformer: hc_head_fn [hc_mult, hc_mult * hidden]); distinct from the
  // per-layer mHC mixing matrices (hc_coeff_rows, (2+hc_mult)*hc_mult rows).
  int hc_head_rows() const { return hc_mult; }

  // The routed chain's configuration (models/glm/moe.hpp) for a backbone
  // layer or a draft stage: the sqrtsoftplus router with its bias, the
  // shared expert in the chain, the clamped SwiGLU. `local_inter` is this
  // rank's slice of moe_intermediate_size; `layer` sets the 0731 hash
  // routing on the first num_hash_layers backbone layers.
  GlmMoeConfig moe_config(int local_inter, bool draft, int layer = -1) const;
};

}  // namespace dgpp
