#pragma once
// Synthetic mini-checkpoint writer for the DeepSeek-V4.1-Flash tests:
// enumerates the binding table for a small config and writes config.json
// + one safetensors shard, so fixture and table cannot disagree. Values
// are deterministic per tensor NAME (glm_rng's scheme) in the release's
// own formats: fp8 e4m3 payloads with e8m0 scales on 32 x 32 blocks,
// MXFP4 experts (random e2m1 nibbles, e8m0 per 32), fp8 + e8m0 Engram
// tables, bf16 / fp32 for the rest — magnitudes that keep a forward's
// nonlinearities informative.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "glm_rng.hpp"
#include "loaders/minijson.hpp"
#include "models/dsv41/binding.hpp"
#include "models/dsv41/config.hpp"
#include "models/dsv41/engram_tables.hpp"

namespace dsv41fx {

namespace fs = std::filesystem;
using dgpp::Dsv41ExpectedTensor;
using dgpp::Dsv41TensorRole;
using dgpp::Dsv41TextConfig;
using dgpp::Dsv41WeightClass;
using dgpp::float_to_bf16_bits;
using glmrng::Rng;
using glmrng::seed_for;

// The tiny release: hidden 256, 16 heads on the 512-wide latent (4 heads
// per rank at world 4 — the attention head tiling's floor; the listed
// flash path at world 1 —, 4 output groups of 4 heads), q_lora 128, o_lora
// 64, a 32 x 128 indexer (the select kernels pin 32 heads) selecting 16 of a 2 x 8 candidate pool, eight
// backbone layers with the CSA2 schedule [window, window, full/2,
// reuse/2, full/1, reuse/1, reindex/1, reuse/1] (kv sources 2 and 4,
// index sources 2, 4 and 6, candidates from 4), Engram on layers 1 and 3
// (4 hash heads x 3 n-gram sizes; table rows 1276 = the sum of the
// primes 79..137 and 2040 = 139..197, so a sidecar with those bucket
// moduli fits exactly), 8 routed experts of 256 (64 per rank at world
// 4) top-2 sqrtsoftplus, vocab 512, two DSpark stages (block 5, targets
// 5-7, Markov rank 32, 4 experts top-2, noise token 500).
inline const char* tiny_config_json() {
  return R"json({
 "architectures": ["DeepseekV41ForCausalLM"],
 "model_type": "deepseek_v41",
 "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
 "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic",
                         "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
 "text_config": {
  "model_type": "deepseek_v41_text",
  "vocab_size": 512, "hidden_size": 256, "moe_intermediate_size": 256,
  "num_hidden_layers": 8, "num_attention_heads": 16, "num_key_value_heads": 1,
  "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 128, "o_lora_rank": 64, "o_groups": 4,
  "hidden_act": "silu", "swiglu_limit": 10.0, "rms_norm_eps": 1e-20,
  "max_position_embeddings": 4096, "sliding_window": 16,
  "rope_theta": 10000.0, "compress_rope_theta": 160000.0,
  "rope_scaling": {"rope_type": "yarn", "factor": 16.0, "original_max_position_embeddings": 256,
                   "beta_fast": 32, "beta_slow": 1},
  "num_nextn_predict_layers": 3,
  "compress_ratios": [0, 0, 2, 2, 1, 1, 1, 1, 0, 0, 0],
  "kv_source_layer_ids": [2, 4], "index_source_layer_ids": [2, 4, 6],
  "candidate_source_layer_id": 4, "candidate_topk_blocks": 2, "candidate_block_size": 8,
  "index_n_heads": 32, "index_head_dim": 128, "index_topk": 16,
  "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
  "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2, "norm_topk_prob": true,
  "routed_scaling_factor": 1.5, "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
  "engram_layer_ids": [1, 3], "engram_num_embeddings": [1276, 2040], "engram_max_ngram_size": 4,
  "engram_vocab_size": 16000, "engram_n_heads": 4, "engram_head_dim": 256, "engram_pad_token_id": 2,
  "engram_compressed_vocab_size": 300,
  "dspark_block_size": 5, "dspark_noise_token_id": 500, "dspark_target_layer_ids": [5, 6, 7],
  "dspark_markov_rank": 32, "dspark_n_routed_experts": 4, "dspark_num_experts_per_tok": 2
 }
})json";
}

inline Dsv41TextConfig tiny_config() {
  const auto t = dgpp::minijson::parse(tiny_config_json());
  return Dsv41TextConfig::parse(t.root);
}

inline bool has(const std::string& name, const char* needle) { return name.find(needle) != std::string::npos; }

// The bytes of one tensor.
inline std::vector<uint8_t> tensor_bytes(const Dsv41ExpectedTensor& e) {
  std::vector<uint8_t> out(e.nbytes());
  const std::string& name = e.name;
  Rng rng(seed_for(name));
  const size_t n = e.numel();
  switch (e.role) {
    case Dsv41TensorRole::Fp8Payload:
    case Dsv41TensorRole::TablePayload:
      // e4m3 codes of ~N(0, 0.67): the scales below put the weights near
      // 0.03 and a table row near 0.05.
      for (size_t i = 0; i < n; ++i) out[i] = dgpp::float_to_fp8_e4m3_bits(2.0f * rng.normal3());
      return out;
    case Dsv41TensorRole::Fp8Scale:
    case Dsv41TensorRole::TableScale:
      for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(121 + rng.next() % 3);  // 2^-6 .. 2^-4
      return out;
    case Dsv41TensorRole::Fp4Payload:
      for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(rng.next());  // every e2m1 code is finite
      return out;
    case Dsv41TensorRole::Fp4Scale:
      for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(118 + rng.next() % 3);  // 2^-9 .. 2^-7
      return out;
    case Dsv41TensorRole::Plain:
    case Dsv41TensorRole::Skipped:
      break;
  }
  const bool is_norm = has(name, "norm") && e.shape.size() == 1;  // gains near 1
  const bool is_bias = has(name, ".bias");
  const bool is_hc_scale = e.cls == Dsv41WeightClass::Mhc && has(name, "_scale");
  const bool is_small = e.cls == Dsv41WeightClass::Mhc || has(name, "attn_sink") || is_bias;
  const bool is_embed = e.cls == Dsv41WeightClass::Embed || e.cls == Dsv41WeightClass::LmHead ||
                        has(name, "markov_head.embed");
  for (size_t i = 0; i < n; ++i) {
    float v;
    if (is_norm) v = 0.9f + 0.2f * (0.5f * (rng.unit() + 1.0f));
    else if (is_hc_scale) v = 1.0f + 0.1f * rng.unit();
    else if (is_small) v = 0.02f * rng.normal3();
    else if (is_embed) v = 0.3f * rng.normal3();
    else v = 0.05f * rng.normal3();  // projections, routers, indexers, compressors, gates
    if (e.dtype == dgpp::DType::BF16) {
      const uint16_t bits = float_to_bf16_bits(v);
      std::memcpy(&out[i * 2], &bits, 2);
    } else if (e.dtype == dgpp::DType::F32) {
      std::memcpy(&out[i * 4], &v, 4);
    } else {
      throw std::runtime_error("fixture dtype not handled: " + name);
    }
  }
  return out;
}

inline std::vector<uint8_t> fixture_bytes(const Dsv41ExpectedTensor& e) { return tensor_bytes(e); }

// Writes `dir` (config.json + one safetensors shard holding every text
// tensor, the Engram tables included) for `cfg`.
inline void write_fixture(const Dsv41TextConfig& cfg, const std::string& dir, const char* config_json) {
  fs::path root(dir);
  fs::remove_all(root);
  fs::create_directories(root);
  {
    const fs::path p = root / "config.json";
    std::FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write config.json");
    std::fwrite(config_json, 1, std::strlen(config_json), f);
    std::fclose(f);
  }
  const auto table = dgpp::dsv41_expected_text_tensors(cfg);
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  bool first = true;
  for (const auto& e : table) {
    const auto b = fixture_bytes(e);
    std::string shape = "[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) shape += ",";
      shape += std::to_string(e.shape[i]);
    }
    shape += "]";
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + std::string(dgpp::dtype_name(e.dtype)) +
              "\",\"shape\":" + shape + ",\"data_offsets\":[" + std::to_string(off) + "," +
              std::to_string(off + b.size()) + "]}";
    data.insert(data.end(), b.begin(), b.end());
    off += b.size();
  }
  header += "}";
  const fs::path shard = root / "model.safetensors";
  std::FILE* f = std::fopen(shard.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write shard");
  const uint64_t hlen = header.size();
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header.data(), 1, hlen, f);
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  std::printf("dsv41 fixture written: %zu tensors, %.2f MB payload\n", table.size(),
              static_cast<double>(data.size()) / 1048576.0);
}

inline void write_fixture(const Dsv41TextConfig& cfg, const std::string& dir) {
  write_fixture(cfg, dir, tiny_config_json());
}

// The Engram sidecar (tools/dsv41_engram_tables.py's output) for the
// fixture: bucket moduli that sum to the table rows exactly (79..137 and
// 139..197), a token map over the compressed classes (id mod classes);
// break_offsets writes one offset off its running sum.
inline std::string sidecar_json(const Dsv41TextConfig& cfg, bool break_offsets) {
  const std::vector<std::vector<int>> primes = {
      {79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137},
      {139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197}};
  std::string s = "{\"format\": \"dgpp-dsv41-engram-tables-1\", \"tokenizer_sha256\": \"fixture\", ";
  s += "\"vocab_size\": " + std::to_string(cfg.vocab_size) + ", \"compressed_vocab_size\": " +
       std::to_string(cfg.engram_compressed_vocab_size) + ", \"pad_id\": " + std::to_string(cfg.engram_pad_token_id) +
       ", \"pad_class\": 2, \"layer_ids\": [1, 3], \"max_ngram_size\": 4, \"n_heads\": 4, ";
  s += "\"num_embeddings\": [1276, 2040], \"primes\": [";
  std::string offsets = "[";
  for (size_t l = 0; l < 2; ++l) {
    if (l) { s += ", "; offsets += ", "; }
    s += "[["; offsets += "[[";
    int64_t run = 0;
    for (size_t i = 0; i < 12; ++i) {
      if (i && i % 4 == 0) { s += "], ["; offsets += "], ["; }
      else if (i) { s += ", "; offsets += ", "; }
      s += std::to_string(primes[l][i]);
      offsets += std::to_string(break_offsets && i == 5 ? run + 1 : run);
      run += primes[l][i];
    }
    s += "]]"; offsets += "]]";
  }
  s += "], \"offsets\": " + offsets + "], \"multipliers\": [[3, 5, 7, 11], [13, 17, 19, 23]], \"token_map\": [";
  for (int t = 0; t < cfg.vocab_size; ++t) {
    if (t) s += ", ";
    s += std::to_string(t % cfg.engram_compressed_vocab_size);
  }
  return s + "]}";
}

// The fixture's sidecar as the loader's struct (no file needed).
inline dgpp::Dsv41EngramSidecar fixture_sidecar(const Dsv41TextConfig& cfg, const std::string& dir) {
  const fs::path p = fs::path(dir) / dgpp::kDsv41EngramSidecarName;
  const std::string text = sidecar_json(cfg, false);
  std::FILE* f = std::fopen(p.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write the fixture sidecar");
  std::fwrite(text.data(), 1, text.size(), f);
  std::fclose(f);
  return dgpp::dsv41_load_engram_sidecar_for(dir, cfg);
}

}  // namespace dsv41fx
