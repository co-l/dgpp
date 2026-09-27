#pragma once
// The DeepSeek-V4-Flash-0731 config.json (deepseek-ai/DeepSeek-V4-Flash-0731,
// transcribed 2026-09-27 from the revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062
// snapshot on spark1) as the unit tests build it. The flat layout, the fp8
// 128x128 grid, the hash-routed prefix (num_hash_layers), the per-layer CSA2
// schedule (compress_ratios only — no kv/index source lists) and the DSpark
// head (dspark_target_layer_ids) with `num_nextn_predict_layers: 1` while
// three mtp.S stages ship. `config_json(from, to)` patches one anchor.
#include <stdexcept>
#include <string>

namespace dsv4_test {

inline const char* kConfig = R"JSON({
 "architectures": ["DeepseekV4ForCausalLM"],
 "model_type": "deepseek_v4",
 "transformers_version": "4.57.1",
 "torch_dtype": "bfloat16",
 "bos_token_id": 0,
 "eos_token_id": 1,
 "expert_dtype": "fp4",
 "attention_bias": false,
 "attention_dropout": 0.0,
 "initializer_range": 0.02,
 "use_cache": true,
 "tie_word_embeddings": false,
 "quantization_config": {
  "quant_method": "fp8",
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "scale_fmt": "ue8m0",
  "weight_block_size": [128, 128]
 },
 "vocab_size": 129280,
 "hidden_size": 4096,
 "moe_intermediate_size": 2048,
 "num_hidden_layers": 43,
 "num_attention_heads": 64,
 "num_key_value_heads": 1,
 "head_dim": 512,
 "qk_rope_head_dim": 64,
 "q_lora_rank": 1024,
 "o_lora_rank": 1024,
 "o_groups": 8,
 "hidden_act": "silu",
 "swiglu_limit": 10.0,
 "rms_norm_eps": 1e-06,
 "max_position_embeddings": 1048576,
 "rope_theta": 10000,
 "rope_scaling": {
  "type": "yarn",
  "factor": 16,
  "beta_fast": 32,
  "beta_slow": 1,
  "original_max_position_embeddings": 65536
 },
 "n_routed_experts": 256,
 "n_shared_experts": 1,
 "num_experts_per_tok": 6,
 "scoring_func": "sqrtsoftplus",
 "topk_method": "noaux_tc",
 "norm_topk_prob": true,
 "routed_scaling_factor": 1.5,
 "sliding_window": 128,
 "compress_rope_theta": 160000,
 "num_hash_layers": 3,
 "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0],
 "index_n_heads": 64,
 "index_head_dim": 128,
 "index_topk": 512,
 "hc_mult": 4,
 "hc_sinkhorn_iters": 20,
 "hc_eps": 1e-06,
 "num_nextn_predict_layers": 1,
 "dspark_block_size": 5,
 "dspark_noise_token_id": 128799,
 "dspark_target_layer_ids": [40, 41, 42],
 "dspark_markov_rank": 256
})JSON";

inline std::string config_json(const std::string& patch_from = "", const std::string& patch_to = "") {
  std::string s = kConfig;
  if (!patch_from.empty()) {
    const size_t at = s.find(patch_from);
    if (at == std::string::npos) throw std::runtime_error("patch anchor missing: " + patch_from);
    s.replace(at, patch_from.size(), patch_to);
  }
  return s;
}

}  // namespace dsv4_test
