// dsv41_load_check: loads a DeepSeek-V4.1-Flash checkpoint through the
// resident loader at one rank's TP geometry and reports per-layer bytes
// and times, the byte formulas against actual usage, the source-byte plan
// against the bytes read, the Engram table mappings and the replicated
// digest (compare across ranks by hand or by the fabric script). No bus,
// no forward: one process per rank.
//
//   dsv41_load_check --model ORG/NAME | --checkpoint-dir DIR
//                    [--world W] [--rank R] [--streaming] [--mtp]
//                    [--layers N] [--from L] [--image-dir DIR|off]
//                    [--engram] [--sidecar]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>

#include "common/log.hpp"
#include "loaders/architecture.hpp"
#include "loaders/hf_cache.hpp"
#include "models/dsv41/config.hpp"
#include "models/dsv41/engram_tables.hpp"
#include "models/dsv41/loader.hpp"

namespace {
const char* mode_name(dgpp::Dsv41LayerMode m) {
  switch (m) {
    case dgpp::Dsv41LayerMode::Window: return "window";
    case dgpp::Dsv41LayerMode::Full: return "full";
    case dgpp::Dsv41LayerMode::Reindex: return "reindex";
    case dgpp::Dsv41LayerMode::Reuse: return "reuse";
  }
  return "?";
}
}  // namespace

int main(int argc, char** argv) {
  std::string model_id, ckpt, image_dir;
  int world = 1, rank = 0, layers = -1, from = 0;
  bool streaming = false, mtp = false, engram = false, sidecar = false;
  auto next = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error("missing value after " + std::string(argv[i]));
    return argv[++i];
  };
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--model") model_id = next(i);
      else if (a == "--checkpoint-dir") ckpt = next(i);
      else if (a == "--world") world = std::stoi(next(i));
      else if (a == "--rank") rank = std::stoi(next(i));
      else if (a == "--streaming") streaming = true;
      else if (a == "--mtp") mtp = true;
      else if (a == "--layers") layers = std::stoi(next(i));
      else if (a == "--from") from = std::stoi(next(i));
      else if (a == "--image-dir") image_dir = next(i);
      else if (a == "--engram") engram = true;
      else if (a == "--sidecar") sidecar = true;
      else throw std::runtime_error("unknown argument " + a);
    }
    if (ckpt.empty()) {
      if (model_id.empty()) throw std::runtime_error("--model or --checkpoint-dir is required");
      std::string err;
      ckpt = dgpp::hf::model_dir(model_id, &err);
      if (ckpt.empty()) throw std::runtime_error("cannot resolve " + model_id + ": " + err);
    }
    const std::string cfg_path = (std::filesystem::path(ckpt) / "config.json").string();
    const dgpp::ModelArchitecture arch = dgpp::detect_architecture_file(cfg_path);
    if (arch != dgpp::ModelArchitecture::DeepseekV41 && arch != dgpp::ModelArchitecture::DeepseekV4)
      throw std::runtime_error("not a DeepseekV4 family checkpoint: " + ckpt);
    const dgpp::Dsv41TextConfig cfg = dgpp::Dsv41TextConfig::from_json_file(cfg_path);
    if (!image_dir.empty()) dgpp::Dsv41LayerStream::set_resident_image_dir(image_dir == "off" ? "" : image_dir);
    const dgpp::Dsv41Residency residency = streaming ? dgpp::Dsv41Residency::Streaming : dgpp::Dsv41Residency::Resident;
    const dgpp::Dsv41HeadSharding head = world > 1 ? dgpp::Dsv41HeadSharding::VocabSharded : dgpp::Dsv41HeadSharding::Full;
    const double kGiB = 1024.0 * 1024.0 * 1024.0;
    DGPP_LOG_INFO("dsv41_load_check: {} world {} rank {} {} — formulas: resident {:.2f} GiB (globals {:.2f}, staging {:.2f}); "
                  "{} layers + {} draft stages, {} experts x {} (shared {}), Engram layers {}",
                  ckpt, world, rank, streaming ? "streaming" : "resident",
                  dgpp::Dsv41LayerStream::resident_bytes(cfg, rank, world, head, mtp) / kGiB,
                  dgpp::Dsv41LayerStream::globals_bytes(cfg, rank, world, head) / kGiB,
                  dgpp::Dsv41LayerStream::staging_plan_bytes(cfg, rank, world, head, mtp) / kGiB,
                  cfg.num_hidden_layers, cfg.num_nextn_predict_layers, cfg.n_routed_experts, cfg.moe_intermediate_size,
                  cfg.shared_expert_inter(), cfg.engram_layer_ids.size());
    if (sidecar) {
      const dgpp::Dsv41EngramSidecar sc = dgpp::dsv41_load_engram_sidecar_for(ckpt, cfg);
      DGPP_LOG_INFO("dsv41_load_check: Engram sidecar ok — {} layers, {} classes, pad class {}, {} primes, tokenizer {}",
                    sc.layers(), sc.compressed_vocab_size, sc.pad_class, sc.primes.size(), sc.tokenizer_sha256.substr(0, 12));
    }
    const auto t0 = std::chrono::steady_clock::now();
    dgpp::Dsv41LayerStream stream(cfg, ckpt, rank, world, residency, head, mtp);
    const auto t1 = std::chrono::steady_clock::now();
    DGPP_LOG_INFO("dsv41_load_check: opened in {:.1f} s; layer capacity {:.2f} GiB",
                  std::chrono::duration<double>(t1 - t0).count(), stream.layer_capacity() / kGiB);
    const dgpp::Dsv41ReplicatedDigest d = stream.hash_replicated();
    DGPP_LOG_INFO("dsv41_load_check: digest globals {:016x} tensors {} bytes {:.2f} GiB; layer 0 {:016x} layer 1 {:016x} last {:016x}",
                  d.globals, d.tensors, d.bytes / kGiB, d.layer[0], d.layer[1], d.layer.back());
    if (engram) {
      const dgpp::Dsv41EngramTables& t = stream.load_engram_tables();
      DGPP_LOG_INFO("dsv41_load_check: Engram tables mmap'ed: {} tables, {:.2f} GiB mapped, {} rows/token/rank", t.tables.size(),
                    t.mapped_bytes() / kGiB, t.rows_local());
    }
    const auto& g = stream.load_globals();
    DGPP_LOG_INFO("dsv41_load_check: globals {:.2f} GiB, lm head rows [{}, +{}), embed rows [{}, +{})", g.bytes / kGiB,
                  g.lm_vocab_begin, g.lm_vocab_count, g.embed_vocab_begin, g.embed_vocab_count);
    const int last = mtp ? cfg.max_layer() : cfg.num_hidden_layers;
    const int n = layers < 0 ? last : std::min(last, from + layers);
    size_t total = 0;
    for (int l = from; l < n; ++l) {
      const auto tl = std::chrono::steady_clock::now();
      const uint64_t before = stream.source_bytes_read();
      const auto& r = stream.load_layer(l);
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count();
      total += r.bytes;
      if (l < from + 4 || l == n - 1 || l % 8 == 0 || cfg.is_draft(l) || r.engram.present)
        DGPP_LOG_INFO("dsv41_load_check: layer {} ({}/{}{}{}{}{}) {:.3f} GiB in {:.2f} s, read {:.3f} GiB; heads {} groups {}; experts {} x inter {} (shared {})",
                      l, mode_name(cfg.layer_mode(l)), cfg.compress_ratio(l), r.attn.index_source() ? ", indexer" : "",
                      r.attn.kv_source() ? ", compressor" : "", r.engram.present ? ", engram" : "",
                      cfg.is_draft(l) ? std::format(", draft stage {}", cfg.draft_stage(l)) : "", r.bytes / kGiB, s,
                      (stream.source_bytes_read() - before) / kGiB, r.attn.local_heads, r.attn.local_groups,
                      r.moe.n_experts, r.moe.local_inter, r.moe.local_shared_inter);
      if (!streaming) continue;
      stream.release_layer();
    }
    if (!streaming) stream.release_sources();
    DGPP_LOG_INFO("dsv41_load_check: {} layers {:.2f} GiB resident (+ globals {:.2f} = {:.2f} GiB); source bytes read {:.2f} GiB (verbatim {:.2f}); image restored {} captured {}; total {:.1f} s",
                  n - from, total / kGiB, g.bytes / kGiB, (total + g.bytes) / kGiB,
                  stream.source_bytes_read() / kGiB, stream.verbatim_source_bytes() / kGiB,
                  stream.image_layers_restored(), stream.image_layers_captured(),
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("dsv41_load_check: {}", e.what());
    return 1;
  }
}
