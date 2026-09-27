// Idle-nap policy tests for the bus engine thread. The policy is a pure
// function (src/net/bus_idle_policy.hpp) so the gate can be pinned down
// host-side without RDMA: legacy knob-unset behavior must stay byte-identical
// to the pre-knob engine loop, and the knob-set path must nap only once the
// engine has been truly quiet for at least the sleep duration (never between
// decode windows).
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "net/bus_idle_policy.hpp"

namespace {

using dgpp::net::BusIdlePolicyInputs;
using dgpp::net::bus_idle_sleep_us;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

constexpr uint64_t kLegacy = 50;
constexpr uint64_t kKnob = 5000;

BusIdlePolicyInputs base() {
  return BusIdlePolicyInputs{
      .idle_count_passed = true,
      .window_live = false,
      .stream_outstanding = false,
      .graph_recorded = false,
      .quiet_us = 100000,
      .idle_sleep_us = 0,
      .legacy_sleep_us = kLegacy,
  };
}

DGPP_TEST(bus_idle_legacy_naps_before_first_graph) {
  const auto in = base();
  require(bus_idle_sleep_us(in) == kLegacy,
          "legacy mode must keep the 50us pre-serving nap");
}

DGPP_TEST(bus_idle_legacy_hot_spins_once_a_graph_is_recorded) {
  auto in = base();
  in.graph_recorded = true;
  require(bus_idle_sleep_us(in) == 0,
          "legacy mode must hot-spin once a decode graph exists");
}

DGPP_TEST(bus_idle_legacy_never_naps_with_a_live_window) {
  auto in = base();
  in.window_live = true;
  require(bus_idle_sleep_us(in) == 0,
          "legacy mode must not nap while a replay window is live");
}

DGPP_TEST(bus_idle_legacy_never_naps_with_an_outstanding_stream) {
  auto in = base();
  in.stream_outstanding = true;
  require(bus_idle_sleep_us(in) == 0,
          "legacy mode must not nap with an outstanding stream collective");
}

DGPP_TEST(bus_idle_legacy_requires_the_idle_count) {
  auto in = base();
  in.idle_count_passed = false;
  require(bus_idle_sleep_us(in) == 0,
          "legacy mode must not nap before the spin-iteration threshold");
}

DGPP_TEST(bus_idle_knob_naps_after_recorded_when_truly_quiet) {
  auto in = base();
  in.graph_recorded = true;
  in.idle_sleep_us = kKnob;
  in.quiet_us = kKnob;
  require(bus_idle_sleep_us(in) == kKnob,
          "knob mode must deep-nap post-recording once quiet >= knob");
}

DGPP_TEST(bus_idle_knob_quiet_must_clear_the_grace) {
  auto in = base();
  in.graph_recorded = true;
  in.idle_sleep_us = kKnob;
  in.quiet_us = kKnob - 1;
  require(bus_idle_sleep_us(in) == 0,
          "knob mode must not nap before the quiet grace elapses");
}

DGPP_TEST(bus_idle_knob_quiet_equal_to_grace_naps) {
  auto in = base();
  in.graph_recorded = true;
  in.idle_sleep_us = kKnob;
  in.quiet_us = kKnob;
  require(bus_idle_sleep_us(in) == kKnob,
          "knob mode must nap exactly at the quiet-grace boundary");
}

DGPP_TEST(bus_idle_knob_never_naps_with_a_live_window) {
  auto in = base();
  in.graph_recorded = true;
  in.idle_sleep_us = kKnob;
  in.quiet_us = kKnob;
  in.window_live = true;
  require(bus_idle_sleep_us(in) == 0,
          "knob mode must not nap while a replay window is live");
}

DGPP_TEST(bus_idle_knob_never_naps_with_an_outstanding_stream) {
  auto in = base();
  in.graph_recorded = true;
  in.idle_sleep_us = kKnob;
  in.quiet_us = kKnob;
  in.stream_outstanding = true;
  require(bus_idle_sleep_us(in) == 0,
          "knob mode must not nap with an outstanding stream collective");
}

DGPP_TEST(bus_idle_knob_requires_the_idle_count) {
  auto in = base();
  in.graph_recorded = true;
  in.idle_sleep_us = kKnob;
  in.quiet_us = kKnob;
  in.idle_count_passed = false;
  require(bus_idle_sleep_us(in) == 0,
          "knob mode must not nap before the spin-iteration threshold");
}

DGPP_TEST(bus_idle_knob_zero_falls_back_to_legacy) {
  auto in = base();
  in.graph_recorded = true;
  in.quiet_us = 0;
  in.idle_sleep_us = 0;
  require(bus_idle_sleep_us(in) == 0,
          "knob 0 must mean legacy hot-spin, never a nap");
}

}  // namespace
