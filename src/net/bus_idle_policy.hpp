#pragma once

#include <cstdint>

namespace dgpp::net {

// Idle-nap policy for the bus engine thread (src/net/collective_bus.cpp).
//
// The engine is a dedicated poller: its poll loop IS the handshake latency,
// and once a decode graph exists it normally hot-spins even between
// requests — one core at ~100% for the life of the process, which shows up
// as idle power vs a server that sleeps between requests. The
// DGPP_BUS_ENGINE_IDLE_SLEEP_US knob trades request-start latency for that
// power: set to N>0, the engine may sleep N us once it has been truly quiet
// for at least N us (no armed replay window, no outstanding stream
// collective, and enough consecutive no-work iterations).
//
// The quiet-grace requirement is what keeps the nap out of an active decode:
// gaps between decode windows are microseconds, so a nap gated on
// quiet >= N (N >= ~1 ms in practice) can never fire mid-decode; it fires
// only between requests. The first request after an idle stretch pays up to
// ~N us (plus wake-up) before its first collective is posted — the accepted
// trade-off. The one thing that must never happen is a nap between decode
// windows: even a 50 us sleep makes the pinned core look idle to CFS, the
// scheduler parks another runnable thread there, and the wake-up waits out
// that thread's slice — 7-10 ms, PREEMPT_NONE/HZ=250 — which every other
// rank then waits on at the next window's first generation. With the knob
// unset (idle_sleep_us == 0) the legacy behavior is preserved exactly: the
// engine naps only before any decode graph exists.
struct BusIdlePolicyInputs {
  bool idle_count_passed;   // enough consecutive no-work iterations
  bool window_live;         // a replay window is armed
  bool stream_outstanding;  // an issued stream collective is un-walked
  bool graph_recorded;      // a decode graph era has been recorded
  uint64_t quiet_us;        // continuous no-work time so far
  uint64_t idle_sleep_us;   // DGPP_BUS_ENGINE_IDLE_SLEEP_US; 0 = legacy
  uint64_t legacy_sleep_us; // nap used in legacy mode (kEngineIdleSleepUs)
};

// Returns the sleep duration in microseconds (0 = keep hot-spinning).
inline uint64_t bus_idle_sleep_us(const BusIdlePolicyInputs& in) {
  if (in.window_live || in.stream_outstanding) return 0;
  if (!in.idle_count_passed) return 0;
  if (in.idle_sleep_us == 0) {
    // Legacy: the nap is a pre-serving power courtesy only. Once a decode
    // graph exists the thread hot-spins — the poll loop IS the handshake.
    if (in.graph_recorded) return 0;
    return in.legacy_sleep_us;
  }
  if (in.quiet_us < in.idle_sleep_us) return 0;
  return in.idle_sleep_us;
}

}  // namespace dgpp::net
