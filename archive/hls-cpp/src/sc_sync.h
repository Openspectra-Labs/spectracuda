// ============================================================
// sc_sync.h -- Schmidl & Cox preamble detector for Vitis HLS
//
// Port of spectracuda's SchmidlCoxSync. The algorithm is taken from
// spectracuda/sync/_numba_schmidl_cox.py, NOT from schmidl_cox.py's
// numpy path -- the numba kernel is already the single-pass sliding-
// window form (O(1) per sample, no prefix-sum arrays), which is exactly
// what fabric wants. The numpy path's ten full-array passes are a
// batch-machine idiom with no hardware equivalent.
//
// Why this replaces ~/work/ofdm-hls's sync_detect: that block is a
// Zadoff-Chu matched filter and needs a 200-entry complex ROM of the
// known template. Schmidl-Cox correlates the signal against a delayed
// copy of ITSELF -- it uses the preamble's two-identical-halves
// structure and never its content -- so the ROM disappears entirely.
//
//   P(d) = sum_{m<L} conj(r[d+m]) * r[d+m+L]
//   R(d) = ( sum_{m<L} |r[d+m]|^2 + sum_{m<L} |r[d+m+L]|^2 ) / 2
//   metric(d) = |P(d)|^2 / R(d)^2
//
// Note R(d) is the SYMMETRIC (both-halves) energy, not the 1997 paper's
// second-half-only version -- spectracuda deviates deliberately there
// (a straddling window can drive the one-sided denominator toward zero
// and produce unbounded spurious peaks; measured at ~77 for a metric
// bounded at 1.0). Reproduced here because bit-comparability with the
// golden model is the contract.
//
// STAGE 1 SCOPE -- whole-buffer argmax, matching Python exactly.
// Real fabric cannot wait for a whole buffer; it needs threshold-then-
// peak-hold on a free-running stream (what ofdm-hls's sync_detect does).
// That is a SEPARATE change with its own failure modes, so it is not
// mixed into this one. Prove the arithmetic first, then the state
// machine.
// ============================================================
#pragma once

#include "hls_stream.h"

// ---- numeric configuration ---------------------------------
// SC_SYNC_USE_FIXED selects the fabric types. Left off by default so
// C-sim proves the ALGORITHM against Python without quantization in the
// way; turning it on is how word lengths get swept, against this float
// build as the reference rather than against an end-to-end metric that
// channel noise dominates.
//
// WORD LENGTHS ARE NOT ARBITRARY -- the first synthesis of this block
// came back at 41% of the XC7A50T's LUTs, 58% of its DSPs, and failed
// timing at 18.3 ns against a 10 ns target, almost entirely because of
// choices made here and in the .cpp for C-sim convenience:
//
//   * `double` anywhere in the datapath inferred a 64-bit FP multiplier
//     (11 DSP) and a 64-bit FP divider. Removed: the fixed build now has
//     no `double` at all, and the metric leaves as a fixed-point type.
//   * Wide accumulators squared directly gave 40x40 and 48x48
//     multipliers (19 DSP). Fixed by NARROWING BEFORE SQUARING -- the
//     DSP48E1 is 25x18, so an 18-bit operand costs one DSP slice and a
//     40-bit one costs five to nine.
//
// The precision this discards is provably irrelevant: the word-length
// sweep (`make sweep`) shows start_index stays exact against Python down
// to 8-bit samples, so the metric comparison has bits to spare.
#ifdef SC_SYNC_USE_FIXED
  #include "ap_fixed.h"
  #ifndef SC_SAMPLE_W
    #define SC_SAMPLE_W 16
  #endif
  typedef ap_fixed<SC_SAMPLE_W, 1> sc_sample_t;
  // Running sums over L terms. Each term is bounded by 2 (|a_i*b_i| +
  // |a_q*b_q| with both operands < 1), so L=128 gives |acc| <= 256:
  // 9 magnitude bits plus sign.
  // AP_RND, not the ap_fixed default. Vitis defaults to AP_TRN, which
  // truncates toward -inf and so biases EVERY accumulation by up to half
  // an LSB in the same direction. This is a running sum updated once per
  // sample -- ~4400 times per frame -- so a systematic bias accumulates
  // where random rounding error would not. Measured: AP_TRN gave a 0.9%
  // metric error, ~9x the tolerance, while start_index stayed exact.
  // (spectracuda/sim/fixedpoint.py's docstring says exactly this; the
  // first version of this file did not apply it.)
  typedef ap_fixed<28, 11, AP_RND> sc_acc_t;
  // Truncated copy taken JUST before squaring, so the squarer is an
  // 18x18 that maps to a single DSP48E1.
  // Width swept by `make narrow-sweep`. 18 was the first guess and it
  // FAILED (3/12 cases): the accumulators peak at 256 in theory but sit
  // near 10 in practice, so 11 integer bits leave only 7 fractional --
  // about 10 bits of relative precision on the value that matters, which
  // the squaring and division then compound. Sizing a narrow type from
  // the worst case rather than the typical case is the trap here.
  #ifndef SC_NARROW_W
    #define SC_NARROW_W 24
  #endif
  typedef ap_fixed<SC_NARROW_W, 11> sc_narrow_t;
  // Squares: |p|^2 and r^2 reach ~2*256^2 = 131072 -> 18 bits + sign.
  typedef ap_fixed<36, 20> sc_pow_t;
  // The metric itself is a ratio in [0, ~1].
  typedef ap_fixed<24, 3>  sc_metric_t;
#else
  typedef double sc_sample_t;
  typedef double sc_acc_t;
  typedef double sc_narrow_t;
  typedef double sc_pow_t;
  typedef double sc_metric_t;
#endif

struct sc_iq_t {
    sc_sample_t i;
    sc_sample_t q;
};

// Delay-line depth comes from the GENERATED geometry, not a local
// #define. hls/gen/emit.py writes SC_HALF_L = fft_size/2 from the same
// Ofdm object that produced the golden vectors, so the block and the
// reference model cannot drift apart.
//
// SC_HALF_L sizes the ring buffer at compile time -- it has to, it is an
// array bound. `L` stays a runtime argument so one build can be exercised
// at several lengths by the testbench; it must satisfy L <= SC_MAX_L.
#include "generated/ofdm_params.h"

#ifndef SC_MAX_L
  #define SC_MAX_L SC_HALF_L
#endif

// ---- top-level ---------------------------------------------
//   iq_in       : n_samples complex samples, streamed in order
//   n_samples   : how many to consume
//   L           : half the FFT size (preamble half-length), <= SC_MAX_L
//   best_d      : argmax candidate offset  -> spectracuda's start_index
//   best_metric : the peak metric value    -> spectracuda's metric
void sc_sync(
    hls::stream<sc_iq_t> &iq_in,
    int                   n_samples,
    int                   L,
    int                  &best_d,
    sc_metric_t          &best_metric
);
