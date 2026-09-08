#include "correct/convolutional/hexagon/convolutional.h"

/* Hexagon HVX add-compare-select (ACS) inner loop for conv_v27's
 * Viterbi decode -- the QCS6490/Radxa-Q6A counterpart to
 * neon/decode.c's convolutional_neon_decode_inner(). Ported from THAT
 * file's algorithm (not translated from sse/decode.c -- same reasoning
 * neon/decode.c's own comment gives).
 *
 * *** DESIGN SKETCH, NOT VERIFIED. *** Written with no access to the
 * Hexagon SDK (hexagon-clang, hexagon-sim) or real Hexagon/Q6A
 * hardware -- see docs/hexagon-fec-offload-plan.md. Every HVX
 * intrinsic used below follows Qualcomm's documented
 * Q6_<ResultType>_<opname>_<ArgTypes> naming convention (confirmed
 * against the two real intrinsics used in
 * reference/qc6490/Hexagon_DSP_programming/06-hvx-example/04-hvx-example.c,
 * Q6_V_vzero() and Q6_Vh_vdmpy_VubRb()) but the SPECIFIC names below
 * (vadd/vcmp.gtu/vmux/vsplat) have NOT been individually confirmed
 * against reference/qc6490/docs/hexagon_v68_hvx_programmers_reference.pdf
 * -- do that, one intrinsic at a time, before this is ever fed to
 * hexagon-clang for real. Treat every `HVX_Vector` line as "probably
 * close, verify before trusting."
 *
 * DELIBERATE SCOPE, first cut: this mirrors the FIRST (slower)
 * NEON attempt's shape -- gather into flat scalar arrays, vector-only
 * for the add/compare/select math, then a scalar copy-out loop back
 * into write_errors[]/history[] -- NOT the tightened second NEON
 * attempt's vld1-direct-load / vld2-deinterleave / vst2-interleaved-
 * store version (see neon/decode.c's own comment for that
 * history). That tightening needs an HVX deinterleave/interleave
 * intrinsic (Hexagon calls these "deal"/"shuffle" -- Q6_W_vdeal_VVR /
 * Q6_W_vshuff_VVR by the same naming convention, unconfirmed here) that
 * this sketch deliberately avoids depending on for its FIRST pass, to
 * keep the unverified-intrinsic surface as small as possible: correctness
 * first (this project's own two-gate rule, see fec/_native.py's module
 * docstring), only THEN chase the memory-traffic tightening NEON's own
 * history shows is where the real win lives -- expect this first cut to
 * likely UNDERPERFORM its theoretical ceiling for exactly the reason
 * the first NEON attempt did, until that follow-up pass happens.
 *
 * Width: one HVX vector is 128 bytes in "128B mode" (see
 * reference/qc6490/Hexagon_DSP_programming/06-hvx-example/README.txt's
 * own `-mhvx-length=128B` build line) = 64 lanes of uint16_t (distance_t)
 * -- exactly 8x NEON's uint16x8_t width, processing 64 base_offsets
 * (128 successor states) per HVX sequence instead of NEON's 8
 * base_offsets (16 states) -- IN THEORY. See the load-bearing problem
 * this sketch ran into, below.
 *
 * *** OPEN PROBLEM, found while writing this sketch (not yet solved):
 * *** for THIS PROJECT'S ACTUAL CODE (K=7, N_STATES=64), highbase (the
 * number of valid base_offset values per trellis half, per time step)
 * is 32 -- narrower than one 64-lane HVX vector. The loop below is
 * bounded by `base + HEXAGON_ACS_LANES <= highbase`, so at K=7 that
 * condition (0 + 64 <= 32) is ALWAYS FALSE: the HVX main loop never
 * runs, every trellis step falls through entirely to the scalar tail,
 * and this file, as literally written, accelerates NOTHING for this
 * project's real conv_v27 code. (An earlier draft of this file bounded
 * the loop by `high <= num_iter` instead, copying NEON's exact
 * arithmetic without checking it against HEXAGON_ACS_LANES > highbase
 * -- that version silently read pair_lookup.keys[] out of bounds for
 * bo in [32,64) and corrupted every successor state past the first 64;
 * fixed here by bounding on `base`/`highbase` directly instead, which
 * is correct regardless of how LANES compares to highbase, but
 * correct-and-inert is not the goal.)
 *
 * NEON never hit this because 8 <= 32 (highbase) with margin; HVX's
 * width being LARGER than this code's own per-codeword half-trellis is
 * a real mismatch, not a detail to shrug off. Two ways forward, neither
 * attempted in this sketch:
 *   (a) batch 2+ INDEPENDENT codewords' 32-wide half-trellises into one
 *       64-lane HVX vector, interleaving their ACS math lane-wise --
 *       doable, but a real restructuring of libcorrect's inherently
 *       single-codeword-at-a-time correct_convolutional struct (its
 *       error_buffer/history_buffer/pair_lookup are all sized and
 *       indexed for ONE trellis), not a mechanical port.
 *   (b) don't port libcorrect's C struct at all -- instead port
 *       fec/viterbi.py's OWN numpy ACS step, which already vectorizes
 *       over (n_batch, 64 states) jointly every trellis time step (see
 *       that file's own module docstring), straight to an HVX kernel
 *       operating on the flattened n_batch*64 array per time step. For
 *       any real n_batch >= 2 this trivially exceeds 64 lanes, so HVX
 *       stays fed regardless of K -- and it matches this codebase's own
 *       batch-shape-first design (block.py's docstring) instead of
 *       fighting it. Recommended direction -- see
 *       docs/hexagon-fec-offload-plan.md.
 * This file is kept anyway as the more direct NEON-mirroring attempt,
 * for the record, and because (a) may still be worth a real
 * measurement once (b) has been tried and either wins or doesn't.
 */

#define HEXAGON_ACS_LANES 64 /* 128 bytes / sizeof(uint16_t) -- verify against the actual HVX_VLEN for QCS6490's HVX rev (see this file's own top comment) before trusting this constant */

static inline void convolutional_hexagon_decode_inner(correct_convolutional *conv, unsigned int sets,
                                                       const uint8_t *soft) {
    shift_register_t highbit = 1 << (conv->order - 1);
    for (unsigned int i = conv->order - 1; i < (sets - conv->order + 1); i++) {
        distance_t *distances = conv->distances;
        if (soft) {
            if (conv->soft_measurement == CORRECT_SOFT_LINEAR) {
                for (unsigned int j = 0; j < 1 << (conv->rate); j++) {
                    distances[j] = metric_soft_distance_linear(j, soft + i * conv->rate, conv->rate);
                }
            } else {
                for (unsigned int j = 0; j < 1 << (conv->rate); j++) {
                    distances[j] = metric_soft_distance_quadratic(j, soft + i * conv->rate, conv->rate);
                }
            }
        } else {
            unsigned int out = bit_reader_read(conv->bit_reader, conv->rate);
            for (unsigned int j = 0; j < 1 << (conv->rate); j++) {
                distances[j] = metric_distance(j, out);
            }
        }
        pair_lookup_t pair_lookup = conv->pair_lookup;
        pair_lookup_fill_distance(pair_lookup, distances);

        unsigned int num_iter = highbit << 1;
        const distance_t *read_errors = conv->errors->read_errors;
        distance_t *write_errors = conv->errors->write_errors;
        uint8_t *history = history_buffer_get_slice(conv->history_buffer);

        shift_register_t highbase = highbit >> 1;
        shift_register_t low = 0, high = highbit, base = 0;

        // -- Main loop: HEXAGON_ACS_LANES base_offsets (2x as many
        // successor states) per HVX sequence. Bounded on
        // base+LANES<=highbase (the actual valid base_offset count),
        // NOT high<=num_iter the way neon/decode.c's loop is (that
        // bound only happens to be equivalent to this one when
        // LANES<=highbase divides evenly, which NEON's 8<=32 satisfies
        // but HVX's 64<=32 does NOT -- see this file's own "OPEN
        // PROBLEM" comment above for why bounding on num_iter here
        // would silently read pair_lookup.keys[] out of range). This
        // form is correct for any LANES/highbase relationship, but at
        // this project's actual K=7 (highbase=32 < LANES=64) it
        // correctly never executes -- everything falls through to the
        // tail loop below. --
        for (; base + HEXAGON_ACS_LANES <= highbase;
             low += 2 * HEXAGON_ACS_LANES, high += 2 * HEXAGON_ACS_LANES, base += HEXAGON_ACS_LANES) {
            // Gather step: genuinely data-dependent (pair_lookup.keys[...]
            // -> pair_lookup.distances[key]), stays scalar -- same as
            // NEON, no ARM/Hexagon has a general gather instruction for
            // this. Unlike neon/decode.c's SECOND attempt (which packs
            // lo/hi into one uint32_t concat value per lane and
            // de-interleaves with vld2q_u16), this first Hexagon cut
            // splits lo/hi directly in the scalar loop -- same shape as
            // neon/decode.c's own TAIL fallback -- to avoid depending on
            // an unverified HVX deinterleave intrinsic on this first
            // pass (see this file's own top comment).
            uint16_t low_lo[HEXAGON_ACS_LANES], low_hi[HEXAGON_ACS_LANES];
            uint16_t high_lo[HEXAGON_ACS_LANES], high_hi[HEXAGON_ACS_LANES];
            uint16_t low_past[HEXAGON_ACS_LANES], high_past[HEXAGON_ACS_LANES];
            for (unsigned int bo = 0; bo < HEXAGON_ACS_LANES; bo++) {
                distance_pair_key_t low_key = pair_lookup.keys[base + bo];
                distance_pair_key_t high_key = pair_lookup.keys[highbase + base + bo];
                distance_pair_t low_concat = pair_lookup.distances[low_key];
                distance_pair_t high_concat = pair_lookup.distances[high_key];
                low_lo[bo] = (uint16_t)(low_concat & 0xffff);
                low_hi[bo] = (uint16_t)(low_concat >> 16);
                high_lo[bo] = (uint16_t)(high_concat & 0xffff);
                high_hi[bo] = (uint16_t)(high_concat >> 16);
                low_past[bo] = read_errors[base + bo];
                high_past[bo] = read_errors[highbase + base + bo];
            }

            // VERIFY: aligned 128-byte vector load off a plain pointer,
            // following the `*(HVX_Vector*)ptr` pattern shown in
            // 04-hvx-example.c. The scratch arrays above are NOT
            // guaranteed 128-byte aligned as plain stack arrays --
            // either force alignment (e.g. a compiler-specific aligned
            // attribute) or use an unaligned-load intrinsic (Hexagon:
            // vmemu-style) instead of a raw pointer deref. Left as a
            // plain deref here ONLY to keep this sketch's shape close
            // to the confirmed example; fix this before compiling.
            HVX_Vector low_lo_v = *(HVX_Vector *)low_lo;
            HVX_Vector low_hi_v = *(HVX_Vector *)low_hi;
            HVX_Vector high_lo_v = *(HVX_Vector *)high_lo;
            HVX_Vector high_hi_v = *(HVX_Vector *)high_hi;
            HVX_Vector low_past_v = *(HVX_Vector *)low_past;
            HVX_Vector high_past_v = *(HVX_Vector *)high_past;

            // -- "successor" stream (even successors: low, low+2, ...) --
            // VERIFY: Q6_Vh_vadd_VhVh / Q6_Q_vcmp_gtu_VhVh / Q6_V_vmux_QVV
            // against the ISA reference -- named by convention, not
            // individually confirmed (see this file's top comment).
            HVX_Vector low_error_v = Q6_Vh_vadd_VhVh(low_lo_v, low_past_v);
            HVX_Vector high_error_v = Q6_Vh_vadd_VhVh(high_lo_v, high_past_v);
            HVX_VectorPred choose_high = Q6_Q_vcmp_gtu_VhVh(low_error_v, high_error_v); // low > high -> take high
            HVX_Vector error_v = Q6_V_vmux_QVV(choose_high, high_error_v, low_error_v);
            HVX_Vector hist_v = Q6_V_vmux_QVV(choose_high, Q6_Vh_vsplat_R(1), Q6_V_vzero());

            // -- "plus one" stream (odd successors: low+1, low+3, ...) --
            HVX_Vector low_error2_v = Q6_Vh_vadd_VhVh(low_hi_v, low_past_v);
            HVX_Vector high_error2_v = Q6_Vh_vadd_VhVh(high_hi_v, high_past_v);
            HVX_VectorPred choose_high2 = Q6_Q_vcmp_gtu_VhVh(low_error2_v, high_error2_v);
            HVX_Vector error2_v = Q6_V_vmux_QVV(choose_high2, high_error2_v, low_error2_v);
            HVX_Vector hist2_v = Q6_V_vmux_QVV(choose_high2, Q6_Vh_vsplat_R(1), Q6_V_vzero());

            // Scalar copy-out -- deliberately, see this file's own
            // "DELIBERATE SCOPE" comment above: this is the same
            // memory-round-trip shape the FIRST NEON attempt used and
            // that its SECOND attempt tightened away with vst2q_u16.
            // Doing the equivalent tightening here needs the
            // interleaved-store counterpart to the deinterleaved-load
            // intrinsic this sketch is deliberately avoiding on pass
            // one -- a documented follow-up, not an oversight.
            uint16_t error_arr[HEXAGON_ACS_LANES], hist_arr[HEXAGON_ACS_LANES];
            uint16_t error2_arr[HEXAGON_ACS_LANES], hist2_arr[HEXAGON_ACS_LANES];
            *(HVX_Vector *)error_arr = error_v;
            *(HVX_Vector *)hist_arr = hist_v;
            *(HVX_Vector *)error2_arr = error2_v;
            *(HVX_Vector *)hist2_arr = hist2_v;
            for (unsigned int bo = 0; bo < HEXAGON_ACS_LANES; bo++) {
                write_errors[low + 2 * bo] = error_arr[bo];
                history[low + 2 * bo] = (uint8_t)hist_arr[bo];
                write_errors[low + 2 * bo + 1] = error2_arr[bo];
                history[low + 2 * bo + 1] = (uint8_t)hist2_arr[bo];
            }
        }

        // -- Tail: any base_offsets left after the main loop above --
        // for this project's actual K=7 code, that's ALL of them (0..31,
        // see this file's own "OPEN PROBLEM" comment) -- plain portable
        // scalar ACS, one base_offset at a time, identical to
        // ../decode.c's own inner loop body. --
        for (; base < highbase; low += 2, high += 2, base += 1) {
            distance_pair_key_t low_key = pair_lookup.keys[base];
            distance_pair_key_t high_key = pair_lookup.keys[highbase + base];
            distance_pair_t low_concat = pair_lookup.distances[low_key];
            distance_pair_t high_concat = pair_lookup.distances[high_key];
            distance_t low_past = read_errors[base];
            distance_t high_past = read_errors[highbase + base];

            distance_t low_error = (distance_t)((low_concat & 0xffff) + low_past);
            distance_t high_error = (distance_t)((high_concat & 0xffff) + high_past);
            write_errors[low] = low_error <= high_error ? low_error : high_error;
            history[low] = low_error <= high_error ? 0 : 1;

            distance_t low_error2 = (distance_t)((low_concat >> 16) + low_past);
            distance_t high_error2 = (distance_t)((high_concat >> 16) + high_past);
            write_errors[low + 1] = low_error2 <= high_error2 ? low_error2 : high_error2;
            history[low + 1] = low_error2 <= high_error2 ? 0 : 1;
        }

        history_buffer_process(conv->history_buffer, write_errors, conv->bit_writer);
        error_buffer_swap(conv->errors);
    }
}

static ssize_t _convolutional_hexagon_decode(correct_convolutional_hexagon *hex_conv,
                                             size_t num_encoded_bits, size_t num_encoded_bytes,
                                             uint8_t *msg, const soft_t *soft_encoded) {
    correct_convolutional *conv = &hex_conv->base_conv;
    if (!conv->has_init_decode) {
        uint64_t max_error_per_input = conv->rate * soft_max;
        unsigned int renormalize_interval = distance_max / max_error_per_input;
        _convolutional_decode_init(conv, 5 * conv->order, 15 * conv->order, renormalize_interval);
    }

    size_t sets = num_encoded_bits / conv->rate;
    size_t decoded_len_bytes = num_encoded_bytes;
    bit_writer_reconfigure(conv->bit_writer, msg, decoded_len_bytes);

    error_buffer_reset(conv->errors);
    history_buffer_reset(conv->history_buffer);

    // Warmup and tail: UNCHANGED portable functions, same as
    // neon/decode.c's own choice -- only the bulk "inner" ACS phase is
    // HVX-accelerated here.
    convolutional_decode_warmup(conv, sets, soft_encoded);
    convolutional_hexagon_decode_inner(conv, sets, soft_encoded);
    convolutional_decode_tail(conv, sets, soft_encoded);

    history_buffer_flush(conv->history_buffer, conv->bit_writer);

    return bit_writer_length(conv->bit_writer);
}

ssize_t correct_convolutional_hexagon_decode(correct_convolutional_hexagon *conv, const uint8_t *encoded,
                                             size_t num_encoded_bits, uint8_t *msg) {
    if (num_encoded_bits % conv->base_conv.rate) {
        return -1;
    }
    size_t num_encoded_bytes =
        (num_encoded_bits % 8) ? (num_encoded_bits / 8 + 1) : (num_encoded_bits / 8);
    bit_reader_reconfigure(conv->base_conv.bit_reader, encoded, num_encoded_bytes);

    return _convolutional_hexagon_decode(conv, num_encoded_bits, num_encoded_bytes, msg, NULL);
}

/* The actual FastRPC-facing entry point (see correct-hexagon.h's own
 * comment, and fec/_native_hexagon.py's module docstring, "WHY
 * BATCHING IS NOT OPTIONAL HERE"): decodes n_batch independent
 * codewords in ONE call, looping over the batch HERE, on the DSP side,
 * so the CPU<->DSP FastRPC boundary is crossed exactly once per
 * Mac/Ofdm-level decode() call regardless of n_batch -- not once per
 * row the way NativeConvolutional's in-process ctypes loop
 * (fec/_native.py) correctly does, since that loop's per-call cost is
 * negligible in-process but would NOT be negligible per FastRPC call.
 *
 * Row layout: `encoded` is n_batch rows of num_encoded_bits_per_row
 * bits each, densely packed (row i starts at byte
 * i * ceil(num_encoded_bits_per_row/8)); `msg` likewise for
 * msg_bits_per_row-bit decoded rows. Matches the flattened layout
 * fec/_native_hexagon.py's (not-yet-written) marshaling code should
 * produce from a (n_batch, k) numpy array via a single np.packbits()
 * call, rather than n_batch separate small buffers -- keeps the
 * FastRPC payload as one contiguous buffer, one copy.
 */
int correct_convolutional_hexagon_decode_batch(correct_convolutional_hexagon *conv,
                                               const uint8_t *encoded, size_t n_batch,
                                               size_t num_encoded_bits_per_row,
                                               uint8_t *msg, size_t msg_bits_per_row) {
    size_t encoded_row_bytes =
        (num_encoded_bits_per_row % 8) ? (num_encoded_bits_per_row / 8 + 1) : (num_encoded_bits_per_row / 8);
    size_t msg_row_bytes = (msg_bits_per_row % 8) ? (msg_bits_per_row / 8 + 1) : (msg_bits_per_row / 8);

    for (size_t b = 0; b < n_batch; b++) {
        ssize_t written = correct_convolutional_hexagon_decode(
            conv, encoded + b * encoded_row_bytes, num_encoded_bits_per_row, msg + b * msg_row_bytes);
        if (written < 0) {
            return -1;
        }
    }
    return 0;
}
