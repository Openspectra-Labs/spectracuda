#ifndef CORRECT_FAST_H
#define CORRECT_FAST_H
#include <correct.h>

/* "Fast" Viterbi decode for libcorrect's rate-1/2, K=7 convolutional
 * code (the ONLY code this variant supports -- create() returns NULL for
 * anything else) -- NOT part of upstream libcorrect. Written for
 * spectracuda specifically (MIT, like correct-neon.h; see that header's
 * own licensing note), as a different decoder FORMULATION rather than a
 * port of the portable/SSE/NEON inner loops:
 *
 *   - Branch metrics are precomputed per possible received 2-bit symbol
 *     as per-predecessor constant vectors, so the per-step work is a
 *     fixed sequence of vector add / min / compare / interleave with NO
 *     data-dependent table gather (the gather is what keeps the
 *     portable pair_lookup design partly scalar on NEON -- see
 *     neon/decode.c's own comment).
 *   - Path metrics are 8-bit (16 states per 128-bit vector) instead of
 *     the portable build's 16-bit, renormalized by subtracting the
 *     minimum every few steps. The K=7 survivor-metric spread is bounded
 *     (<= 12 for hard decisions), so this never wraps; and since every
 *     decision is a comparison of two sums that share the same subtracted
 *     constants, decisions -- and therefore the decoded bits -- are
 *     identical to the 16-bit build's.
 *   - The 6-step warmup and 6-step tail are literal scalar ports of the
 *     portable decode.c (only ~130 state updates in total), which is
 *     what keeps the phase-specific tie-breaking (low wins ties in the
 *     inner loop, high wins ties in the tail) and the no-history-during-
 *     warmup behavior bit-exact without any SIMD special-casing.
 *   - The traceback schedule (140-slice ring, first traceback after 140
 *     steps emitting the oldest 105 bits, then every 105, flush from
 *     state 0) and the bit_writer output packing are reproduced exactly
 *     -- including upstream's "withheld trailing bits" quirk that
 *     spectracuda/fec/_native.py works around with _DECODE_PAD_PAIRS, so
 *     that workaround applies to this variant unchanged.
 *
 * Bit-exactness against correct_convolutional_decode() is the contract
 * (tests/test_fec_fast_viterbi.py), on clean AND heavily corrupted
 * input -- corrupted input is what exercises the tie-breaks and the
 * truncated-traceback paths.
 *
 * Same shape as correct-neon.h: these instances must not be used with
 * the non-fast functions and vice versa. */

struct correct_convolutional_fast;
typedef struct correct_convolutional_fast correct_convolutional_fast;

correct_convolutional_fast *correct_convolutional_fast_create(
    size_t rate, size_t order, const correct_convolutional_polynomial_t *poly);

void correct_convolutional_fast_destroy(correct_convolutional_fast *conv);

size_t correct_convolutional_fast_encode_len(correct_convolutional_fast *conv, size_t msg_len);

size_t correct_convolutional_fast_encode(correct_convolutional_fast *conv, const uint8_t *msg,
                                         size_t msg_len, uint8_t *encoded);

ssize_t correct_convolutional_fast_decode(correct_convolutional_fast *conv, const uint8_t *encoded,
                                          size_t num_encoded_bits, uint8_t *msg);

#endif
