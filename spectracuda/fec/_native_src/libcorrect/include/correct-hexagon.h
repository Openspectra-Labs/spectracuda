#ifndef CORRECT_HEXAGON_H
#define CORRECT_HEXAGON_H
#include <correct.h>

/* Hexagon-HVX-accelerated add-compare-select (ACS) inner loop for
 * conv_v27's Viterbi decode -- the QCS6490/Radxa-Q6A counterpart to
 * correct-neon.h, NOT part of upstream libcorrect (no such target
 * exists there -- see that header's own comment). Written for
 * spectracuda specifically, reusing upstream's own portable
 * pair_lookup_t/history_buffer/bit_reader-writer machinery unchanged,
 * mirroring correct-neon.h's shape exactly -- see
 * src/convolutional/hexagon/decode.c's own comment for the ACS kernel
 * design and, importantly, its verification status.
 *
 * *** NOT YET COMPILED, NOT YET RUN. *** Written without access to the
 * Hexagon SDK (hexagon-clang) or any Hexagon/Q6A hardware -- see
 * docs/hexagon-fec-offload-plan.md for the full story. Treat this
 * header (and its .c counterparts) as a design sketch to build FROM,
 * not a working port to trust as-is. Every intrinsic name used in
 * decode.c must be checked against reference/qc6490/docs/
 * hexagon_v68_hvx_programmers_reference.pdf (QCS6490's actual HVX
 * generation, per its product brief -- v73 is a later chip's HVX
 * revision, use it only to cross-check, not as the primary source)
 * before this is ever fed to hexagon-clang for real.
 *
 * Also NOTE (unlike correct-neon.h's *_create/_destroy/_encode/_decode
 * shape, which all run in the SAME process as their caller): this
 * struct and these functions are meant to run ON THE HEXAGON DSP, as
 * the implementation behind a FastRPC interface (see
 * spectracuda/fec/_native_src/hexagon/fec_hexagon.idl), never linked
 * directly into the CPU-side Python process the way correct_convolutional_neon
 * is. correct_convolutional_hexagon_decode() below is written to be
 * called from the generated FastRPC skel's dispatch code, batched
 * across n_batch rows in ONE call -- see fec/_native_hexagon.py's
 * module docstring, "WHY BATCHING IS NOT OPTIONAL HERE", for why the
 * per-row correct_convolutional_neon_decode() shape would be wrong to
 * copy here.
 */

struct correct_convolutional_hexagon;
typedef struct correct_convolutional_hexagon correct_convolutional_hexagon;

correct_convolutional_hexagon *correct_convolutional_hexagon_create(
    size_t rate, size_t order, const correct_convolutional_polynomial_t *poly);

void correct_convolutional_hexagon_destroy(correct_convolutional_hexagon *conv);

size_t correct_convolutional_hexagon_encode_len(correct_convolutional_hexagon *conv, size_t msg_len);

size_t correct_convolutional_hexagon_encode(correct_convolutional_hexagon *conv, const uint8_t *msg,
                                            size_t msg_len, uint8_t *encoded);

ssize_t correct_convolutional_hexagon_decode(correct_convolutional_hexagon *conv, const uint8_t *encoded,
                                             size_t num_encoded_bits, uint8_t *msg);

/* Batched entry point -- the one this project's FastRPC interface
 * (fec_hexagon.idl) should actually expose, per this header's own
 * comment on why per-row calls are wrong here. Decodes n_batch
 * independent codewords, each num_encoded_bits_per_row/msg_bits_per_row
 * long, in ONE call -- looping over n_batch happens INSIDE this
 * function (DSP-side), not by the FastRPC caller making n_batch calls.
 * Returns 0 on success, negative on failure (mirrors the single-row
 * function's ssize_t-error convention, collapsed to a single pass/fail
 * since a per-row error count would need its own out-parameter this
 * sketch hasn't designed yet -- TODO once real error-injection testing
 * on hardware shows whether that granularity is actually needed).
 */
int correct_convolutional_hexagon_decode_batch(correct_convolutional_hexagon *conv,
                                               const uint8_t *encoded, size_t n_batch,
                                               size_t num_encoded_bits_per_row,
                                               uint8_t *msg, size_t msg_bits_per_row);

#endif
