#include "correct/convolutional/hexagon/convolutional.h"

/* Same reasoning as neon/encode.c and sse/encode.c: libcorrect's
 * encoder is a simple shift-register convolution, not the Viterbi
 * add-compare-select bottleneck -- no separate speedup claim here (and
 * arguably even less reason to bother on Hexagon specifically: putting
 * the tiny, branchy encoder on the DSP would mean paying a FastRPC
 * round trip for work cheaper than the round trip itself -- see
 * docs/hexagon-fec-offload-plan.md's batching-cost discussion). Kept
 * for interface symmetry with correct_convolutional_hexagon_decode*()
 * only; the actual FastRPC interface (fec_hexagon.idl) is not expected
 * to expose this at all -- encode should stay on the CPU path
 * (ConvolutionalCode.encode()'s existing xp-vectorized loop), per
 * fec/viterbi.py's own encode()/decode() split. */

size_t correct_convolutional_hexagon_encode_len(correct_convolutional_hexagon *conv, size_t msg_len) {
    return correct_convolutional_encode_len(&conv->base_conv, msg_len);
}

size_t correct_convolutional_hexagon_encode(correct_convolutional_hexagon *conv, const uint8_t *msg,
                                            size_t msg_len, uint8_t *encoded) {
    return correct_convolutional_encode(&conv->base_conv, msg, msg_len, encoded);
}
