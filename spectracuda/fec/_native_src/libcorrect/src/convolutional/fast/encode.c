#include "correct/convolutional/fast/convolutional.h"

/* Same reasoning as neon/encode.c and sse/encode.c: the encoder is a
 * simple shift-register convolution, not the Viterbi bottleneck -- no
 * separate speed claim, just the identical portable encoder. */

size_t correct_convolutional_fast_encode_len(correct_convolutional_fast *conv, size_t msg_len) {
    return correct_convolutional_encode_len(&conv->base_conv, msg_len);
}

size_t correct_convolutional_fast_encode(correct_convolutional_fast *conv, const uint8_t *msg,
                                         size_t msg_len, uint8_t *encoded) {
    return correct_convolutional_encode(&conv->base_conv, msg, msg_len, encoded);
}
