#include "correct/convolutional/fast/convolutional.h"

correct_convolutional_fast *correct_convolutional_fast_create(size_t rate, size_t order,
                                                               const polynomial_t *poly) {
    /* This formulation hard-codes the rate-1/2, K=7 trellis shape (64
     * states, 4 possible received symbols, 2 branches per state). Refuse
     * anything else up front rather than decode it wrong. */
    if (rate != 2 || order != 7) {
        return NULL;
    }
    correct_convolutional_fast *conv = calloc(1, sizeof(correct_convolutional_fast));
    if (!_correct_convolutional_init(&conv->base_conv, rate, order, poly)) {
        free(conv);
        return NULL;
    }
    const unsigned int *table = conv->base_conv.table;
    for (unsigned int out = 0; out < 4; out++) {
        for (unsigned int p = 0; p < 32; p++) {
            conv->bm[out][0][p] = (uint8_t)popcount(table[2 * p] ^ out);
            conv->bm[out][1][p] = (uint8_t)popcount(table[2 * p + 1] ^ out);
            conv->bm[out][2][p] = (uint8_t)popcount(table[64 + 2 * p] ^ out);
            conv->bm[out][3][p] = (uint8_t)popcount(table[64 + 2 * p + 1] ^ out);
        }
    }
    conv->history = calloc((size_t)FAST_CAP * FAST_NUM_STATES, 1);
    conv->index = 0;
    conv->len = 0;
    return conv;
}

void correct_convolutional_fast_destroy(correct_convolutional_fast *conv) {
    free(conv->history);
    /* base_conv never went through _convolutional_decode_init() (this
     * variant owns its own decode state), so teardown only releases the
     * table and the bit reader/writer. */
    _correct_convolutional_teardown(&conv->base_conv);
    free(conv);
}
