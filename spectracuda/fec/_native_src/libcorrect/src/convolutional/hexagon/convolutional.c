#include "correct/convolutional/hexagon/convolutional.h"

/* Mechanical mirror of neon/convolutional.c -- create/destroy just
 * wrap the portable _correct_convolutional_init()/_teardown(), same as
 * every other accelerated variant in this tree. Nothing HVX-specific
 * lives here; see hexagon/decode.c for the actual kernel. */

correct_convolutional_hexagon *correct_convolutional_hexagon_create(size_t rate,
                                                                     size_t order,
                                                                     const polynomial_t *poly) {
    correct_convolutional_hexagon *conv = malloc(sizeof(correct_convolutional_hexagon));
    correct_convolutional *init_conv = _correct_convolutional_init(&conv->base_conv, rate, order, poly);
    if (!init_conv) {
        free(conv);
        conv = NULL;
    }
    return conv;
}

void correct_convolutional_hexagon_destroy(correct_convolutional_hexagon *conv) {
    _correct_convolutional_teardown(&conv->base_conv);
    free(conv);
}
