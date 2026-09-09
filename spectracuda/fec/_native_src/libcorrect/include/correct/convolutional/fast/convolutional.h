#ifndef CORRECT_CONVOLUTIONAL_FAST_H
#define CORRECT_CONVOLUTIONAL_FAST_H
#include "correct/convolutional/convolutional.h"
#include "correct-fast.h"

/* Fixed by the only supported code (rate 1/2, K=7): 2^(7-1) = 64
 * trellis states; the history ring and traceback depths mirror the
 * portable _convolutional_decode()'s own 5*order / 15*order. */
#define FAST_NUM_STATES 64
#define FAST_MIN_TRACEBACK 35
#define FAST_TRACEBACK_GROUP 105
#define FAST_CAP (FAST_MIN_TRACEBACK + FAST_TRACEBACK_GROUP)

struct correct_convolutional_fast {
    correct_convolutional base_conv;  /* table + bit_writer reused; decode state below is our own */

    /* Precomputed branch metrics, [received 2-bit symbol][kind][predecessor 0..31]:
     * kind 0/1 = low predecessor p (oldest bit 0), new input bit 0/1
     *          -> popcount(table[2p + bit] ^ out)
     * kind 2/3 = high predecessor p+32 (oldest bit 1), new input bit 0/1
     *          -> popcount(table[64 + 2p + bit] ^ out)
     * Laid out so lanes 0..15 / 16..31 load straight into the two
     * 16-lane halves of the ACS. */
    uint8_t bm[4][4][32] __attribute__((aligned(16)));

    uint8_t metrics[2][FAST_NUM_STATES] __attribute__((aligned(16)));  /* ping-pong, used by the scalar phases */
    uint8_t *history;                 /* FAST_CAP slices x FAST_NUM_STATES decision bytes */
    uint8_t fetched[FAST_CAP];
    unsigned int index;               /* next slice to write */
    unsigned int len;                 /* slices queued for traceback */
};

#endif
