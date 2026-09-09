#include "correct/convolutional/fast/convolutional.h"
#include <string.h>

/* See correct-fast.h for the design. This file is bit-exact with the
 * portable ../decode.c by construction: the warmup and tail below are
 * literal ports of convolutional_decode_warmup()/_tail(), the traceback
 * mirrors history_buffer.c's schedule, and only convolutional_decode_inner()
 * is replaced -- by a branch-metric-broadcast butterfly that computes the
 * exact same add-compare-select with the exact same tie rule
 * (low_error <= high_error -> low predecessor wins). */

/* ---- 16 x uint8 vector helpers ----------------------------------------
 * SSE2 (x86_64 baseline) and NEON (AArch64 baseline) need no extra compile
 * flags; the generic GCC/clang vector-extension fallback keeps the file
 * building anywhere else (unmeasured there). */
#if defined(__SSE2__)
#include <emmintrin.h>
typedef __m128i v16;
static inline v16 v_load(const uint8_t *p) { return _mm_loadu_si128((const __m128i *)p); }
static inline void v_store(uint8_t *p, v16 a) { _mm_storeu_si128((__m128i *)p, a); }
static inline v16 v_add(v16 a, v16 b) { return _mm_add_epi8(a, b); }
static inline v16 v_min(v16 a, v16 b) { return _mm_min_epu8(a, b); }
/* 0xFF where a < b, unsigned: a < b  <=>  max(a, b) != a */
static inline v16 v_lt(v16 a, v16 b) {
    return _mm_xor_si128(_mm_cmpeq_epi8(_mm_max_epu8(a, b), a), _mm_set1_epi8(-1));
}
static inline v16 v_zip_lo(v16 a, v16 b) { return _mm_unpacklo_epi8(a, b); }
static inline v16 v_zip_hi(v16 a, v16 b) { return _mm_unpackhi_epi8(a, b); }
static inline v16 v_splat(uint8_t x) { return _mm_set1_epi8((char)x); }
static inline v16 v_sub(v16 a, v16 b) { return _mm_sub_epi8(a, b); }
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
typedef uint8x16_t v16;
static inline v16 v_load(const uint8_t *p) { return vld1q_u8(p); }
static inline void v_store(uint8_t *p, v16 a) { vst1q_u8(p, a); }
static inline v16 v_add(v16 a, v16 b) { return vaddq_u8(a, b); }
static inline v16 v_min(v16 a, v16 b) { return vminq_u8(a, b); }
static inline v16 v_lt(v16 a, v16 b) { return vcltq_u8(a, b); }
static inline v16 v_zip_lo(v16 a, v16 b) { return vzip1q_u8(a, b); }
static inline v16 v_zip_hi(v16 a, v16 b) { return vzip2q_u8(a, b); }
static inline v16 v_splat(uint8_t x) { return vdupq_n_u8(x); }
static inline v16 v_sub(v16 a, v16 b) { return vsubq_u8(a, b); }
#else
typedef uint8_t v16 __attribute__((vector_size(16)));
typedef signed char v16s __attribute__((vector_size(16)));
static inline v16 v_load(const uint8_t *p) { v16 a; memcpy(&a, p, 16); return a; }
static inline void v_store(uint8_t *p, v16 a) { memcpy(p, &a, 16); }
static inline v16 v_add(v16 a, v16 b) { return a + b; }
static inline v16 v_lt(v16 a, v16 b) { return (v16)(v16s)(a < b); }
static inline v16 v_min(v16 a, v16 b) { v16 m = v_lt(a, b); return (a & m) | (b & ~m); }
#if defined(__clang__)
static inline v16 v_zip_lo(v16 a, v16 b) {
    return __builtin_shufflevector(a, b, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
}
static inline v16 v_zip_hi(v16 a, v16 b) {
    return __builtin_shufflevector(a, b, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
}
#else
static inline v16 v_zip_lo(v16 a, v16 b) {
    const v16 m = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};
    return __builtin_shuffle(a, b, m);
}
static inline v16 v_zip_hi(v16 a, v16 b) {
    const v16 m = {8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
    return __builtin_shuffle(a, b, m);
}
#endif
static inline v16 v_splat(uint8_t x) { v16 a = {x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x}; return a; }
static inline v16 v_sub(v16 a, v16 b) { return a - b; }
#endif

/* Subtract the minimum path metric every this many inner steps. Metrics
 * grow by at most 2 per step, and the survivor spread is bounded (<= 12),
 * so the largest metric stays <= 12 + 2*32 + (12 more from the tail) --
 * far from wrapping an 8-bit lane. Subtracting a common constant changes
 * no comparison and no argmin, so the schedule is invisible to the
 * output. */
#define FAST_RENORM_INTERVAL 32

/* The received 2-bit symbol for step i, in the portable bit_reader's
 * convention: first stream bit -> bit 0, second -> bit 1 (bit_reader_read()
 * reverses the bits it reads). Both bits of a pair always sit in one byte
 * since pairs start at even bit offsets. */
static inline unsigned int read_symbol(const uint8_t *encoded, size_t i) {
    unsigned int two = (encoded[i >> 2] >> (6 - 2 * (i & 3))) & 3;
    return ((two & 1) << 1) | (two >> 1);
}

/* history_buffer_search(): first state (index order, given stride) with
 * the least metric. */
static unsigned int fast_search(const uint8_t *metrics, unsigned int stride) {
    unsigned int best = 0, least = 0xFFFF;
    for (unsigned int s = 0; s < FAST_NUM_STATES; s += stride) {
        if (metrics[s] < least) {
            least = metrics[s];
            best = s;
        }
    }
    return best;
}

/* history_buffer_traceback(): walk back min_tb slices without emitting,
 * then emit every remaining queued slice's path bit, oldest first. */
static void fast_traceback(correct_convolutional_fast *conv, unsigned int bestpath, unsigned int min_tb) {
    unsigned int index = conv->index;
    unsigned int fi = 0;
    for (unsigned int j = 0; j < min_tb; j++) {
        index = index ? index - 1 : FAST_CAP - 1;
        uint8_t h = conv->history[index * FAST_NUM_STATES + bestpath];
        bestpath = (bestpath | (h ? 64u : 0u)) >> 1;
    }
    for (unsigned int j = min_tb; j < conv->len; j++) {
        index = index ? index - 1 : FAST_CAP - 1;
        uint8_t h = conv->history[index * FAST_NUM_STATES + bestpath];
        bestpath = (bestpath | (h ? 64u : 0u)) >> 1;
        conv->fetched[fi++] = h ? 1 : 0;
    }
    bit_writer_write_bitlist_reversed(conv->base_conv.bit_writer, conv->fetched, fi);
    conv->len -= fi;
}

/* history_buffer_process_skip(): advance the ring; traceback once 140
 * slices are queued. (The portable build also renormalizes here on its
 * own 128-step schedule; that never affects decisions, so it is not
 * mirrored.) */
static inline void fast_process(correct_convolutional_fast *conv, const uint8_t *metrics, unsigned int stride) {
    if (++conv->index == FAST_CAP) {
        conv->index = 0;
    }
    if (++conv->len == FAST_CAP) {
        fast_traceback(conv, fast_search(metrics, stride), FAST_MIN_TRACEBACK);
    }
}

ssize_t correct_convolutional_fast_decode(correct_convolutional_fast *conv, const uint8_t *encoded,
                                          size_t num_encoded_bits, uint8_t *msg) {
    correct_convolutional *base = &conv->base_conv;
    if (num_encoded_bits % 2) {
        return -1;
    }
    size_t sets = num_encoded_bits / 2;
    if (sets < 12) {
        /* Fewer sets than warmup + tail: the portable loops' unsigned
         * bounds wrap here; no caller of ours ever produces this (the
         * zero tail alone is 6 sets and _native.py pads further). */
        return -1;
    }
    size_t num_encoded_bytes = (num_encoded_bits % 8) ? (num_encoded_bits / 8 + 1) : (num_encoded_bits / 8);
    /* decoded_len_bytes = num_encoded_bytes, exactly as the portable
     * _convolutional_decode() does (its own "XXX fix this"): the writer
     * only ever emits whole bytes, which is the withheld-trailing-bits
     * quirk _native.py's _DECODE_PAD_PAIRS works around. */
    bit_writer_reconfigure(base->bit_writer, msg, num_encoded_bytes);
    conv->index = 0;
    conv->len = 0;

    const unsigned int *table = base->table;
    uint8_t *read = conv->metrics[0];
    uint8_t *write = conv->metrics[1];
    memset(read, 0, FAST_NUM_STATES);   /* error_buffer_reset(): both buffers zeroed */
    memset(write, 0, FAST_NUM_STATES);

    /* ---- warmup: port of convolutional_decode_warmup() (no history) ----
     * INCLUDING an upstream quirk that is part of the reference behavior:
     * error_buffer_reset() leaves read=errors[0]/write=errors[1] with
     * index=0, and error_buffer_swap() assigns read=errors[index] BEFORE
     * toggling index -- so after warmup step 0 the read buffer is still
     * the zeroed errors[0], and step 1 overwrites step 0's results.
     * Net effect: the FIRST received symbol pair contributes nothing to
     * the portable (and SSE/NEON, which share the file) decoder. Found by
     * a step-by-step dump of the real decoder against a scalar port of
     * its source (2026-09-09); reproduced here by skipping step 0, since
     * bit-exactness with the portable decoder is this kernel's contract.
     * A decoder that USED that symbol would be marginally stronger on the
     * first few bits but would no longer be bit-exact with libcorrect. */
    for (unsigned int i = 1; i < 6; i++) {
        unsigned int out = read_symbol(encoded, i);
        for (unsigned int j = 0; j < (1u << (i + 1)); j++) {
            write[j] = (uint8_t)(popcount(table[j] ^ out) + read[j >> 1]);
        }
        uint8_t *t = read; read = write; write = t;
    }

    /* ---- inner: vectorized ACS for steps 6 .. sets-7 ---- */
    v16 m0 = v_load(read), m1 = v_load(read + 16), m2 = v_load(read + 32), m3 = v_load(read + 48);
    unsigned int since_renorm = 0;
    for (size_t i = 6; i < sets - 6; i++) {
        unsigned int out = read_symbol(encoded, i);
        const uint8_t *bm = &conv->bm[out][0][0];   /* [kind][32] */

        /* half 0: predecessors 0..15 (m0) and 32..47 (m2) -> successors 0..31 */
        v16 a0 = v_add(m0, v_load(bm + 0 * 32));
        v16 a1 = v_add(m0, v_load(bm + 1 * 32));
        v16 b0 = v_add(m2, v_load(bm + 2 * 32));
        v16 b1 = v_add(m2, v_load(bm + 3 * 32));
        v16 n0 = v_min(a0, b0), d0 = v_lt(b0, a0);   /* high wins only if strictly smaller */
        v16 n1 = v_min(a1, b1), d1 = v_lt(b1, a1);
        v16 s0 = v_zip_lo(n0, n1), s1 = v_zip_hi(n0, n1);
        v16 h0 = v_zip_lo(d0, d1), h1 = v_zip_hi(d0, d1);

        /* half 1: predecessors 16..31 (m1) and 48..63 (m3) -> successors 32..63 */
        a0 = v_add(m1, v_load(bm + 0 * 32 + 16));
        a1 = v_add(m1, v_load(bm + 1 * 32 + 16));
        b0 = v_add(m3, v_load(bm + 2 * 32 + 16));
        b1 = v_add(m3, v_load(bm + 3 * 32 + 16));
        n0 = v_min(a0, b0); d0 = v_lt(b0, a0);
        n1 = v_min(a1, b1); d1 = v_lt(b1, a1);
        v16 s2 = v_zip_lo(n0, n1), s3 = v_zip_hi(n0, n1);
        v16 h2 = v_zip_lo(d0, d1), h3 = v_zip_hi(d0, d1);

        uint8_t *slice = conv->history + (size_t)conv->index * FAST_NUM_STATES;
        v_store(slice, h0); v_store(slice + 16, h1); v_store(slice + 32, h2); v_store(slice + 48, h3);
        m0 = s0; m1 = s1; m2 = s2; m3 = s3;

        if (++conv->index == FAST_CAP) {
            conv->index = 0;
        }
        if (++conv->len == FAST_CAP) {
            v_store(write, m0); v_store(write + 16, m1); v_store(write + 32, m2); v_store(write + 48, m3);
            fast_traceback(conv, fast_search(write, 1), FAST_MIN_TRACEBACK);
        }
        if (++since_renorm == FAST_RENORM_INTERVAL) {
            since_renorm = 0;
            v_store(write, v_min(v_min(m0, m1), v_min(m2, m3)));
            uint8_t least = write[0];
            for (unsigned int k = 1; k < 16; k++) {
                if (write[k] < least) least = write[k];
            }
            v16 sub = v_splat(least);
            m0 = v_sub(m0, sub); m1 = v_sub(m1, sub); m2 = v_sub(m2, sub); m3 = v_sub(m3, sub);
        }
    }
    v_store(read, m0); v_store(read + 16, m1); v_store(read + 32, m2); v_store(read + 48, m3);

    /* ---- tail: port of convolutional_decode_tail() (only 0-successors;
     * note the tie rule flips: high predecessor wins ties here) ---- */
    for (size_t i = sets - 6; i < sets; i++) {
        unsigned int out = read_symbol(encoded, i);
        uint8_t *slice = conv->history + (size_t)conv->index * FAST_NUM_STATES;
        unsigned int skip = 1u << (7 - (unsigned int)(sets - i));
        unsigned int base_skip = skip >> 1;
        for (unsigned int low = 0, high = 64, b = 0; high < 128; low += skip, high += skip, b += base_skip) {
            unsigned int low_error = popcount(table[low] ^ out) + read[b];
            unsigned int high_error = popcount(table[high] ^ out) + read[32 + b];
            if (low_error < high_error) {
                write[low] = (uint8_t)low_error;
                slice[low] = 0;
            } else {
                write[low] = (uint8_t)high_error;
                slice[low] = 1;
            }
        }
        fast_process(conv, write, skip);
        uint8_t *t = read; read = write; write = t;
    }

    /* history_buffer_flush(): traceback from state 0, emit everything left */
    fast_traceback(conv, 0, 0);
    return (ssize_t)bit_writer_length(base->bit_writer);
}
