#include "correct.h"

/* Batched entry points over upstream's per-block Reed-Solomon encode/
 * decode -- NOT part of upstream libcorrect (spectracuda addition, MIT).
 *
 * Why: a 64032-bit PDU is ~36 RS(255,223) blocks, and spectracuda's
 * NativeReedSolomon wrapper used to make one Python->ctypes round trip
 * per block (plus a numpy concatenate and a frombuffer copy each time).
 * Measured on the x86 dev box (2026-09-09): 26.7 us/block as wired vs
 * 11.3 us/block for the bare C calls -- 58% of the Reed-Solomon RX line
 * was Python glue. These loops move that per-block iteration into C so
 * the Python side makes exactly ONE call per PDU with pre-laid-out
 * contiguous (n_blocks x block_length) buffers. The per-block work is the
 * unchanged upstream correct_reed_solomon_encode()/decode(), so the
 * output is byte-identical to the per-block path
 * (tests/test_fec_reed_solomon_batch.py).
 *
 * Layout contract (all row-major, contiguous):
 *   encode: msgs    [n_blocks][msg_length]   -> encoded [n_blocks][block_length]
 *   decode: encoded [n_blocks][block_length] -> msgs_out[n_blocks][msg_length],
 *           n_written_out[n_blocks] = each block's own return value from
 *           correct_reed_solomon_decode() (<= 0 means that block was
 *           uncorrectable -- the caller decides what to do, as before).
 * Returns the number of blocks processed, or -1 on an encode failure
 * (encode never fails for a well-formed msg_length; decode failures are
 * reported per block, not as a return value, so one bad block does not
 * hide the others' results). */

ssize_t correct_reed_solomon_encode_batch(correct_reed_solomon *rs, const uint8_t *msgs,
                                          size_t n_blocks, size_t msg_length, uint8_t *encoded,
                                          size_t block_length) {
    for (size_t i = 0; i < n_blocks; i++) {
        ssize_t r = correct_reed_solomon_encode(rs, msgs + i * msg_length, msg_length,
                                                encoded + i * block_length);
        if (r < 0) {
            return -1;
        }
    }
    return (ssize_t)n_blocks;
}

ssize_t correct_reed_solomon_decode_batch(correct_reed_solomon *rs, const uint8_t *encoded,
                                          size_t n_blocks, size_t block_length, uint8_t *msgs_out,
                                          size_t msg_length, ssize_t *n_written_out) {
    for (size_t i = 0; i < n_blocks; i++) {
        n_written_out[i] = correct_reed_solomon_decode(rs, encoded + i * block_length, block_length,
                                                       msgs_out + i * msg_length);
    }
    return (ssize_t)n_blocks;
}
