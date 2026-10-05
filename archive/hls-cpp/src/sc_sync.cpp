#include "sc_sync.h"

// Ring buffer of the most recent 2L+1 samples. A candidate at offset d
// needs r[d-1], r[d-1+L] and r[d-1+2L] simultaneously, so the block must
// hold 2L+1 samples of history -- one BRAM at L=128, versus the 200-entry
// complex ROM plus multiply-accumulate chain a matched filter needs.
//
// Indices are advanced by COMPARE-AND-WRAP, never by `%`. The first
// synthesis of this file used `idx % span`; span is 2L+1 = 257, not a
// power of two, so each `%` became a 32-bit integer remainder core
// (srem_32ns_32ns_32_36_seq, ~394 FF + 238 LUT each, two instances, and
// a long combinational path that cost the block its timing closure).
// A wrapping counter is free.
static const sc_acc_t SC_HALF = 0.5;

static inline void bump(int &idx, int span) {
#pragma HLS INLINE
    idx++;
    if (idx == span) idx = 0;
}

void sc_sync(
    hls::stream<sc_iq_t> &iq_in,
    int                   n_samples,
    int                   L,
    int                  &best_d,
    sc_metric_t          &best_metric
) {
#pragma HLS INTERFACE axis port=iq_in
#pragma HLS INTERFACE s_axilite port=n_samples
#pragma HLS INTERFACE s_axilite port=L
#pragma HLS INTERFACE s_axilite port=best_d
#pragma HLS INTERFACE s_axilite port=best_metric
#pragma HLS INTERFACE s_axilite port=return

    static sc_sample_t buf_i[2 * SC_MAX_L + 1];
    static sc_sample_t buf_q[2 * SC_MAX_L + 1];
#pragma HLS BIND_STORAGE variable=buf_i type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=buf_q type=ram_2p impl=bram

    const int span = 2 * L + 1;

    // ---- prime the ring with the first 2L samples ----
    // Nothing can be evaluated until both correlation windows are full.
    int w = 0;
PRIME:
    for (int n = 0; n < 2 * L; n++) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256
        sc_iq_t s = iq_in.read();
        buf_i[w] = s.i;
        buf_q[w] = s.q;
        bump(w, span);
    }

    // ---- d = 0, computed directly (nothing to slide from yet) ----
    sc_acc_t p_re = 0, p_im = 0, r1 = 0, r2 = 0;
    int ia = 0, ib = L;
INIT:
    for (int m = 0; m < L; m++) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
        sc_sample_t a_i = buf_i[ia], a_q = buf_q[ia];
        sc_sample_t b_i = buf_i[ib], b_q = buf_q[ib];
        // conj(a) * b
        p_re += a_i * b_i + a_q * b_q;
        p_im += a_i * b_q - a_q * b_i;
        r1   += a_i * a_i + a_q * a_q;
        r2   += b_i * b_i + b_q * b_q;
        bump(ia, span);
        bump(ib, span);
    }

    // metric = |P|^2 / R^2, R = (r1 + r2) / 2.
    //
    // Everything is NARROWED to sc_narrow_t before squaring. An 18-bit
    // operand is one DSP48E1 (25x18); the 28-bit accumulator squared
    // directly is five to nine. The discarded bits do not matter -- the
    // word-length sweep shows start_index is exact against Python even
    // with 8-bit samples.
    //
    // One division per candidate, into a narrow result, then compare
    // narrow metrics. The first version compared by cross-multiplication
    // to "avoid a divider" and got two 48x48 multipliers (9 DSP each)
    // for its trouble -- more expensive than the divider it dodged.
    int    bd = 0;
    sc_metric_t bm = 0;

    // Slide: one sample in, one candidate evaluated.
    const int n_candidates = n_samples - 2 * L + 1;
    int i_out = 0, i_mid = L, i_in = 2 * L;
SLIDE:
    for (int d = 0; d < n_candidates; d++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=65536
#pragma HLS PIPELINE II=4
        if (d > 0) {
            sc_iq_t s = iq_in.read();
            buf_i[i_in] = s.i;
            buf_q[i_in] = s.q;

            sc_sample_t o_i = buf_i[i_out], o_q = buf_q[i_out];
            sc_sample_t m_i = buf_i[i_mid], m_q = buf_q[i_mid];
            sc_sample_t n_i = s.i,          n_q = s.q;

            // a_in = conj(mid)*in ; a_out = conj(out)*mid
            p_re += (m_i * n_i + m_q * n_q) - (o_i * m_i + o_q * m_q);
            p_im += (m_i * n_q - m_q * n_i) - (o_i * m_q - o_q * m_i);

            sc_acc_t e_out = o_i * o_i + o_q * o_q;
            sc_acc_t e_mid = m_i * m_i + m_q * m_q;
            sc_acc_t e_in  = n_i * n_i + n_q * n_q;
            r1 += e_mid - e_out;
            r2 += e_in  - e_mid;

            bump(i_out, span);
            bump(i_mid, span);
            bump(i_in,  span);
        }

        sc_narrow_t pn_re = (sc_narrow_t)p_re;
        sc_narrow_t pn_im = (sc_narrow_t)p_im;
        // Typed constant, not `>> 1` (invalid on double in the float
        // build) and not `* 0.5` (a bare double literal risks promoting
        // the whole expression). HLS folds a constant 0.5 multiply into
        // a shift, so this costs nothing in fabric.
        sc_narrow_t rn    = (sc_narrow_t)((r1 + r2) * SC_HALF);

        sc_pow_t num = (sc_pow_t)(pn_re * pn_re) + (sc_pow_t)(pn_im * pn_im);
        sc_pow_t den = (sc_pow_t)(rn * rn);

        // Python guards with R^2 + 1e-12. That epsilon is far below this
        // type's LSB, so it is expressed as an explicit zero test -- same
        // intent (a silent window can never win), no dead arithmetic.
        sc_metric_t metric = 0;
        if (den > (sc_pow_t)0) {
            metric = (sc_metric_t)(num / den);
        }

        if (d == 0 || metric > bm) {
            bm = metric;
            bd = d;
        }
    }

    best_d      = bd;
    best_metric = bm;
}
