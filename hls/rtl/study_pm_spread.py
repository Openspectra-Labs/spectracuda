"""Path-metric spread of a soft-decision K=7 Viterbi with 4-bit signed LLRs.

VALIDATION of the analytic bound, not the proof (the proof is in
viterbi_dec_ovl.v's header):

    branch-metric spread per step  B = 2 * (|l0| + |l1|) <= 28   (L = 7)
    path-metric spread, t >= 6     <= 6 * B = 168
    early steps, init offset X     <= X + 168 = 337   (X = 169)

Exact (unbounded) integer ACS, the correlation metric bm = s0*l0 + s1*l1
(s = +1 for an expected 0, -1 for an expected 1; LLR > 0 means 0), exactly
the arithmetic the RTL soft decoder implements. LLRs: max-log per bit for
Gray QPSK / 16-QAM / 64-QAM over AWGN, clipped to +/-c and quantized to
the signed 4-bit range [-7, 7]. Also an adversarial stream (random +/-7)
that maximises |l0| + |l1| every step.

    python study_pm_spread.py
"""
import numpy as np

K, M = 7, 6
NS = 64
L = 7
X = 6 * 2 * 2 * L + 1      # 169: > the most a path can gain in 6 steps


def tables():
    o1 = np.array([bin(s & 0o171).count("1") & 1 for s in range(NS)])
    o2 = np.array([bin(s & 0o133).count("1") & 1 for s in range(NS)])
    return o1, o2


def encode(bits):
    st, out = 0, []
    for b in np.concatenate([bits, np.zeros(M, int)]):
        reg = ((st << 1) | int(b)) & 0x7F
        out += [bin(reg & 0o171).count("1") & 1, bin(reg & 0o133).count("1") & 1]
        st = reg & 63
    return np.array(out)


def constellation(bps):
    if bps == 2:
        pts = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2)
        labels = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        return pts, labels
    m = 1 << (bps // 2)
    lv = np.arange(m) * 2 - (m - 1)
    gray = np.array([i ^ (i >> 1) for i in range(m)])
    pts, labels = [], []
    for i in range(m):
        for q in range(m):
            pts.append(lv[i] + 1j * lv[q])
            bi = [(gray[i] >> (bps // 2 - 1 - k)) & 1 for k in range(bps // 2)]
            bq = [(gray[q] >> (bps // 2 - 1 - k)) & 1 for k in range(bps // 2)]
            labels.append(bi + bq)
    pts = np.array(pts)
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2)), np.array(labels)


def llrs(coded, bps, snr_db, clip, rng):
    pts, lab = constellation(bps)
    n = -(-len(coded) // bps)
    c = np.concatenate([coded, np.zeros(n * bps - len(coded), int)]).reshape(n, bps)
    idx = np.array([np.where((lab == row).all(1))[0][0] for row in c])
    sigma2 = 10 ** (-snr_db / 10)
    y = pts[idx] + rng.normal(0, np.sqrt(sigma2 / 2), n) + 1j * rng.normal(0, np.sqrt(sigma2 / 2), n)
    d = np.abs(y[:, None] - pts[None, :]) ** 2 / sigma2
    out = np.empty((n, bps))
    for k in range(bps):
        out[:, k] = d[:, lab[:, k] == 1].min(1) - d[:, lab[:, k] == 0].min(1)   # >0 => 0
    q = np.clip(np.round(np.clip(out, -clip, clip) / clip * L), -L, L).astype(int)
    return q.reshape(-1)[:len(coded)]


def acs_spread(l):
    """Exact ACS over LLR pairs; returns (max spread, max |PM|)."""
    o1, o2 = tables()
    s1, s2 = 1 - 2 * o1, 1 - 2 * o2              # +1 expected 0, -1 expected 1
    pred_a = np.arange(NS) >> 1
    pred_b = pred_a + 32
    pm = np.full(NS, X, dtype=np.int64); pm[0] = 0
    early, steady, amax = int(pm.max() - pm.min()), 0, int(np.abs(pm).max())
    for t in range(len(l) // 2):
        bm_a = s1 * l[2 * t] + s2 * l[2 * t + 1]
        ca, cb = pm[pred_a] + bm_a, pm[pred_b] - bm_a     # bm_b = -bm_a
        pm = np.where(cb < ca, cb, ca)
        sp = int(pm.max() - pm.min())
        if t + 1 >= M: steady = max(steady, sp)
        else:          early = max(early, sp)
        amax = max(amax, int(np.abs(pm).max()))
    return early, steady, amax


def main():
    rng = np.random.default_rng(1)
    n_info = 6000
    print(f"bound: early {X + 6 * 4 * L}, steady {6 * 4 * L}; X = {X}")
    worst_e = worst_s = 0
    for bps, name in ((2, "QPSK"), (4, "16-QAM"), (6, "64-QAM")):
        for clip in (2.0, 3.0, 6.0):
            row = []
            for snr in (0, 4, 8, 12, 16, 20, 25, 30):
                coded = encode(rng.integers(0, 2, n_info))
                e, s, a = acs_spread(llrs(coded, bps, snr, clip, rng))
                worst_e, worst_s = max(worst_e, e), max(worst_s, s)
                row.append(f"{snr:>2}dB:{s:>3}")
            print(f"{name:7s} clip {clip:>3}: steady-state max spread  " + "  ".join(row))
    for trial in range(3):
        e, s, a = acs_spread(rng.choice([-L, L], 2 * (n_info + M)))
        worst_e, worst_s = max(worst_e, e), max(worst_s, s)
        print(f"adversarial random +/-7 #{trial}: early {e}, steady {s}, max |PM| {a} over {n_info + M} steps")
    print(f"\nMAX MEASURED SPREAD: early (t<6) {worst_e} [bound 337], steady (t>=6) {worst_s} [bound 168]")


if __name__ == "__main__":
    main()
