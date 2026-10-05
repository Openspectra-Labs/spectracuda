// C-sim testbench for sc_sync: feeds the golden IQ capture produced by
// hls/gen/golden.py (spectracuda's own pipeline) and compares against
// what spectracuda's SchmidlCoxSync returned for that same input.
//
// Pass criteria are deliberately asymmetric:
//   start_index  must match EXACTLY -- it is an integer sample offset,
//                and "close" is not a thing. A one-sample error puts the
//                FFT window in the wrong place.
//   metric       is compared with a tolerance, since it is a ratio of
//                float accumulations whose summation ORDER differs
//                between the sliding-window form and numpy's cumsum.
//
// The metric tolerance differs by build, on purpose (SC_METRIC_TOL):
//   float build  1e-6  -- tight, because its only job is to prove the
//                        ALGORITHM was ported correctly. Anything worse
//                        than round-off means a real transcription bug.
//   fixed build  1e-3  -- the metric is not a measurement, it is fed to
//                        a detection threshold (spectracuda's
//                        DEFAULT_SYNC_THRESHOLD, order 0.5). Precision
//                        beyond ~1e-3 changes no decision the block
//                        makes. Holding it to float parity would be
//                        measuring the wrong thing.
// start_index has no tolerance in either build.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include "../src/sc_sync.h"

static std::string golden_path(const char *name) {
    const char *env = getenv("HLS_GOLDEN_DIR");
    std::string dir = env ? env : "../golden";
    return dir + "/" + name;
}

// Minimal scalar pull from the JSON -- avoids a JSON library dependency
// in a testbench that only needs four numbers.
static bool json_number(const std::string &txt, const char *key, double &out) {
    std::string pat = std::string("\"") + key + "\"";
    size_t k = txt.find(pat);
    if (k == std::string::npos) return false;
    k = txt.find(':', k + pat.size());
    if (k == std::string::npos) return false;
    out = atof(txt.c_str() + k + 1);
    return true;
}

int main() {
    // ---- load meta ----
    FILE *mf = fopen(golden_path("sc_sync_meta.json").c_str(), "r");
    if (!mf) { printf("FAIL: cannot open sc_sync_meta.json -- run "
                      "`python -m hls.gen.golden` first\n"); return 1; }
    std::string meta;
    { char b[4096]; size_t n; while ((n = fread(b, 1, sizeof b, mf)) > 0) meta.append(b, n); }
    fclose(mf);

    double d_L = 0, d_n = 0, d_start = 0, d_metric = 0, d_true = 0;
    if (!json_number(meta, "L", d_L) ||
        !json_number(meta, "n_samples", d_n) ||
        !json_number(meta, "expected_start_index", d_start) ||
        !json_number(meta, "expected_metric", d_metric) ||
        !json_number(meta, "expected_true_start", d_true)) {
        printf("FAIL: meta JSON missing keys\n"); return 1;
    }
    const int L = (int)d_L, n_samples = (int)d_n;
    const int expect_d = (int)d_start, true_start = (int)d_true;

    // ---- load IQ ----
    FILE *f = fopen(golden_path("sc_sync_rx.txt").c_str(), "r");
    if (!f) { printf("FAIL: cannot open sc_sync_rx.txt\n"); return 1; }
    hls::stream<sc_iq_t> iq_in("iq_in");
    int count = 0;
    double vi, vq;
    while (fscanf(f, "%lf %lf", &vi, &vq) == 2) {
        sc_iq_t s;
        s.i = (sc_sample_t)vi;
        s.q = (sc_sample_t)vq;
        iq_in.write(s);
        count++;
    }
    fclose(f);
    if (count != n_samples) {
        printf("FAIL: read %d samples, meta says %d\n", count, n_samples);
        return 1;
    }

    // ---- run ----
    int got_d = -1;
    sc_metric_t got_metric = 0;
    sc_sync(iq_in, n_samples, L, got_d, got_metric);
    const double got_metric_d = (double)got_metric;

    // ---- compare ----
#ifndef SC_METRIC_TOL
  #define SC_METRIC_TOL 1e-6
#endif
    const double tol = SC_METRIC_TOL;
    const double err = fabs(got_metric_d - d_metric) /
                       (fabs(d_metric) > 0 ? fabs(d_metric) : 1.0);
    printf("samples=%d  L=%d  true_frame_start=%d\n", n_samples, L, true_start);
    printf("  start_index : hls=%d  python=%d  %s\n",
           got_d, expect_d, got_d == expect_d ? "MATCH" : "MISMATCH");
    printf("  metric      : hls=%.9g  python=%.9g  rel_err=%.3g  %s\n",
           got_metric_d, d_metric, err, err <= tol ? "MATCH" : "MISMATCH");

    int fails = 0;
    if (got_d != expect_d) fails++;
    if (!(err <= tol))     fails++;
    printf(fails ? "\nFAIL (%d check%s)\n" : "\nPASS\n",
           fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
