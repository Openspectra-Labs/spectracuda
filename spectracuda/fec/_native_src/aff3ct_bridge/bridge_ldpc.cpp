// bridge_ldpc.cpp -- minimal standalone LDPC decode server, driving AFF3CT's
// Decoder_LDPC module directly via its bootstrap-style buffer API
// (decode_siho()), bypassing AFF3CT's own Source/Channel/Modem/BFER
// simulation loop entirely. This is the "isolate the hard problem" step in
// bridging AFF3CT's LDPC decoder into spectracuda's real pipeline: prove a
// standalone driver can construct the decoder from spectracuda's own
// exported .qc matrix and decode LLRs correctly, before building any
// Python-side persistent-process plumbing around it.
//
// Protocol (stdin/stdout, binary, persistent process -- one process serves
// many frames, avoiding AFF3CT's own per-invocation startup cost that the
// subprocess-per-call benchmark in examples/benchmark_x86_stages_ldpc_aff3ct.py
// pays):
//   startup args: argv[1] = path to .qc matrix file (from
//                 spectracuda/fec/_native_src/aff3ct_bridge/qc_export.py)
//   per call, on stdin:  N_cw float32 LLRs, native-endian, no framing
//   per call, on stdout: 1 int32 status (0 = converged to a valid
//                        codeword per AFF3CT's own syndrome check, same
//                        as spectracuda's own decode()'s "fail loud"
//                        convention -- nonzero = did not converge, the
//                        K bits that follow should not be trusted) then
//                        K int32 decoded bits (0/1), native-endian
//   loop until stdin EOF (Python side closes the pipe to shut it down)
//
// Decoder settings match examples/benchmark_x86_stages_ldpc_aff3ct.py's
// verified CLI flags exactly: BP_FLOODING / NMS / norm=0.75 / ite=50 /
// syndrome-based early termination ON (AFF3CT's default behavior, and the
// main real-world advantage over spectracuda's own fec/ldpc.py decode(),
// which always runs the full max_iterations -- see that file's docstring).

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <aff3ct.hpp>

using namespace aff3ct;

namespace
{
bool read_exact(int fd, void* buf, size_t n)
{
    auto* p = static_cast<unsigned char*>(buf);
    size_t got = 0;
    while (got < n)
    {
        ssize_t r = ::read(fd, p + got, n - got);
        if (r == 0) return false; // clean EOF -- normal shutdown
        if (r < 0)
        {
            if (errno == EINTR) continue;
            std::perror("bridge_ldpc: read");
            std::exit(1);
        }
        got += static_cast<size_t>(r);
    }
    return true;
}

void write_exact(int fd, const void* buf, size_t n)
{
    auto* p = static_cast<const unsigned char*>(buf);
    size_t sent = 0;
    while (sent < n)
    {
        ssize_t w = ::write(fd, p + sent, n - sent);
        if (w < 0)
        {
            if (errno == EINTR) continue;
            std::perror("bridge_ldpc: write");
            std::exit(1);
        }
        sent += static_cast<size_t>(w);
    }
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 2 && argc != 3)
    {
        std::fprintf(stderr, "usage: %s <path-to.qc>\n", argv[0]);
        return 2;
    }
    const std::string h_path = argv[1];

    // Load H the same way the real aff3ct-bin does for a QC-format matrix
    // (see src/Factory/Module/Decoder/LDPC/Decoder_LDPC.cpp's own store()),
    // deriving K/N_cw from the matrix itself rather than hardcoding them --
    // this bridge works for any of spectracuda's exported .qc variants, not
    // just ldpc_1944_r12.
    //
    // info_bits_pos is set below, NOT read from the file -- the QC format
    // carries no info-bit-position section at all (only AList does), and
    // AFF3CT's own auto-derivation (transform_H_to_G_identity()) picks an
    // arbitrary valid choice that doesn't match spectracuda's actual
    // convention anyway (see the real explanation further down, where
    // info_bits_pos is actually populated).
    std::vector<unsigned> info_bits_pos;
    tools::Sparse_matrix H = tools::LDPC_matrix_handler::read(h_path, &info_bits_pos);

    int M, N_cw;
    tools::LDPC_matrix_handler::read_matrix_size(h_path, M, N_cw);
    const int K = N_cw - M;

    if (argc == 3 && std::string(argv[2]) == "--dump")
    {
        // Diagnostic: print exactly what read() handed back (before any
        // transpose of ours), for direct cross-check against an
        // independent Python reconstruction of the same .qc file.
        std::fprintf(stderr, "H as returned by read(): n_rows=%zu n_cols=%zu\n",
                     H.get_n_rows(), H.get_n_cols());
        int printed = 0;
        for (size_t r = 0; r < H.get_n_rows() && printed < 40; ++r)
            for (size_t c = 0; c < H.get_n_cols() && printed < 40; ++c)
                if (H.at(r, c))
                {
                    std::fprintf(stderr, "(%zu,%zu) ", r, c);
                    printed++;
                }
        std::fprintf(stderr, "\n");
        return 0;
    }

    // info_bits_pos: NOT derived via transform_H_to_G_identity(). That
    // function searches H for *some* invertible KxK submatrix -- any
    // valid systematic choice -- with no knowledge of which K columns
    // spectracuda's own encoder actually uses. Confirmed by direct
    // testing: decode_siho() was reconstructing the correct N-bit
    // codeword the whole time (verified via the syndrome + connectivity
    // checks in this bridge's own git history), but reading the "K info
    // bits" back from transform_H_to_G_identity()'s own auto-picked
    // positions returned ~50%-mismatched garbage even for a clean,
    // noiseless codeword -- because those aren't the positions
    // spectracuda's encoder put the message in.
    //
    // spectracuda's actual convention (fec/ldpc.py's encode(): `H_m =
    // H[:, :k]`, `H_p = H[:, k:]`, parity computed via H_p's inverse) is
    // fixed and simple: the message occupies variable positions
    // [0, K), the parity occupies [K, N_cw) -- always, by construction,
    // not something to search for.
    info_bits_pos.resize(K);
    for (int i = 0; i < K; ++i) info_bits_pos[i] = static_cast<unsigned>(i);

    factory::Decoder_LDPC dec_params;
    dec_params.K          = K;
    dec_params.N_cw       = N_cw;
    dec_params.type       = "BP_FLOODING";
    dec_params.implem     = "NMS";
    dec_params.min        = "MINL";
    dec_params.norm_factor = 0.75f;
    dec_params.n_ite      = 50;
    dec_params.enable_syndrome = true;
    dec_params.syndrome_depth  = 1;

    auto* decoder = dec_params.build<int, float>(H, info_bits_pos);

    std::fprintf(stderr,
                 "bridge_ldpc: ready -- H=%s K=%d N_cw=%d (BP_FLOODING/NMS, "
                 "norm=0.75, ite=50, early-term ON)\n",
                 h_path.c_str(), K, N_cw);
    std::fflush(stderr);

    mipp::vector<float> llrs(N_cw);
    mipp::vector<int> bits(K);

    while (read_exact(STDIN_FILENO, llrs.data(), N_cw * sizeof(float)))
    {
        // Public raw-pointer decode_siho() returns the same `status` its
        // internal _decode() produced (0 = syndrome satisfied, matching
        // AFF3CT's own CWD = !status convention -- see
        // Decoder_LDPC_BP_flooding::_decode_siso/_decode_siho).
        int32_t status = static_cast<int32_t>(decoder->decode_siho(llrs.data(), bits.data()));
        write_exact(STDOUT_FILENO, &status, sizeof(status));
        write_exact(STDOUT_FILENO, bits.data(), K * sizeof(int));
    }

    delete decoder;
    return 0;
}
