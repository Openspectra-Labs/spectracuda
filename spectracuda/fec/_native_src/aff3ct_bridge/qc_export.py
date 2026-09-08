"""Export one of spectracuda's own 802.11n QC-LDPC base matrices (see
``spectracuda/fec/ldpc_tables.py``) into AFF3CT's native ".qc" file
format, so AFF3CT's decoder can be driven against the EXACT same code
this project uses elsewhere (CPU array-op decode in fec/ldpc.py, the
hand-written CUDA kernel in prototype_ldpc_cuda_kernel.py) -- not just
"a" 802.11n-shaped code of the same size. Used by both
``examples/export_ldpc_qc_for_aff3ct.py`` (standalone CLI, for manual
AFF3CT benchmarking) and ``fec/_native_aff3ct.py`` (the real decode
backend, which generates/caches a `.qc` per variant on demand).

Lives alongside ``bridge_ldpc.cpp`` here, not in examples/, for the same
reason libcorrect's C source lives under `fec/_native_src/` rather than
examples/: this is glue a real decode path depends on, even though (like
the rest of this directory's AFF3CT half) it requires reference/aff3ct/
to be built locally to be useful -- see that directory's own
"reference, not shipped" status.

## Format, verified by reading AFF3CT's own parser
(``reference/aff3ct/src/Tools/Code/LDPC/QC/QC.cpp``, function
``QC::_read``) directly, and cross-checked end-to-end against a real
spectracuda-encoded codeword decoded through AFF3CT's own C++
Decoder_LDPC API (see ``verify_bridge.py`` in this same directory, which
caught the two mistakes below -- an aff3ct-bin self-consistency run
alone, decoding only its own internally-generated payloads, can't: it
never round-trips a real spectracuda codeword through it, so a wrong
export can still look "correct" from that side alone):

NEITHER of the two transforms once assumed necessary here are: no
transpose, no shift-sign negation. Both were traced from a genuine
misreading of `_read`'s own index math and disproven by that real
codeword round-trip (a clean, noiseless spectracuda codeword failed to
satisfy AFF3CT's loaded H before this fix, and decoded correctly after
it):

- **Layout**: header is ``N_red M_red Z``; `_read` allocates
  ``H_red(M_red, N_red)`` (M_red *rows* of N_red values) and reads
  exactly M_red file lines. That's spectracuda's base matrix
  transcribed row-for-row as-is: N_red = nb (=24, variable-block-cols,
  values per line), M_red = mb (check-block-rows, number of lines).
- **Shift sign**: `_read`'s edge rule is
  ``add_connection(idxLgn+k, idxCol+(k+value)%Z)`` where the first arg
  is the check row (``idxLgn+k``) and the second is the variable column
  (``idxCol+(k+value)%Z``) -- i.e. ``var = (check_local + value) % Z``.
  That's identical to spectracuda's own ``_expand_base_matrix()``
  (``var = (check_local + shift) % Z``, see ldpc.py) with ``value =
  shift`` copied directly. The base-matrix entries are written to the
  file unchanged.
"""

from __future__ import annotations

from spectracuda.fec.ldpc_tables import BASE_MATRICES


def export_qc(variant: str, out_path: str, *, quiet: bool = False) -> None:
    spec = BASE_MATRICES[variant]
    Z = spec["Z"]
    base = spec["base"]
    mb = len(base)  # check-block-rows
    nb = len(base[0])  # variable-block-cols (always 24 for 802.11n)

    # Header order + body layout verified empirically against the real
    # aff3ct-4.7.0 binary (not just QC::_read's raw connectivity): the
    # Codec/Decoder_LDPC factory reads the *first* header number as the
    # codeword size (N_cw) axis, which must be nb*Z = n. That means the
    # header is "nb mb Z" and the body is mb LINES (one per spectracuda
    # base-matrix row / check-block-row) of nb values each -- i.e.
    # spectracuda's base matrix transcribed row-for-row, unswapped.
    lines = [f"{nb} {mb} {Z}"]
    for j in range(mb):  # one file-line per check-block-row (as-is)
        row_vals = []
        for i in range(nb):  # each line lists all var-block-col shifts
            k = base[j][i]
            row_vals.append(k)  # -1 (no edge) or the shift itself, unmodified
        lines.append(" ".join(str(v) for v in row_vals))

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    if not quiet:
        n = spec["n"]
        n_checks = mb * Z
        k_bits = n - n_checks
        print(
            f"wrote {out_path}: variant={variant} Z={Z} mb={mb} nb={nb} "
            f"n={n} k={k_bits} rate={spec['rate']}"
        )
