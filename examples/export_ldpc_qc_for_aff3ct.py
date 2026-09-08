"""CLI wrapper around ``spectracuda.fec._native_src.aff3ct_bridge.qc_export.export_qc``
-- the actual export logic lives there now (alongside ``bridge_ldpc.cpp``,
the C++ driver that consumes its output), not here, since it's real glue
a decode backend depends on (``fec/_native_aff3ct.py``), not a one-off
script -- see that module's docstring for the format details and the two
real bugs found deriving it.

Not a shipped runtime dependency by itself -- generating a `.qc` still
requires reference/aff3ct/ built locally to be useful for anything (same
"reference, not shipped" status as that directory), but this script
remains here as the manual entry point for producing one outside of
`fec/_native_aff3ct.py`'s own on-demand caching.

Usage:
    python examples/export_ldpc_qc_for_aff3ct.py ldpc_1944_r12 out.qc
"""

from __future__ import annotations

import sys

from spectracuda.fec._native_src.aff3ct_bridge.qc_export import export_qc

if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "ldpc_1944_r12"
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"{variant}.qc"
    export_qc(variant, out_path)
