"""Pin the spectracuda golden model the RTL is verified against.

The RTL implements the frame format of spectracuda at REF_COMMIT: a
one-symbol, uncoded BPSK header. Later spectracuda commits changed the
format (e374fdf: two-symbol header with CRC-16 + conv_v27; then DMRS and
the C2 region), which this RTL does not decode yet. Running the harness
against the working-tree spectracuda therefore does not test the RTL, it
tests a format mismatch -- and it fails in confusing ways (a header that
"never decodes", or the header's conv_v27 geometry mistaken for the
payload's).

So every harness script imports spectracuda from a git worktree of
REF_COMMIT, created by setup_ref.sh, and VERIFIES what it imported. Use:

    import golden_ref; golden_ref.use()      # before any spectracuda import

Override (for deliberate experiments only): SPECTRACUDA_REF=/path/to/tree.
The format check still runs, so pointing it at a tree with the protected
header stops with an explanation instead of producing wrong verdicts.

When the RTL gains the new header (docs/rx_modular_architecture.md
section 11, step 7), move REF_COMMIT forward and drop the format check.
"""
from __future__ import annotations

import os
import subprocess
import sys

REF_COMMIT = "ad0a396"

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "build", "spectracuda_ref")


def _fail(msg: str) -> None:
    raise SystemExit(f"golden_ref: {msg}")


def use() -> str:
    """Put the pinned spectracuda first on sys.path, import it, and check
    it is the right one. Returns the tree it was imported from."""
    tree = os.environ.get("SPECTRACUDA_REF", REF_DIR)
    if not os.path.isdir(os.path.join(tree, "spectracuda")):
        _fail(f"no spectracuda tree at {tree}\n"
              f"  run: hls/rtl/setup_ref.sh   (creates a worktree of {REF_COMMIT})")

    if "SPECTRACUDA_REF" not in os.environ:
        head = subprocess.run(["git", "-C", tree, "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        if head != REF_COMMIT:
            _fail(f"{tree} is at {head or '?'}, expected {REF_COMMIT}\n"
                  f"  run: hls/rtl/setup_ref.sh")

    if "spectracuda" in sys.modules:
        mod = sys.modules["spectracuda"]
        got = os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))
        if os.path.realpath(got) == os.path.realpath(tree):
            return tree                      # already pinned by an earlier call
        _fail(f"spectracuda was already imported from {got} before "
              f"golden_ref.use() -- call it before any spectracuda import")
    sys.path.insert(0, tree)

    # The venv has an editable install of the working tree; a sys.path
    # entry normally wins over it, but check rather than assume.
    import spectracuda
    got = os.path.dirname(os.path.dirname(os.path.abspath(spectracuda.__file__)))
    if os.path.realpath(got) != os.path.realpath(tree):
        _fail(f"imported spectracuda from {got}, not {tree}")

    # The format the RTL implements: uncoded header, no protected-header API.
    from spectracuda.framing import header
    if hasattr(header, "header_wire_len_bits"):
        _fail(f"{tree} has the protected (CRC-16 + conv_v27) header; the RTL "
              f"still implements the uncoded header of {REF_COMMIT}")
    return tree
