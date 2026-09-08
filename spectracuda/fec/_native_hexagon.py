"""Hexagon-DSP-offloaded ConvolutionalCode (conv_v27) backend -- the
Qualcomm-silicon counterpart to NativeConvolutionalNEON/SSE in
fec/_native.py, for QCS6490-class SoCs (e.g. the Radxa Q6A) that have a
Hexagon DSP with HVX alongside their Kryo/Cortex CPU cores.

STATUS AS OF THIS FILE'S CREATION: design scaffolding only, NOT a
working accelerator. hexagon_available() below always returns False on
every machine this has actually been run on (no Hexagon SDK, no Q6A
hardware were available in that session -- see
docs/hexagon-fec-offload-plan.md, written the same session, for the
full plan/rationale/open risks). This file exists so that:
  (a) fec/viterbi.py's dispatch chain has a real, correctly-gated slot
      to check, wired in now rather than left as a TODO comment -- it
      is provably inert everywhere except real Hexagon hardware with a
      compiled+deployed DSP skel, because of the AND of conditions in
      hexagon_available() below.
  (b) whoever picks this up next (once the Hexagon SDK + a Radxa Q6A
      are both in hand -- see docs/hexagon-fec-offload-plan.md's own
      "Do I have..." checklist) has the CPU-side shape already decided
      and only needs to fill in _open()/_decode_batch()/_close() with
      calls into the qaic-generated FastRPC stub, not design the whole
      seam from scratch.

WHY THIS IS ITS OWN MODULE, not just another branch in fec/_native.py
alongside NativeConvolutionalSSE/NEON: those two are genuinely the same
kind of thing (compile a local .so with cc/gcc/clang, dlopen it,
ctypes-call straight into it, same process, same address space). This
is categorically different -- a Hexagon build needs Qualcomm's own
hexagon-clang cross-toolchain (not the system cc), produces a "skel"
.so that runs in a SEPARATE processor's execution domain (the DSP, not
the CPU this Python process runs on), reached only through FastRPC (a
kernel-mediated RPC transport, not a local function call), and that
transport's per-call latency is the whole design constraint (see
"Batching is not optional" below) -- none of _native.py's existing
"compile once, cache the .so, ctypes.CDLL it" machinery applies.

WHY BATCHING IS NOT OPTIONAL HERE (the one thing to get right before
writing any DSP-side code): NativeConvolutional/NEON/SSE's decode()
loops over the batch in PYTHON, one ctypes call per row
(_native.py's NativeConvolutional.decode(), etc.) -- fine there, because
a ctypes call into code already loaded in this same process costs
nanoseconds. A FastRPC call crosses into a different execution domain
via the kernel and costs a fixed overhead per call, easily worth more
than the entire NEON decode time for one small PDU/row. Doing the
existing "one call per row" pattern over FastRPC would make this
backend a certain, provable regression before a single instruction of
HVX code even had a chance to help. The batched decode below is
designed around passing the WHOLE (n_batch, k) array to the DSP in ONE
FastRPC call (mirroring the batch-shape contract every Block in this
project already uses, per block.py's own docstring on why batching
matters for GPU dispatch -- same reasoning, different accelerator) and
looping over n_batch INSIDE the DSP-side C, not in this Python wrapper.
"""
from __future__ import annotations

import ctypes
import os
import platform
import threading
from typing import Optional

import numpy as np

_hexagon_lock = threading.Lock()
_hexagon_checked = False
_hexagon_available = False

# Where a real Radxa Q6A image would place a signed/deployed DSP skel
# for this interface, if one existed -- NOT verified against Radxa's
# own BSP docs (not in this repo's reference/qc6490/ material, which is
# HVX-ISA-reference and product-brief PDFs only, no FastRPC/skel-
# deployment documentation -- see docs/hexagon-fec-offload-plan.md's
# "Open risks" section). Almost certainly wrong path/filename; treat as
# a placeholder to correct once real deployment docs are in hand, not
# as a verified fact.
_SKEL_CANDIDATE_PATHS = (
    "/usr/lib/rfsa/adsp/libfec_hexagon_skel.so",
    "/vendor/lib/rfsa/adsp/libfec_hexagon_skel.so",
    "/usr/lib/rfsa/adsp/libfec_hexagon_skel.so.debug",  # different Qualcomm domains do not always agree
)

# CDSP/FastRPC userspace client libs -- one of these must be loadable
# for FastRPC to even be reachable from userspace at all, independent
# of whether OUR skel is deployed. Real Qualcomm library names
# (libcdsprpc.so for the "compute DSP" domain, libadsprpc.so for the
# older "audio DSP" domain some FastRPC examples target) -- which
# domain a stock Radxa Q6A image actually exposes to an unprivileged
# process, and whether it needs an entitlement/signing step Radxa's own
# BSP would have to grant, is an OPEN QUESTION, not yet answered (see
# docs/hexagon-fec-offload-plan.md).
_FASTRPC_LIBS = ("libcdsprpc.so", "libadsprpc.so")


def hexagon_available() -> bool:
    """True only if ALL of: this is an aarch64 machine, a FastRPC
    client library is loadable, AND a compiled+deployed skel for this
    project's specific fec_hexagon interface is present at one of the
    (currently placeholder/unverified) paths above.

    On every machine this project has actually run on so far, this
    returns False at the FIRST gate that doesn't hold -- there is no
    path in this function that can accidentally claim availability on
    hardware that doesn't have the real thing, same "decline rather
    than guess" discipline as _cpu_supports_sse41()/neon_available() in
    fec/_native.py. Checked once per process, cached -- same rationale
    as every other *_available() in that file.
    """
    global _hexagon_checked, _hexagon_available
    if _hexagon_checked:
        return _hexagon_available
    with _hexagon_lock:
        if _hexagon_checked:
            return _hexagon_available
        _hexagon_checked = True

        if platform.machine() not in ("aarch64", "arm64"):
            return False

        if not any(_try_load(lib) for lib in _FASTRPC_LIBS):
            return False

        if not any(os.path.exists(p) for p in _SKEL_CANDIDATE_PATHS):
            return False

        # Both gates passed -- but note this STILL doesn't prove the
        # skel actually implements the interface this module expects
        # (no handshake/version check exists yet -- see NativeConvolutionalHexagon's
        # _open() docstring). That verification step belongs there, at
        # first actual use, not here.
        _hexagon_available = True
    return _hexagon_available


def _try_load(name: str) -> bool:
    try:
        ctypes.CDLL(name)
        return True
    except OSError:
        return False


class NativeConvolutionalHexagon:
    """Drop-in accelerated backend for ConvolutionalCode's encode()/
    decode(), same batch-shape contract as NativeConvolutional/NEON/SSE
    in fec/_native.py -- BUT NOT YET IMPLEMENTED. Constructing this
    class always raises NotImplementedError right now; it exists so
    fec/viterbi.py's dispatch chain has a real class to import and a
    real (always-False) gate to check, not a speculative name.

    What goes here once the Hexagon SDK + a Radxa Q6A are both in hand
    (see docs/hexagon-fec-offload-plan.md for the full plan):
      - __init__: `qaic`-compile spectracuda/fec/_native_src/hexagon/
        fec_hexagon.idl into its generated stub, ctypes-load the
        generated stub .so (NOT the DSP-side skel directly -- the stub
        runs on THIS cpu and marshals the FastRPC call; the skel runs
        on the DSP, loaded by the OS's FastRPC daemon, never dlopen'd
        by this process), call its generated `_open()` to get a handle
        to one persistent correct_convolutional_hexagon instance on the
        DSP (mirrors NativeConvolutional.__init__'s
        correct_convolutional_create() call exactly, just across the
        RPC boundary instead of in-process).
      - encode()/decode(): package the WHOLE (n_batch, k) array (see
        this module's own docstring, "WHY BATCHING IS NOT OPTIONAL") and
        make ONE generated-stub call per encode()/decode() invocation,
        not one per row -- the opposite of NativeConvolutional's
        _encode_one/_decode_one per-row loop pattern in fec/_native.py,
        deliberately, for the reason that module's docstring gives.
      - __del__ or an explicit close(): release the DSP-side handle via
        the generated stub's `_close()`.
      - Correctness gate FIRST (bit-exact vs the pure-Python decoder,
        same k=1,6,39,194,4001,4002-residue sweep style
        tests/test_fec_native_acceleration.py already uses for
        SSE/NEON), THEN benchmark at REAL Mac/Ofdm-level batch sizes
        (not a synthetic single-row microbenchmark -- see this module's
        docstring on why the batching granularity IS the speed claim
        here), THEN wire into fec/viterbi.py's dispatch priority based
        on what that benchmark actually shows, not on the a priori "HVX
        is 8x wider than NEON" theory alone -- same "measured, not
        assumed" discipline every other backend in this codebase was
        held to before being trusted.
    """

    def __init__(self) -> None:
        if not hexagon_available():
            raise RuntimeError("Hexagon DSP FEC backend is not available")
        # Unreachable while hexagon_available() is unconditionally
        # False on every machine this has run on -- see module
        # docstring. Left as an explicit NotImplementedError (rather
        # than silently doing nothing useful) so this is loud if
        # hexagon_available()'s gates are ever loosened without also
        # filling in the real FastRPC calls below.
        raise NotImplementedError(
            "NativeConvolutionalHexagon's FastRPC calls are not yet implemented -- "
            "see this class's own docstring and docs/hexagon-fec-offload-plan.md"
        )

    def encode(self, bits: np.ndarray) -> np.ndarray:  # pragma: no cover - unimplemented
        raise NotImplementedError

    def decode(self, bits: np.ndarray) -> np.ndarray:  # pragma: no cover - unimplemented
        raise NotImplementedError
