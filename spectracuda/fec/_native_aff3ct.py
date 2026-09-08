"""Optional AFF3CT-backed LDPC decode -- NOT auto-detected/silently used
the way libcorrect's SSE/NEON native accel is (see _native.py): AFF3CT
is a large external project, built separately under reference/aff3ct/
(that directory's own "reference, not shipped" status), not something
every machine running spectracuda has. `LDPCCode(variant, decoder="aff3ct")`
opts into this explicitly and raises a clear error if the prerequisites
aren't there, rather than silently falling back to the native numpy/cupy
decode -- same "fail loud, don't guess" spirit as the rest of this
codebase's FEC modules.

Drives spectracuda/fec/_native_src/aff3ct_bridge/bridge_ldpc (a
standalone C++ process built by that directory's build.sh, linked
directly against AFF3CT's own library and calling its Decoder_LDPC/
decode_siho() API -- see that file's own module docstring for exactly
how, and for the two real bugs found getting it bit-exact against
spectracuda's own decode()). One persistent process per LDPC variant,
reused across calls and across LDPCCode instances within this
interpreter -- avoids paying AFF3CT's own per-invocation startup cost
that a subprocess-per-frame approach would (the actual problem the
"persistent process" protocol in bridge_ldpc.cpp exists to solve).

Protocol with the child process (binary, over its stdin/stdout -- see
bridge_ldpc.cpp's own module docstring for the authoritative spec):
    per call, write:  n_cw float32 LLRs, native-endian
    per call, read:   1 int32 status (0 = converged to a valid codeword,
                       matching AFF3CT's own syndrome check -- nonzero
                       means BP did not converge, mirroring spectracuda's
                       own decode()'s "raise on non-convergence" contract)
                       then k int32 decoded bits (0/1), native-endian
"""

from __future__ import annotations

import atexit
import os
import struct
import subprocess
import tempfile
from typing import Dict, Tuple

import numpy as np

_BRIDGE_DIR = os.path.join(os.path.dirname(__file__), "_native_src", "aff3ct_bridge")
_BRIDGE_BIN = os.path.join(_BRIDGE_DIR, "bridge_ldpc")

_QC_CACHE_DIR = os.path.join(tempfile.gettempdir(), "spectracuda_aff3ct_qc")


class Aff3ctUnavailable(RuntimeError):
    """Raised when decoder="aff3ct" is requested but bridge_ldpc isn't
    built (or reference/aff3ct isn't) -- never silently falls back."""


def _check_available() -> None:
    if not os.path.isfile(_BRIDGE_BIN):
        raise Aff3ctUnavailable(
            f"decoder='aff3ct' requires a built bridge_ldpc at {_BRIDGE_BIN!r}. "
            f"Nothing clones/builds AFF3CT automatically (it's ~850MB and takes "
            f"several minutes to build -- deliberately not a pip dependency or "
            f"part of a fresh git clone). One-shot setup (clones AFF3CT into "
            f"reference/aff3ct/, builds it, then builds this bridge):\n"
            f"    {os.path.join(_BRIDGE_DIR, 'setup_aff3ct.sh')}\n"
            f"If reference/aff3ct/ is already built, just:\n"
            f"    {os.path.join(_BRIDGE_DIR, 'build.sh')}\n"
            f"This is dev/benchmark-only infrastructure, not a shipped runtime "
            f"dependency -- use decoder='native' (the default) if you don't need it."
        )


def _qc_path_for(variant: str) -> str:
    """Regenerated on every call, not trusted stale -- cheap (one-off
    table expansion), same philosophy as
    examples/benchmark_x86_stages_ldpc_aff3ct.py's own export_qc() call."""
    os.makedirs(_QC_CACHE_DIR, exist_ok=True)
    path = os.path.join(_QC_CACHE_DIR, f"{variant}.qc")
    from .ldpc_tables import BASE_MATRICES  # noqa: F401 -- validates variant early

    from ._native_src.aff3ct_bridge.qc_export import export_qc

    export_qc(variant, path, quiet=True)
    return path


def _read_exact(f, n: int) -> bytes:
    chunks = []
    got = 0
    while got < n:
        chunk = f.read(n - got)
        if not chunk:
            raise RuntimeError(
                "bridge_ldpc process closed its stdout unexpectedly "
                "(crashed? check its stderr)"
            )
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


class _BridgeProcess:
    def __init__(self, variant: str, n: int, k: int) -> None:
        _check_available()
        qc_path = _qc_path_for(variant)
        self.n = n
        self.k = k
        self.proc = subprocess.Popen(
            [_BRIDGE_BIN, qc_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def decode_one(self, channel_llr: np.ndarray) -> Tuple[int, np.ndarray]:
        """channel_llr: (n,) float32 array, spectracuda's own LLR sign
        convention (positive -> bit 0, negative -> bit 1 -- see
        ldpc.py's decode(), and bridge_ldpc.cpp's own confirmation this
        matches AFF3CT's BPSK modem convention exactly). Returns
        (status, bits) -- status==0 means converged (bits trustworthy),
        matching bridge_ldpc's own protocol."""
        assert channel_llr.shape == (self.n,), channel_llr.shape
        payload = channel_llr.astype("<f4", copy=False).tobytes()
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()
        resp = _read_exact(self.proc.stdout, 4 + self.k * 4)
        (status,) = struct.unpack("<i", resp[:4])
        bits = np.frombuffer(resp[4:], dtype="<i4").astype(np.uint8)
        return status, bits

    def close(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


_processes: Dict[str, _BridgeProcess] = {}


def _close_all() -> None:
    for p in _processes.values():
        p.close()
    _processes.clear()


atexit.register(_close_all)


def get_process(variant: str, n: int, k: int) -> _BridgeProcess:
    proc = _processes.get(variant)
    if proc is None:
        proc = _BridgeProcess(variant, n, k)
        _processes[variant] = proc
    return proc


def decode_batch(variant: str, n: int, k: int, channel_llr_batch: np.ndarray) -> np.ndarray:
    """channel_llr_batch: (n_batch, n) float32. Returns (n_batch, k)
    uint8 decoded bits. Raises ValueError (matching LDPCCode.decode()'s
    own convention) listing which batch item(s) AFF3CT failed to
    converge on, rather than returning untrustworthy bits silently."""
    proc = get_process(variant, n, k)
    n_batch = channel_llr_batch.shape[0]
    out = np.empty((n_batch, k), dtype=np.uint8)
    bad_items = []
    for i in range(n_batch):
        status, bits = proc.decode_one(channel_llr_batch[i])
        out[i] = bits
        if status != 0:
            bad_items.append(i)
    if bad_items:
        raise ValueError(
            f"AFF3CT LDPC decode failed to converge to a zero-syndrome codeword "
            f"for batch item(s) {bad_items} (variant {variant!r})"
        )
    return out
