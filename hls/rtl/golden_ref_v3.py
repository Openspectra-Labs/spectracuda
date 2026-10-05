"""Pinned Python reference for the v3 RTL profile (protected header,
interleaver2, DMRS, 4-bit soft decision with the streaming LLR scale).

Every v3 harness (RX and TX) imports spectracuda from a worktree pinned
here, never from the working tree, so results reproduce from the branch.
The old-format flows keep using golden_ref.py (ad0a396).

    hls/rtl/build/spectracuda_ref_v3  ->  REF_COMMIT
    create with: git worktree add --detach hls/rtl/build/spectracuda_ref_v3 <REF_COMMIT>
"""
import os, subprocess, sys

REF_COMMIT = "dc8b882"
HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "build", "spectracuda_ref_v3")


def use() -> str:
    if not os.path.isdir(REF_DIR):
        raise SystemExit(f"missing {REF_DIR}\n  run: git worktree add --detach "
                         f"hls/rtl/build/spectracuda_ref_v3 {REF_COMMIT}")
    head = subprocess.check_output(["git", "-C", REF_DIR, "rev-parse", "HEAD"], text=True).strip()
    if not head.startswith(REF_COMMIT):
        raise SystemExit(f"{REF_DIR} is at {head[:7]}, expected {REF_COMMIT}")
    dirty = subprocess.check_output(["git", "-C", REF_DIR, "status", "--porcelain", "spectracuda"], text=True)
    if dirty.strip():
        raise SystemExit(f"{REF_DIR} has local changes to spectracuda/ -- not a pinned reference")
    if REF_DIR not in sys.path:
        sys.path.insert(0, REF_DIR)
    return head


def ofdm(modem: str, dmrs_interval: int = 0, **extra):
    """The exact profile the v3 RTL implements."""
    from spectracuda.pipeline import Ofdm
    kw = dict(fft_size=256, cp_len=32, n_data=216, n_pilot=8,
              sync="schmidl_cox", cfo="schmidl_cox", channel_estimator="ls", equalizer="mmse",
              modem=modem, fec1="conv_v27", interleaver="block",
              interleaver_kwargs={"unit_bits": 8}, interleaver2="block",
              # soft decision: 4-bit LLRs, scale known before the payload
              # (header noise, training |H|^2) -- see Ofdm soft_llr_scale
              soft_decision=True, soft_llr_bits=4, soft_llr_scale="stream",
              soft_llr_clip=3.0, dmrs_interval=dmrs_interval)
    kw.update(extra)
    return Ofdm(**kw)


DMRS_CODE = {0: 0, 16: 1, 32: 2, 64: 3}
MOD_CODE = {"qpsk": 1, "qam16": 2, "qam64": 3}
