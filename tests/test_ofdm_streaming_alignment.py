"""Regression tests for two alignment-dependent frame losses in
Ofdm.rx_streaming()'s SEEKING state, both found 2026-09-09 on a CLEAN,
zero-noise channel while measuring streaming per-call overhead (see
docs/2026-09-09-rx-streaming-partial-preamble-fix.md):

1. Premature sync trigger on a PARTIALLY-arrived preamble. With k of the
   preamble's fft_size samples in the buffer (k > L = fft_size/2), the
   Schmidl-Cox metric at the very last candidate offset is already
   (2(k-L)/k)^2 -- 0.44 at k=192, 0.73 at k=224 -- over the 0.3 default
   threshold, with a start_index (fft_size - k) samples EARLY. Past the
   32-sample CP that means ISI and a failed decode.
2. The fixed-length SEEKING cap (`buffer[-2048:]` after the concat)
   discarding the whole previous buffer when one chunk is as long as the
   cap (2048 -- exactly what examples/pluto_*_unit.py feed), cutting off
   the head of a not-yet-detected preamble that started in the previous
   chunk's tail.

Both are pure functions of where the preamble start lands relative to
chunk boundaries, so these tests sweep that alignment explicitly, per
chunk size, and require EVERY alignment to decode. Zero lead-in, zero
noise: any failure here is structural, not a channel effect. Before the
fix: chunk=2048 lost 9.4% of alignments, 256 lost 6.2%, 64 lost ~30% in
steady state (the demos never saw it because they feed every frame from
buffer offset 0, where the premature candidate index is negative).
"""
from __future__ import annotations

import numpy as np
import pytest

from spectracuda.mac import Mac

PHY = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="rs_m8", fec1="conv_v27", crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse", backend="numpy",
)


def _zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype="complex64")


@pytest.fixture(scope="module")
def frame_and_ofdm():
    """One real, bound, single-PDU QPSK frame (the standard benchmark
    config) plus the receiving Ofdm -- same construction as
    examples/benchmark_x86_stages_v3.py, so this is the real wire
    format, not a synthetic preamble."""
    tx = Mac(mode="um", ofdm_kwargs=PHY)
    rx = Mac(mode="um", ofdm_kwargs=PHY)
    resp = rx.handle_bind_request_iq(tx.build_bind_request())
    assert tx.handle_bind_response_iq(resp)
    rng = np.random.default_rng(0)
    frames = tx.send_iq(rng.integers(0, 2, size=24000).astype("uint8"))  # exactly 1 PDU at QPSK
    assert len(frames) == 1
    return np.asarray(frames[0])[0].astype("complex64"), rx.ofdm


def _stream_decodes(ofdm, frame: np.ndarray, lead: int, chunk: int) -> bool:
    """Feed `lead` zero samples, the frame, then a zero tail, in `chunk`-
    sample pieces; True iff a CRC-valid frame came out."""
    ofdm.reset_stream()
    sig = np.concatenate([_zeros(lead), frame, _zeros(2048)])
    ok = False
    for i in range(0, sig.shape[-1], chunk):
        result = ofdm.rx_streaming(sig[i : i + chunk])
        if result is not None and bool(np.all(np.asarray(result["crc_valid"]))):
            ok = True
    return ok


# Lead-in offsets = where the preamble start lands relative to chunk
# boundaries. Each list covers the band that failed before the fix plus a
# couple of known-good alignments as controls. chunk=64 uses a long
# lead-in so the preamble sits in steady-state stream position, not at
# buffer offset 0 (the special case the demos happen to exercise).
CASES = {
    2048: list(range(1792, 2048, 16)) + [0, 1024],
    1024: list(range(800, 864, 4)) + [0, 512],
    256: list(range(32, 100, 4)) + [0],
    64: [4096 + o for o in range(0, 64, 4)],
}


@pytest.mark.parametrize("chunk", sorted(CASES))
def test_every_preamble_alignment_decodes_on_a_clean_channel(frame_and_ofdm, chunk):
    frame, ofdm = frame_and_ofdm
    failed = [lead for lead in CASES[chunk] if not _stream_decodes(ofdm, frame, lead, chunk)]
    assert not failed, (
        f"chunk={chunk}: {len(failed)}/{len(CASES[chunk])} preamble alignments failed to "
        f"decode on a clean channel -- lead-in offsets {failed}"
    )


def test_partial_preamble_does_not_trigger_early(frame_and_ofdm):
    """The mechanism itself, not just the outcome: with only 192 of the
    preamble's 256 samples in the buffer (metric ~0.44 at the trailing
    edge, over the 0.3 threshold) SEEKING must NOT have triggered; once
    the full preamble plus the L-sample margin is in, it must trigger at
    the exact true start."""
    frame, ofdm = frame_and_ofdm
    ofdm.reset_stream()
    lead = 4096
    assert ofdm.rx_streaming(_zeros(lead)) is None
    assert ofdm.rx_streaming(frame[:192]) is None
    assert ofdm._stream_state == "SEEKING", "triggered on a 3/4-arrived preamble"

    rest = 256 - 192 + ofdm.fft_size // 2  # the rest of the preamble + the L-sample margin
    assert ofdm.rx_streaming(frame[192 : 192 + rest]) is None  # header not in yet -> still no frame
    assert ofdm._stream_state == "WAITING_HEADER"
    fed = lead + 192 + rest
    buffer_abs_start = fed - ofdm._stream_buffer.shape[-1]
    assert ofdm._stream_frame_start == lead - buffer_abs_start, "detected start is not the true preamble start"


def test_cap_does_not_chop_a_preamble_straddling_a_2048_chunk_boundary(frame_and_ofdm):
    """Mechanism 2 in isolation: preamble starts 100 samples before the
    end of one 2048-sample chunk (only 100 of 256 samples present -> no
    metric at all, nothing to trigger on), and the next 2048-sample chunk
    must not evict its head. Before the fix the SEEKING cap did exactly
    that and the frame was never found."""
    frame, ofdm = frame_and_ofdm
    assert _stream_decodes(ofdm, frame, lead=2048 - 100, chunk=2048)
