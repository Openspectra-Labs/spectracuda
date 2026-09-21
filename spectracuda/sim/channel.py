"""Channel: reusable impairment simulator, modeled on liquid-dsp's
channel_cccf (docs/liquid-dsp-api-inventory.md lists its four
impairments: add_awgn, add_carrier_offset, add_multipath, add_shadowing).
Covers the first three -- shadowing (correlated log-normal fading) is a
documented future addition, not built here.

liquid-dsp builds a channel incrementally and applies it with one call:

    channel_cccf channel = channel_cccf_create();
    channel_cccf_add_awgn(channel, noise_floor, SNRdB);
    channel_cccf_add_carrier_offset(channel, dphi, 0.0f);
    channel_cccf_execute_block(channel, buf, buf_len, buf);

This class takes the same impairments as constructor parameters instead
of a stateful add_*() builder -- matches spectracuda's
everything-in-the-constructor convention (established by Ofdm) rather
than reproducing liquid-dsp's builder idiom -- and applies them all with
one process() call. Not OFDM-specific; works on any complex time-domain
samples. This consolidates impairment code that used to be hand-rolled
inline in examples/ofdm_256_*.py.

Batch-shape contract: process(tx_iq) takes (n_batch, n_samples) complex
-> (n_batch, n_samples) complex (same shape), unless `tail_samples` is
set -- see below. Multipath is applied as a full convolution, then
truncated back to the input length -- the physically-realistic channel
tail beyond that point is discarded, matching what a frame's cyclic
prefix/guard interval is meant to absorb.

TIME-VARYING MULTIPATH (`tap_doppler_hz`)
-----------------------------------------
`multipath_taps` alone is a STATIC channel: the taps never move, so H[k]
never ages. `cfo` does not fix that either -- it rotates the whole signal
by one common phase, which a per-symbol CPE correction removes entirely.
Neither can express the impairment that actually ages a channel estimate:
different propagation paths arriving with DIFFERENT Doppler shifts.

`tap_doppler_hz` gives each tap its own shift:

    y[n] = sum_k taps[k] * exp(j*2*pi*f_k*n/fs) * x[n-k]

The quantity that ages H[k] is the DIFFERENCE between tap Dopplers, not
their absolute values. `tap_doppler_hz=[1600, 1600]` is a 1600 Hz common
shift that CFO+CPE absorb, leaving the frequency-selective shape static;
`[1600, 1900]` carries the same common shift plus 300 Hz of DIFFERENTIAL
Doppler, and only the latter makes the initial estimate go stale. See
docs/2026-09-21-dmrs-differential-doppler-characterization.md, which
measures that separation directly.

The phase uses the OUTPUT sample index n, not n-k, so `taps[k]`'s shift
is evaluated at the instant the sample leaves the channel rather than
when it entered. The two conventions differ by a constant per-tap phase
exp(-j*2*pi*f_k*k/fs) -- for the sub-microsecond delays and sub-kHz
shifts this is built for, that is a fraction of a degree, and it is
folded into the tap's own complex value either way.

TWO METHODOLOGY KNOBS (`tail_samples`, `noise_draw_len`)
--------------------------------------------------------
Both exist because getting them wrong silently corrupted a real round of
measurements -- see docs/2026-09-20-dmrs-static-channel-cost.md, where
the resulting artifact was initially mistaken for a genuine DMRS cost:

* `tail_samples` appends that many zeros to the input BEFORE the channel
  runs, so the output is (n_samples + tail_samples) long. Multipath can
  move a receiver's detected frame start a sample late; a frame that ends
  flush with the buffer then needs one sample past the end and extraction
  fails, which looks like a lost packet but is a simulation boundary. A
  real capture always has samples after the frame. Default None keeps the
  shape-preserving behaviour exactly.

* `noise_draw_len` draws the noise at a FIXED length and slices it to fit
  instead of drawing exactly n_samples. Drawing `standard_normal(n)`
  twice makes the imaginary part depend on n, so the same seed hands
  different-length frames an entirely UNCORRELATED noise realization --
  and anything that changes frame length (a DMRS interval, a payload
  size) then gets different noise, producing apparent differences that
  are pure artifact. Must be >= n_samples. Default None keeps current
  behaviour.

  Precisely: this fixes the noise SAMPLES, not the final noise
  amplitude. `snr_db` still scales against each input's own measured
  power, so two different-length inputs get the same noise sequence
  scaled slightly differently. That residual is second-order and
  deliberate -- SNR is defined against the signal actually present --
  whereas the draw-length effect it replaces was a wholesale change of
  realization.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..block import Block


class Channel(Block):
    """Parameters
    ----------
    snr_db:
        AWGN signal-to-noise ratio in dB, measured against the actual
        input signal's own power (per batch item) -- None disables noise.
    multipath_taps:
        Complex FIR tap array (length = number of channel taps), or None
        to disable multipath. Use `Channel.random_multipath_taps(...)`
        for a quick random Rayleigh-ish channel.
    cfo:
        Carrier frequency offset as a fraction of subcarrier spacing
        (same normalized units SchmidlCoxCFO estimates/corrects), or
        None to disable. Requires `cfo_fft_size`.
    cfo_fft_size:
        OFDM FFT size used to convert `cfo` into a phase-per-sample
        rotation: phase[n] = exp(j*(2*pi*cfo*n/cfo_fft_size + cfo_phase0)).
    cfo_phase0:
        Initial (constant) phase offset in radians, applied alongside cfo.
    seed:
        Seed for the AWGN generator (noise is generated on host via numpy
        regardless of backend, then moved to the active xp -- fine for a
        test/simulation utility, not a hot-path production concern).
    """

    def __init__(
        self,
        *,
        snr_db: Optional[float] = None,
        multipath_taps: Optional[Any] = None,
        cfo: Optional[float] = None,
        cfo_fft_size: Optional[int] = None,
        cfo_phase0: float = 0.0,
        tap_doppler_hz: Optional[Any] = None,
        sample_rate_hz: Optional[float] = None,
        tail_samples: Optional[int] = None,
        noise_draw_len: Optional[int] = None,
        seed: Optional[int] = None,
        backend=None,
    ) -> None:
        super().__init__(backend=backend)
        if cfo is not None and cfo_fft_size is None:
            raise ValueError("cfo_fft_size is required when cfo is set")

        self.snr_db = snr_db
        self.multipath_taps = (
            None if multipath_taps is None else self.xp.asarray(multipath_taps, dtype="complex64")
        )
        # Hz, not normalized units: a per-tap Doppler shift is a physical
        # frequency and its effect depends on elapsed TIME, so it needs a
        # real sample rate rather than cfo's subcarrier-spacing fractions.
        if tap_doppler_hz is not None:
            if self.multipath_taps is None:
                raise ValueError(
                    "tap_doppler_hz needs multipath_taps -- it gives each "
                    "EXISTING tap its own Doppler shift, it does not create taps"
                )
            if sample_rate_hz is None:
                raise ValueError("sample_rate_hz is required when tap_doppler_hz is set")
            tap_doppler_hz = np.asarray(tap_doppler_hz, dtype="float64").reshape(-1)
            if tap_doppler_hz.shape[-1] != self.multipath_taps.shape[-1]:
                raise ValueError(
                    f"tap_doppler_hz has {tap_doppler_hz.shape[-1]} entries but "
                    f"multipath_taps has {self.multipath_taps.shape[-1]} -- one "
                    f"shift per tap"
                )
        self.tap_doppler_hz = tap_doppler_hz
        self.sample_rate_hz = sample_rate_hz
        if tail_samples is not None and tail_samples < 0:
            raise ValueError(f"tail_samples must be >= 0, got {tail_samples}")
        self.tail_samples = tail_samples
        self.noise_draw_len = noise_draw_len
        self.cfo = cfo
        self.cfo_fft_size = cfo_fft_size
        self.cfo_phase0 = cfo_phase0
        self._rng = np.random.default_rng(seed)
        self.batch_shape_doc = "(n_batch, n_samples) complex tx in -> (n_batch, n_samples) complex rx out"

    @staticmethod
    def random_multipath_taps(n_taps: int, seed: Optional[int] = None) -> Any:
        """Unit-total-energy complex Gaussian taps -- a quick stand-in
        Rayleigh-ish multipath channel, matching what the OFDM examples
        generated inline before this class existed. Normalized explicitly
        against its own realized energy (not just in expectation) so
        "unit energy" holds exactly for every draw, not just on average."""
        rng = np.random.default_rng(seed)
        taps = rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)
        taps = taps / np.sqrt(np.sum(np.abs(taps) ** 2))
        return taps.astype("complex64")

    def process(self, tx_iq: Any, **kwargs: Any) -> Any:
        xp = self.xp
        tx_iq = xp.asarray(tx_iq)
        if tx_iq.ndim == 1:
            tx_iq = tx_iq[None, :]
        n_batch, n_samples = tx_iq.shape

        # Trailing capture samples, BEFORE the channel runs so multipath
        # smears into them the way it would in a real capture -- see the
        # module docstring for why a frame ending flush with the buffer
        # produces false losses.
        if self.tail_samples:
            tx_iq = xp.concatenate(
                [tx_iq, xp.zeros((n_batch, self.tail_samples), dtype=tx_iq.dtype)], axis=-1
            )
            n_samples = tx_iq.shape[-1]

        rx = tx_iq
        if self.multipath_taps is not None and self.tap_doppler_hz is None:
            n_taps = self.multipath_taps.shape[-1]
            convolved = xp.empty((n_batch, n_samples + n_taps - 1), dtype="complex64")
            for b in range(n_batch):
                convolved[b] = xp.convolve(rx[b], self.multipath_taps)
            rx = convolved[:, :n_samples]  # truncate back to input length -- see module docstring
        elif self.multipath_taps is not None:
            # Time-varying: each tap carries its own Doppler, so this is a
            # per-tap shift-and-add rather than one convolution. Equivalent
            # to the branch above (convolve, then truncate) when every
            # tap_doppler_hz is 0 -- asserted in tests/test_channel_sim.py.
            n = xp.arange(n_samples, dtype="float64")
            out = xp.zeros((n_batch, n_samples), dtype="complex64")
            for k in range(self.multipath_taps.shape[-1]):
                shifted = rx if k == 0 else xp.concatenate(
                    [xp.zeros((n_batch, k), dtype=rx.dtype), rx[:, :-k]], axis=-1
                )
                rot = xp.exp(
                    1j * 2 * xp.pi * float(self.tap_doppler_hz[k]) * n / self.sample_rate_hz
                ).astype("complex64")
                out = out + self.multipath_taps[k] * rot[None, :] * shifted
            rx = out

        if self.snr_db is not None:
            sig_power = xp.mean(xp.abs(rx) ** 2, axis=-1, keepdims=True)
            noise_power = sig_power / (10 ** (self.snr_db / 10))
            if self.noise_draw_len is None:
                noise = (
                    self._rng.standard_normal((n_batch, n_samples))
                    + 1j * self._rng.standard_normal((n_batch, n_samples))
                )
            else:
                # Fixed-length draw, then sliced: keeps the same seed
                # sample-identical across frame lengths -- module docstring.
                if self.noise_draw_len < n_samples:
                    raise ValueError(
                        f"noise_draw_len={self.noise_draw_len} is shorter than the "
                        f"{n_samples} samples being processed"
                        + (" (tail_samples included)" if self.tail_samples else "")
                    )
                draw = (
                    self._rng.standard_normal((n_batch, self.noise_draw_len))
                    + 1j * self._rng.standard_normal((n_batch, self.noise_draw_len))
                )
                noise = draw[:, :n_samples]
            rx = rx + xp.asarray(noise, dtype="complex64") * xp.sqrt(noise_power / 2)

        if self.cfo is not None:
            n = xp.arange(n_samples)
            phase = xp.exp(1j * (2 * xp.pi * self.cfo * n / self.cfo_fft_size + self.cfo_phase0))
            rx = rx * phase[None, :]

        return rx.astype("complex64")
