"""Shared PlutoSDR open/configure helpers for the debug/ scripts.

Every RX script here configured its adi.Pluto identically (same rf_bw
formula, same manual-gain setup); every TX script did the same on the TX
side. Pulled out once so there's a single place to change e.g. the rf_bw
formula or add a new knob, instead of four copies drifting apart.
"""
import adi


def pluto_rx_init(uri, freq, rate, rx_gain, rx_buffer_size):
    """Open a Pluto and configure it for RX-only use. Returns the adi.Pluto."""
    rf_bw = int(max(rate * 1.25, 5e6))
    rx = adi.Pluto(uri=uri)
    rx.sample_rate = int(rate)
    rx.rx_lo = int(freq)
    rx.rx_rf_bandwidth = rf_bw
    rx.gain_control_mode_chan0 = "manual"
    rx.rx_hardwaregain_chan0 = rx_gain
    rx.rx_buffer_size = rx_buffer_size
    return rx


def pluto_tx_init(uri, freq, rate, tx_gain):
    """Open a Pluto and configure it for TX-only use. Returns the adi.Pluto."""
    rf_bw = int(max(rate * 1.25, 5e6))
    sdr = adi.Pluto(uri=uri)
    sdr.sample_rate = int(rate)
    sdr.tx_lo = int(freq)
    sdr.tx_rf_bandwidth = rf_bw
    sdr.tx_hardwaregain_chan0 = float(tx_gain)
    sdr.tx_cyclic_buffer = False  # one-shot burst per tx() call, not a repeating cyclic buffer
    return sdr
