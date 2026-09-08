# What's next

Part I (the PHY chain) and Part II (the MAC layer) are both written now —
Chapters 01–12 cover the whole tx→channel→rx path plus TM/UM/AM on top of
it, every code example on every page verified against real, running
code. This page is what's genuinely still ahead, not a placeholder for
chapters that already exist.

## In progress

**A hand-written, template-parameterized HLS C++ port of the OFDM
pipeline, targeting real FPGA gateware via Vitis HLS.** The active
priority track as of this writing — no synthesizable code exists yet,
only a staged Python↔HLS validation plan. This is a genuine, large
undertaking, not a small extension of the CPU/GPU pipeline: fixed-point
quantization, AXI4-Stream interfaces, and bit-exact C-simulation
verification against this project's own Python reference all have to be
built from nothing. Watch this space rather than the changelog — a
chapter will exist here once there's real synthesized-and-placed
resource-utilization numbers to show, not before.

## Open, honestly

**The Schmidl-Cox sync detector's WiFi-preamble vulnerability**
({doc}`../hardware-validation`) — a pure self-correlation detector can't
structurally distinguish this system's own preamble from a nearby WiFi
one sharing the same repeated-halves structure. Frequency selection works
around it; a cross-correlation confirmation stage would fix it for real,
inside actively-used WiFi channels. Proposed, not built.

**A full recorded AM/ARQ run over real RF.** The two-node hardware test
harness exists (`abhi/mac_am_node_a_test.py`/`mac_am_node_b_test.py`),
and self-interference between simultaneous TX/RX on one node was
pre-characterized (no measurable effect at either power level tested) —
but no complete run with a recorded retransmission result is in yet.

**Hexagon DSP FEC offload** (`spectracuda/fec/_native_hexagon.py`,
{doc}`../fec-c-lib-acceleration`) — design scaffolding for QCS6490-class
SoCs' Hexagon+HVX cores, unconditionally inert until real hardware and
the Hexagon SDK are in hand to benchmark against.

---

Nothing in this book substitutes for {doc}`../architecture`,
{doc}`../mac`, {doc}`../fec-c-lib-acceleration`, or {doc}`../todo` —
those are the actual source of truth, including every real bug found and
fixed along the way. This book is the guided tour; those documents are
the map.
