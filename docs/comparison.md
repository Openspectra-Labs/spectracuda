# How spectracuda compares

spectracuda occupies a narrower space than a general DSP framework and a
more flexible space than a standards-compliant cellular stack. Its design
center is a **custom point-to-point OFDM link whose PHY, framing, simplified
MAC, simulation, and real-radio execution remain in one inspectable Python
codebase**.

The tables below compare intended workflows, not raw feature counts. A check
means that the capability is integrated into the platform's normal workflow;
"build it" means that useful lower-level components exist but the application
must supply the integration and policy.

## Open-source SDR platforms

| Capability | spectracuda | GNU Radio | liquid-dsp | srsRAN Project | OpenAirInterface |
|---|---|---|---|---|---|
| Primary design center | Custom OFDM links | General streaming DSP flowgraphs | Embedded DSP and packet/frame primitives | Standards-compliant 5G CU/DU and 4G stack | Standards-compliant 4G/5G RAN and UE |
| Custom OFDM PHY parameters and algorithms | **First-class:** swap sync, CFO, estimator, equalizer, modem and FEC | **First-class building blocks;** complete custom chain is assembled by the application | Configurable OFDM framing, subcarriers, modulation and FEC; receiver algorithms are comparatively monolithic | Possible only through substantial modification of a 3GPP implementation | Possible only through substantial modification of a 3GPP implementation |
| Complete frame acquisition and payload RX | Yes: batch and arbitrary-chunk streaming APIs | Build or select a flowgraph/application | Yes: `ofdmflexframesync` | Yes, as part of the 5G/4G stack | Yes, as part of the 5G/4G stack |
| FEC included | Convolutional/Viterbi, shortened RS, 12 QC-LDPC variants, concatenation and interleaving | Many blocks/modules; exact set depends on installed components | Broad FEC API, including soft decoding for supported codes; some schemes require external `libfec` | 3GPP coding required by the implemented stack | 3GPP coding required by the implemented stack |
| Link/MAC layer | Simplified point-to-point TM/UM/AM behavior, segmentation, bind, quality reports and ARQ | Application-specific: protocols and schedulers must be selected or built | No integrated link MAC/RLC | Full 3GPP L1/L2/L3, including MAC/RLC | Full 3GPP PHY/MAC and higher RAN layers |
| Adaptive MCS | Experimental link controller in the drone application; not yet a mature scheduler | Build it or use an application/module that supplies it | Build it | Standards-driven cellular scheduling/MCS | Standards-driven cellular scheduling/MCS, including dynamic MCS in supported modes |
| Same high-level implementation for simulation and Pi host | Yes | Often, depending on blocks and scheduler | Same C library, but application/hardware transport is separate | Portable C++, but deployment is a cellular system rather than a small custom link | Portable cellular implementation; substantially heavier deployment |
| Demonstrated project target | Two Raspberry Pi 5 + PlutoSDR nodes over real RF | Very broad hardware ecosystem | Library-level integration into many embedded applications | COTS UE/RU and 5G testbeds | COTS RAN/UE, O-RAN and RF simulators |
| Runtime/language | Python orchestration; NumPy, Numba, native C/SIMD and CuPy paths | C++ runtime with Python and graphical flowgraph tooling | C library | C++ cellular stack | C/C++ cellular stack |
| Best fit | Rapidly changing and validating a proprietary/non-standard OFDM waveform | Building arbitrary streaming DSP/radio applications from a large block ecosystem | Small, efficient DSP/framing library embedded in a C/C++ application | Deploying or researching a standards-compliant private 4G/5G network | Researching and deploying standards-compliant 4G/5G/O-RAN systems |

### Reading the comparison correctly

GNU Radio is broader than spectracuda. It is a general signal-processing
runtime with a large ecosystem, while spectracuda supplies one opinionated
OFDM object and a directly connected link layer. [GNU Radio describes itself
as a free, open-source toolkit of signal-processing blocks usable with SDR
hardware or without hardware](https://www.gnuradio.org/about/). The trade-off
is breadth versus the amount of waveform integration an application author
must own.

liquid-dsp is spectracuda's closest algorithmic relative and already provides
a capable OFDM flexible-frame generator and synchronizer. Its frame header can
signal payload length, modulation, FEC and integrity checking, and the receiver
automatically reconfigures for the payload. [The liquid-dsp OFDM framing
reference documents that complete path](https://liquidsdr.org/doc/ofdmflexframe/).
liquid-dsp also has a broader modem/FEC catalogue and supports soft-demodulator
LLRs for compatible FEC schemes, an area where spectracuda is currently behind.
[Its FEC reference documents the soft-decoding interface](https://liquidsdr.org/doc/fec/).
spectracuda differs by exposing receiver algorithms as swappable strategies,
adding QC-LDPC, and integrating its simplified MAC, Python simulation, Pi 5
runtime, and Pluto transport around the PHY.

srsRAN and OpenAirInterface are much more complete protocol stacks, but they
solve a different problem. srsRAN describes itself as a complete 3GPP/O-RAN
L1/L2/L3 solution, and its DU includes PHY, MAC and RLC. [srsRAN Project
overview](https://docs.srsran.com/projects/project/en/latest/) and
[component architecture](https://docs.srsran.com/projects/project/en/latest/knowledge_base/source/gnb_components/source/index.html).
OpenAirInterface likewise supplies a full 5G RAN/UE stack with PHY/MAC,
scheduling, HARQ and higher layers. [OpenAirInterface RAN capabilities](https://openairinterface.org/ran/).
They are preferable when interoperability with commercial 4G/5G equipment is
the objective. spectracuda is preferable when changing the waveform itself is
the objective and 3GPP conformance is not required.

## MATLAB and Simulink comparison

MATLAB should be treated as an ecosystem rather than one product. Relevant
features are distributed across MATLAB, Simulink, Communications Toolbox,
WLAN/5G toolboxes, SDR support packages, Fixed-Point Designer, HDL products,
and code-generation products. The exact comparison therefore depends on which
licenses and workflow are selected.

| Capability | spectracuda | MATLAB/Simulink ecosystem |
|---|---|---|
| Custom OFDM simulation | Integrated in the library | Strong support through Communications Toolbox functions, System objects, Simulink blocks and examples |
| Complete custom OFDM TX/RX example | `Ofdm.generate_frame()`, `rx_process()` and `rx_streaming()` are the product's central API | Yes. MathWorks publishes a complete SISO link example with synchronization/control signals, convolutional coding, channel impairment, demodulation and decoding |
| Swappable receiver strategies | Named sync, CFO, channel-estimation and equalization objects are public composition points | Algorithms are available, but a custom composition is generally assembled from functions, System objects, blocks or an example model |
| FEC | Convolutional/Viterbi, shortened RS, twelve 802.11n QC-LDPC codes and two-stage composition | Broader mature coding catalogue across Communications, WLAN, 5G and related toolboxes |
| Custom point-to-point MAC included with the PHY | Yes: simplified TM/UM/AM semantics, segmentation/reassembly, bind, link-quality exchange and ARQ | Not a default part of a generic custom-OFDM PHY example; the application must implement policy or adopt a standard-specific toolbox/model |
| Standards-compliant MAC and scheduler | No | Available for supported standards. For example, 5G Toolbox includes an NR scheduler at the gNB and exposes scheduling/MCS configuration |
| Adaptive MCS | Experimental controller integrated into the drone-link application | Supported in standards-oriented simulations and schedulers; custom link adaptation can also be modeled |
| Real SDR testing | Real Pi 5 + PlutoSDR path is maintained in the repository and has two-node OTA results | OTA testing is supported through SDR hardware support packages, including PlutoSDR |
| Embedded Pi 5 execution | Same Python PHY/MAC source runs on the host, with ARM-native acceleration where available | Possible through generated code and supported deployment workflows, but it is distinct from ordinary interactive MATLAB execution and may require additional products |
| FPGA path | Experimental handwritten/generated HLS/RTL using the Python implementation as golden model | Mature HDL generation, verification and hardware-ready block workflow through Wireless HDL Toolbox, HDL Coder and related products |
| Inspectability and modification | Complete application source and tests; MIT-licensed project code | Toolbox implementation details are proprietary; user models and generated outputs can be inspected within license terms |
| Reproducible automation | Normal Python packaging, pytest and command-line workflows; CI still needs strengthening | Strong scripts, apps, Simulink models and testing products; reproducibility depends on MATLAB release and licensed toolbox set |
| License model | Open-source core; no per-seat runtime dependency | Commercial per-user/product licensing; required products depend on workflow |
| Best fit | Small teams that want one modifiable Python implementation from algorithm experiment to inexpensive embedded OTA test | Teams needing a broad, supported modeling ecosystem, standards toolboxes, polished visualization, model-based design, or production HDL/code generation |

### What the MATLAB distinction actually is

It would be inaccurate to say MATLAB provides only isolated PHY blocks or no
FEC. Communications Toolbox explicitly supports end-to-end communications
simulation and OTA verification, and MathWorks publishes a complete OFDM
transmitter/receiver example containing convolutional coding and maximum-
likelihood decoding. [Communications Toolbox overview](https://www.mathworks.com/help/comm/)
and [complete OFDM transmitter/receiver example](https://www.mathworks.com/help/comm/ug/ofdm-transmitter-and-receiver.html).

It would also be inaccurate to say the full MATLAB ecosystem has no MAC or
adaptive MCS. The 5G Toolbox NR scheduler lives at the MAC layer and selects
transmission properties including MCS. [MathWorks NR scheduler
overview](https://www.mathworks.com/help/5g/ug/overview-nr-scheduler.html).

The defensible spectracuda advantage is narrower and practical:

> For a **non-standard, point-to-point OFDM application**, spectracuda already
> joins the custom PHY, independently swappable receiver algorithms, FEC
> composition, streaming receiver, simplified reliable MAC, experimental
> adaptive MCS, embedded ARM execution, and PlutoSDR transport in one open
> Python repository.

Achieving the analogous workflow in MATLAB can involve several products. For
example, Wireless HDL Toolbox requires MATLAB, Simulink, Communications
Toolbox, DSP HDL Toolbox, DSP System Toolbox, Fixed-Point Designer, and Signal
Processing Toolbox; HDL Coder and HDL Verifier are recommended for generation
and verification. [Official Wireless HDL Toolbox requirements](https://www.mathworks.com/support/requirements/wireless-hdl-toolbox.html).
In return, MathWorks provides a far more mature model-based FPGA workflow than
spectracuda's experimental HLS/RTL work. [Wireless HDL Toolbox
overview](https://www.mathworks.com/products/wireless-hdl.html).

## Summary

spectracuda is not the broadest DSP platform, the most complete network stack,
or the most mature FPGA toolchain. Its useful intersection is:

```text
custom OFDM algorithms
        + integrated framing/FEC/link behavior
        + rapid Python experimentation
        + low-cost Pi 5 + Pluto deployment
        + real-RF regression evidence
```

That intersection is the basis on which it should be evaluated and marketed.
The strongest next improvements are soft-output demapping/FEC, timing and
sampling-rate tracking, preamble confirmation, capture/replay regression data,
and a stable hardware-qualified waveform profile.
