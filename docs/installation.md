# Installation

## Base install

```sh
pip install -e ".[dev]"
pytest
```

No GPU is required for development or the test suite — everything runs on
`backend="numpy"` by default when no working CUDA runtime is detected.
Install the `cuda` extra (`pip install -e ".[cuda]"`) on a CUDA-capable
machine (a Jetson, or any desktop CUDA GPU) to exercise `backend="cupy"`.
Every block also accepts `backend="numpy"|"cupy"` explicitly if you want
to pin it regardless of what's detected.

That's the whole install for everything covered in the Field Guide and
the landing page's hero example — Viterbi, Reed-Solomon, and the default
LDPC decoder all work out of the box, no extra download, no compiler
step you have to run yourself. The rest of this page is about the two
*optional* native-code accelerators underneath that default path — see
{doc}`fec-c-lib-acceleration` for the full design of both, this page is
just "what do I need to install, and when."

## Native Viterbi/Reed-Solomon acceleration — automatic, no download

`spectracuda/fec/_native_src/libcorrect/` (the C library behind
`conv_v27`'s and `rs_m8`'s native speedup) is **vendored source already
inside the repo** — nothing to clone, nothing to download. The first time
a `ConvolutionalCode`/`ReedSolomonCode` is constructed with
`backend="numpy"`, it's compiled on the spot (`cc`/`gcc`/`clang`, whatever
your system already has) into a small cached `.so` under
`~/.cache/spectracuda` (override with `SPECTRACUDA_CACHE_DIR`), reused on
every run after that — not recompiled per process.

```{list-table}
:header-rows: 1

* - If...
  - Then
* - a C compiler is available
  - Viterbi/RS silently run the compiled path — ~22x/~130x faster than pure Python, no code change, no flag
* - no compiler is found, or the compile fails for any reason
  - **silent, permanent fallback to pure Python/NumPy** — same output, just slower, same session (not retried on every construction)
```

Nothing to configure either way — this isn't a constructor argument, and
there's no size cost worth mentioning (the vendored source is a few
hundred KB; the compiled `.so` is smaller still).

## Optional: AFF3CT-accelerated LDPC decode — real download, opt-in only

LDPC (`fec="ldpc_648_r12"` … `"ldpc_1944_r56"`) has no native SIMD
acceleration of its own (see {doc}`fec-c-lib-acceleration` for why) — its
**default decoder is plain NumPy/CuPy**, exactly like every other block
in this project, and that default is what you get with zero extra setup:

```python
from spectracuda.fec.ldpc import LDPCCode

code = LDPCCode("ldpc_1944_r12")   # decoder="native" is the default -- nothing else to install
```

A real, much faster decoder exists as an **explicit opt-in**, bridging to
[AFF3CT](https://github.com/aff3ct/aff3ct) (MIT-licensed) — but getting
it requires a genuine, sizeable local build, not a pip install:

```{list-table}
:header-rows: 1

* -
  - Cost
* - Disk
  - **~850MB** — the full AFF3CT source checkout plus its own build output
* - Time
  - Several minutes (hundreds of C++ translation units)
* - Command
  - `spectracuda/fec/_native_src/aff3ct_bridge/setup_aff3ct.sh` (one shot, idempotent — safe to re-run, skips whatever's already done)
```

That script clones AFF3CT into `reference/aff3ct/` (gitignored — never
part of a normal `git clone` of this repo), builds it, then builds the
small bridge binary that talks to it. Once that's done:

```python
code = LDPCCode("ldpc_1944_r12", decoder="aff3ct")   # now available
```

**If you never run that script, nothing breaks** — `decoder="native"`
(the default, shown above) is what every part of this project uses unless
you explicitly ask for `"aff3ct"`. The one difference from the
Viterbi/RS case above: asking for `decoder="aff3ct"` without having built
it does **not** silently fall back to the native path — it raises
`Aff3ctUnavailable` with the exact setup command, on purpose (see
{doc}`fec-c-lib-acceleration`'s "Opt-in, fail loud, never silent
fallback" for why an *explicit* request getting silently substituted
would hide a real, actionable gap rather than surface it).

```{note}
None of this ~850MB download is needed for anything else in this
project — not the PHY chain, not the MAC layer, not `conv_v27`/`rs_m8`,
not the default LDPC path. It's purely for squeezing real decode
throughput out of LDPC specifically, which isn't even the FEC scheme
this project's own real-hardware work (see {doc}`hardware-validation`)
uses today — that's `rs_m8`+`conv_v27`, already fast via the automatic
native path above with no extra download at all.
```
