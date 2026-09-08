#!/usr/bin/env bash
# One-shot setup for decoder="aff3ct": clones + builds AFF3CT itself
# (reference/aff3ct/, ~850MB checked out + built, several minutes on a
# typical machine -- NOT run automatically by anything, and deliberately
# not a pip/setup.py dependency: AFF3CT is a large external project,
# "reference, not shipped" the same way the rest of reference/ is, and
# LDPC.decoder="aff3ct" is an opt-in fast path, not a normal runtime
# requirement -- see fec/_native_aff3ct.py's own docstring), then builds
# this directory's bridge_ldpc against it.
#
# Idempotent: safe to re-run -- skips the clone if reference/aff3ct/
# already exists, skips the cmake configure if build/ already exists,
# `make` itself no-ops on an up-to-date build.
#
# Usage: spectracuda/fec/_native_src/aff3ct_bridge/setup_aff3ct.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
AFF3CT_DIR="$REPO_ROOT/reference/aff3ct"

if [ ! -d "$AFF3CT_DIR" ]; then
    echo "== cloning AFF3CT (recursive -- 6 submodules, ~850MB checked out) =="
    mkdir -p "$REPO_ROOT/reference"
    git clone --recursive https://github.com/aff3ct/aff3ct.git "$AFF3CT_DIR"
else
    echo "== reference/aff3ct/ already exists, skipping clone =="
    # Cheap safety net for a partial/interrupted earlier clone -- a real
    # checkout always has its submodules populated.
    git -C "$AFF3CT_DIR" submodule update --init --recursive
fi

if [ ! -d "$AFF3CT_DIR/build" ]; then
    echo "== configuring AFF3CT (Release, static lib) =="
    mkdir -p "$AFF3CT_DIR/build"
    (cd "$AFF3CT_DIR/build" && cmake .. -DCMAKE_BUILD_TYPE=Release -DAFF3CT_COMPILE_STATIC_LIB=ON)
else
    echo "== reference/aff3ct/build/ already configured, skipping cmake =="
fi

echo "== building AFF3CT (this is the slow part -- several minutes, hundreds of translation units) =="
NPROC="$(command -v nproc >/dev/null && nproc || echo 4)"
(cd "$AFF3CT_DIR/build" && make -j"$NPROC")

echo "== building bridge_ldpc =="
"$HERE/build.sh"

echo "== done -- LDPCCode(variant, decoder='aff3ct') is now usable =="
