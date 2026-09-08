#!/usr/bin/env bash
# Builds bridge_ldpc against the already-built AFF3CT tree, reusing the
# EXACT compile flags/defines/includes AFF3CT's own CMake build recorded
# for aff3ct-obj (build/CMakeFiles/aff3ct-obj.dir/flags.make) -- not
# guessed -- and linking against the same object files aff3ct-bin itself
# links (extracted from build/CMakeFiles/aff3ct-bin.dir/link.txt), minus
# src/main.cpp.o (that's AFF3CT's own main(), which we're replacing with
# ours -- everything else in that object list is the actual library code:
# factories, LDPC decoder variants, Sparse_matrix/QC parsing, streampu
# Module/Task runtime, etc.)
set -euo pipefail

# This file lives at spectracuda/fec/_native_src/aff3ct_bridge/ (alongside
# libcorrect/ and hexagon/ -- the common home for every FEC backend's
# native code), four levels below the repo root; the actual AFF3CT source
# + build tree it links against lives at reference/aff3ct/ (see that
# directory's own "reference, not shipped" status -- AFF3CT is a large
# external project, never vendored into the shipped package the way
# libcorrect is).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ROOT="$REPO_ROOT/reference/aff3ct"
BUILD="$ROOT/build"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FLAGS_MAKE="$BUILD/CMakeFiles/aff3ct-obj.dir/flags.make"
LINK_TXT="$BUILD/CMakeFiles/aff3ct-bin.dir/link.txt"
[ -f "$FLAGS_MAKE" ] || { echo "missing $FLAGS_MAKE -- has aff3ct been built?"; exit 1; }
[ -f "$LINK_TXT" ]   || { echo "missing $LINK_TXT -- has aff3ct-bin been built?"; exit 1; }

CXX_DEFINES=$(grep '^CXX_DEFINES' "$FLAGS_MAKE" | cut -d= -f2-)
CXX_INCLUDES=$(grep '^CXX_INCLUDES' "$FLAGS_MAKE" | cut -d= -f2-)
CXX_FLAGS=$(grep '^CXX_FLAGS' "$FLAGS_MAKE" | cut -d= -f2-)

echo "== compiling bridge_ldpc.cpp =="
# shellcheck disable=SC2086
/usr/bin/c++ $CXX_DEFINES $CXX_INCLUDES $CXX_FLAGS -c "$HERE/bridge_ldpc.cpp" -o "$HERE/bridge_ldpc.o"

echo "== extracting aff3ct-bin's own object list (minus its main.cpp.o) =="
# link.txt is one long line: "<cxx> <flags> <all-the-.o-files-quoted> -o bin/aff3ct-... <trailing-libs>"
# Pull out every "...cpp.o" token, drop the one that's AFF3CT's own main(),
# and keep paths relative to $BUILD (that's what link.txt itself uses).
mapfile -t OBJS < <(grep -o '"[^"]*\.cpp\.o"' "$LINK_TXT" | tr -d '"' | grep -v '/main\.cpp\.o$')
echo "  (${#OBJS[@]} object files, aff3ct's own main.cpp.o excluded)"

echo "== linking bridge_ldpc =="
cd "$BUILD"
/usr/bin/c++ -O3 -DNDEBUG \
    "$HERE/bridge_ldpc.o" \
    "${OBJS[@]}" \
    -o "$HERE/bridge_ldpc" \
    lib/streampu/lib/cpptrace/lib/libcpptrace.a -ldl -lpthread

echo "== done: $HERE/bridge_ldpc =="
