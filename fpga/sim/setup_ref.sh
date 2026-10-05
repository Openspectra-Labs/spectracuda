#!/bin/bash
# Create (or check) the pinned spectracuda worktree the RTL harness
# imports. See golden_ref.py for why it is pinned.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"  # fpga/
REF=$(python3 -c "import sys; sys.path.insert(0, '$HERE/sim'); import golden_ref; print(golden_ref.REF_COMMIT)")
DIR="$HERE/refs/spectracuda_ref_ad0a396"

if [ -d "$DIR/spectracuda" ]; then
    HEAD=$(git -C "$DIR" rev-parse --short=7 HEAD)
    if [ "$HEAD" = "$REF" ]; then echo "ok: $DIR at $REF"; exit 0; fi
    echo "$DIR is at $HEAD, moving to $REF"
    git -C "$DIR" checkout -q --detach "$REF"
else
    mkdir -p "$HERE/refs"
    git -C "$HERE" worktree add -q --detach "$DIR" "$REF"
fi
echo "ok: $DIR at $(git -C "$DIR" rev-parse --short=7 HEAD)"
