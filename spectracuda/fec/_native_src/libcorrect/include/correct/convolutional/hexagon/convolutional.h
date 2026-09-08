#include "correct/convolutional/convolutional.h"
// BIG HEAPING TODO sort out the include mess (same TODO the SSE/NEON
// headers left themselves -- see sse/convolutional.h, neon/convolutional.h)
#include "correct-hexagon.h"
// hexagon_types.h / hvx_hexagon_protos.h / hexagon_protos.h: the
// Hexagon SDK headers exposing HVX intrinsics (HVX_Vector, Q6_* ops --
// see reference/qc6490/Hexagon_DSP_programming/06-hvx-example/ for the
// include pattern this mirrors). NOT present in this repo or on any
// machine this project has built on -- only available inside the real
// Hexagon SDK (see docs/hexagon-fec-offload-plan.md). This #include
// will fail to resolve until that SDK is installed and its include
// path is passed to hexagon-clang; that is expected and correct, not a
// bug to work around.
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/* Same reasoning as correct_convolutional_neon (neon/convolutional.h's
 * own comment): the Hexagon ACS inner loop (see hexagon/decode.c)
 * reuses pair_lookup_t (conv->pair_lookup, already built by the
 * portable _convolutional_decode_init()) unchanged -- no HVX-specific
 * per-instance state needed beyond the portable correct_convolutional
 * itself. */
struct correct_convolutional_hexagon {
    correct_convolutional base_conv;
};
