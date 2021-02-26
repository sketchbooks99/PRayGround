#include <optix.h>

#include "pathtracer.h"
#include "random.h"

#include <sutil/vec_math.h>
#include "helpers.h"

extern "C" {
__constant__ Params params;
}

/** MEMO: 
 * Write intersection algorithm, right now! */

extern "C" __global__ void __intersection__sphere() {
    
}

extern "C" __global void __intersection__plane() {

}