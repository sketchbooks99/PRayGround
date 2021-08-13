#include <vector_functions.h>
#include <vector_types.h>
#include "../optix/macros.h"

#ifndef __CUDACC_RTC__
#include <cmath>
#include <cstdlib>
#endif

namespace oprt {
namespace constants{

constexpr double PI = 3.14159265358979323846f;
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double EPS = 1e-10f;

} // ::constant
} // ::oprt