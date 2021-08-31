#pragma once

#include <prayground/core/shape.h>

namespace prayground {

#ifndef __CUDACC__
class ShapeGroup final : public Shape 
{
public:
    ShapeGroup(OptixBuildInputType buildInputType);
private:
    
};
#endif // __CUDACC__

} // ::prayground