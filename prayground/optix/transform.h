#pragma once 

#include <optix.h>

namespace prayground {

class Transform {
public:
    Transform();

    OptixTraversableHandle handle() const;
private:
    
};

} // ::prayground