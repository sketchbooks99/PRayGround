#pragma once 

#include "../core/util.h"
#include "util.h"

/**
 * @todo
 * OptixInstanceとそれによるIASのビルドに対するラッパーの実装
 */

namespace oprt {

class Instance {
public: 
    Instance();
    void build(const Context& ctx, );

private:
    OptixInstance m_instance;
};

}
