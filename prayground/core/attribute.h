#pragma once 

#include <memory>
#include <string>

namespace prayground {

class Attributes {
public:
    Attributes() {}
    
    template <typename T>
    void addAttrib(const std::string& name, std::unique_ptr<T[]> value, int n);
private:
};

template <typename T>
struct AttribItem {
    const std::string name;
    const std::unique_ptr<T[]> value;
    const int n;
};

} // ::prayground