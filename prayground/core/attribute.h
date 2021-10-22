#pragma once 

#include <memory>
#include <string>
#include <vector>
#include <vector_types.h>

namespace prayground {

template <typename T> struct AttribItem;

class Attributes {
public:
    Attributes() {}
    
    void addFloat(const std::string& name, std::unique_ptr<float[]> values, int n);
    void addFloat2(const std::string& name, std::unique_ptr<float2[]> values, int n);
    void addFloat3(const std::string& name, std::unique_ptr<float3[]> values, int n);

private:
    std::vector<std::shared_ptr<AttribItem<bool>>> m_bools;
    std::vector<std::shared_ptr<AttribItem<float>>> m_floats;
    std::vector<std::shared_ptr<AttribItem<float2>>> m_float2s;
    std::vector<std::shared_ptr<AttribItem<float3>>> m_float3s;
    std::vector<std::shared_ptr<AttribItem<float4>>> m_float4s;
    std::vector<std::shared_ptr<AttribItem<std::string>>> m_strings;
    std::vector<std::shared_ptr<AttribItem<int>>> m_ints;
};

template <typename T>
struct AttribItem {
    const std::string name;
    const std::unique_ptr<T[]> values;
    const int n;
};

} // ::prayground