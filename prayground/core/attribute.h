#pragma once 

#include <memory>
#include <string>
#include <vector>
#include <vector_types.h>

namespace prayground {

template <typename T> struct AttribItem;

class Attributes {
public:
    Attributes();
    
    void addBool(const std::string& name, std::unique_ptr<bool[]> values, int n);
    void addInt(const std::string& name, std::unique_ptr<int[]> values, int n);
    void addFloat(const std::string& name, std::unique_ptr<float[]> values, int n);
    void addFloat2(const std::string& name, std::unique_ptr<float2[]> values, int n);
    void addFloat3(const std::string& name, std::unique_ptr<float3[]> values, int n);
    void addFloat4(const std::string& name, std::unique_ptr<float4[]> values, int n);
    void addString(const std::string& name, std::unique_ptr<std::string[]> values, int n);

    /**
     * find~~(): 
     * - Return attributes with ptr.
     * - If attributes are not found, nullptr will be returned.
     * 
     * findOne~~():
     * - Return single value.
     * - Get arg of default value \c d, and if the attribute is not found, 
     *   default value will be returned.
     */ 
    const bool* findBool(const std::string& name, int* n);
    bool findOneBool(const std::string& name, const bool& d);
    const int* findInt(const std::string& name, int* n);
    int findOneInt(const std::string& name, const int& d);
    const float* findFloat(const std::string& name, int* n);
    float findOneFloat(const std::string& name, const float& d);
    const float2* findFloat2(const std::string& name, int* n);
    float2 findOneFloat2(const std::string& name, const float2& d);
    const float3* findFloat3(const std::string& name, int* n);
    float3 findOneFloat3(const std::string& name, const float3& d);
    const float4* findFloat4(const std::string& name, int* n);
    float4 findOneFloat4(const std::string& name, const float4& d);
    std::string findString(const std::string& name, const std::string& d);

public:
    std::string name;
private:
    std::vector<std::shared_ptr<AttribItem<bool>>> m_bools;
    std::vector<std::shared_ptr<AttribItem<int>>> m_ints;
    std::vector<std::shared_ptr<AttribItem<float>>> m_floats;
    std::vector<std::shared_ptr<AttribItem<float2>>> m_float2s;
    std::vector<std::shared_ptr<AttribItem<float3>>> m_float3s;
    std::vector<std::shared_ptr<AttribItem<float4>>> m_float4s;
    std::vector<std::shared_ptr<AttribItem<std::string>>> m_strings;
};

template <typename T>
struct AttribItem {
    AttribItem(const std::string& name, std::unique_ptr<T[]> values, int n)
    : name(name), values(std::move(values)), numValues(n) {}

    const std::string name;
    const std::unique_ptr<T[]> values;
    const int numValues;
};

} // ::prayground