#include "attribute.h"
#include <optional>

namespace prayground {

#define FIND_PTR(items)                 \
    for (const auto& item : items)      \
    {                                   \
        if (item->name == name)         \
        {                               \
            *n = item->numValues;       \
            return item->values.get();  \
        }                               \
    }                                   \
    return nullptr;

#define FIND_ONE(items)                                 \
    for (const auto& item : items)                      \
    {                                                   \
        if (item->name == name && item->numValues == 1) \
            return *(item->values.get());               \
    }                                                   \
    return d;                              

// Attributes
// --------------------------------------------------------------------------------------
Attributes::Attributes() 
{

}

// --------------------------------------------------------------------------------------
void Attributes::addBool(const std::string& name, std::unique_ptr<bool[]> values, int n)
{
    m_bools.emplace_back(new AttribItem<bool>(name, std::move(values), n));
}

void Attributes::addInt(const std::string& name, std::unique_ptr<int[]> values, int n)
{
    m_ints.emplace_back(new AttribItem<int>(name, std::move(values), n));
}   

void Attributes::addFloat(const std::string& name, std::unique_ptr<float[]> values, int n)
{
    m_floats.emplace_back(new AttribItem<float>(name, std::move(values), n));
}

void Attributes::addFloat2(const std::string& name, std::unique_ptr<float2[]> values, int n)
{
    m_float2s.emplace_back(new AttribItem<float2>(name, std::move(values), n));
}

void Attributes::addFloat3(const std::string& name, std::unique_ptr<float3[]> values, int n)
{
    m_float3s.emplace_back(new AttribItem<float3>(name, std::move(values), n));
}

void Attributes::addFloat4(const std::string& name, std::unique_ptr<float4[]> values, int n)
{
    m_float4s.emplace_back(new AttribItem<float4>(name, std::move(values), n));
}

void Attributes::addString(const std::string& name, std::unique_ptr<std::string[]> values, int n)
{
    m_strings.emplace_back(new AttribItem<std::string>(name, std::move(values), n));
}

// --------------------------------------------------------------------------------------
const bool* Attributes::findBool(const std::string& name, int* n) const
{
    FIND_PTR(m_bools);
}

bool Attributes::findOneBool(const std::string& name, const bool& d) const
{
    FIND_ONE(m_bools);
}

const int* Attributes::findInt(const std::string& name, int* n) const
{
    FIND_PTR(m_ints);
}

int Attributes::findOneInt(const std::string& name, const int& d) const
{
    FIND_ONE(m_ints);
}

const float* Attributes::findFloat(const std::string& name, int* n) const
{
    FIND_PTR(m_floats);
}

float Attributes::findOneFloat(const std::string& name, const float& d) const
{
    FIND_ONE(m_floats);
}

const float2* Attributes::findFloat2(const std::string& name, int* n) const
{
    FIND_PTR(m_float2s);
}

float2 Attributes::findOneFloat2(const std::string& name, const float2& d) const
{
    FIND_ONE(m_float2s)
}

const float3* Attributes::findFloat3(const std::string& name, int* n) const
{
    FIND_PTR(m_float3s);
}

float3 Attributes::findOneFloat3(const std::string& name, const float3& d) const
{
    FIND_ONE(m_float3s);
}

const float4* Attributes::findFloat4(const std::string& name, int* n) const
{
    FIND_PTR(m_float4s);
}

float4 Attributes::findOneFloat4(const std::string& name, const float4& d) const
{
    FIND_ONE(m_float4s);
}

const std::string* Attributes::findString(const std::string& name, int* n) const
{
    FIND_PTR(m_strings);
}

std::string Attributes::findOneString(const std::string& name, const std::string& d) const
{
    FIND_ONE(m_strings);
}

} // ::prayground