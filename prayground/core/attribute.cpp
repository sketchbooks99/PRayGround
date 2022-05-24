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

void Attributes::addVec2f(const std::string& name, std::unique_ptr<Vec2f[]> values, int n)
{
    m_Vec2fs.emplace_back(new AttribItem<Vec2f>(name, std::move(values), n));
}

void Attributes::addVec3f(const std::string& name, std::unique_ptr<Vec3f[]> values, int n)
{
    m_Vec3fs.emplace_back(new AttribItem<Vec3f>(name, std::move(values), n));
}

void Attributes::addVec4f(const std::string& name, std::unique_ptr<Vec4f[]> values, int n)
{
    m_Vec4fs.emplace_back(new AttribItem<Vec4f>(name, std::move(values), n));
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

const Vec2f* Attributes::findVec2f(const std::string& name, int* n) const
{
    FIND_PTR(m_Vec2fs);
}

Vec2f Attributes::findOneVec2f(const std::string& name, const Vec2f& d) const
{
    FIND_ONE(m_Vec2fs)
}

const Vec3f* Attributes::findVec3f(const std::string& name, int* n) const
{
    FIND_PTR(m_Vec3fs);
}

Vec3f Attributes::findOneVec3f(const std::string& name, const Vec3f& d) const
{
    FIND_ONE(m_Vec3fs);
}

const Vec4f* Attributes::findVec4f(const std::string& name, int* n) const
{
    FIND_PTR(m_Vec4fs);
}

Vec4f Attributes::findOneVec4f(const std::string& name, const Vec4f& d) const
{
    FIND_ONE(m_Vec4fs);
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