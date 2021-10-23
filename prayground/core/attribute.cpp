#include "attribute.h"

namespace prayground {

Attributes::Attributes() 
{

}

// --------------------------------------------------------------------------------------
void Attributes::addBool(const std::string& name, std::unique_ptr<bool[]> values, int n)
{

}

void Attributes::addInt(const std::string& name, std::unique_ptr<int[]> values, int n)
{

}

void Attributes::addFloat(const std::string& name, std::unique_ptr<float[]> values, int n)
{

}

void Attributes::addFloat2(const std::string& name, std::unique_ptr<float2[]> values, int n)
{

}

void Attributes::addFloat3(const std::string& name, std::unique_ptr<float3[]> values, int n)
{

}

void Attributes::addFloat4(const std::string& name, std::unique_ptr<float4[]> values, int n)
{

}

void Attributes::addString(const std::string& name, std::unique_ptr<std::string[]> values, int n)
{

}

// --------------------------------------------------------------------------------------
const bool* Attributes::findBool(const std::string& name, int n)
{

}

const bool& Attributes::findOneBool(const std::string& name)
{

}

const int* Attributes::findInt(const std::string& name, int n)
{

}

const int& Attributes::findOneInt(const std::string& name)
{

}

const float* Attributes::findFloat(const std::string& name, int n)
{

}

const float& Attributes::findOneFloat(const std::string& name)
{

}

const float2* Attributes::findFloat2(const std::string& name, int n)
{

}

const float2& Attributes::findOneFloat2(const std::string& name)
{

}

const float3* Attributes::findFloat3(const std::string& name, int n)
{

}

const float3& Attributes::findOneFloat3(const std::string& name)
{

}

const float4* Attributes::findFloat4(const std::string& name, int n)
{

}

const float4& Attributes::findOneFloat4(const std::string& name)
{

}

const std::string& Attributes::findString(const std::string& name)
{

}

} // ::prayground