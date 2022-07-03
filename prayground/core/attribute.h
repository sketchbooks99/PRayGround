#pragma once 

#include <memory>
#include <string>
#include <vector>
#include <prayground/math/vec.h>

namespace prayground {

    template <typename T> struct AttribItem;

    class Attributes {
    public:
        Attributes();
    
        void addBool(const std::string& name, std::unique_ptr<bool[]> values, int n);
        void addInt(const std::string& name, std::unique_ptr<int[]> values, int n);
        void addFloat(const std::string& name, std::unique_ptr<float[]> values, int n);
        void addVec2f(const std::string& name, std::unique_ptr<Vec2f[]> values, int n);
        void addVec3f(const std::string& name, std::unique_ptr<Vec3f[]> values, int n);
        void addVec4f(const std::string& name, std::unique_ptr<Vec4f[]> values, int n);
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
        const bool* findBool(const std::string& name, int* n) const;
        bool findOneBool(const std::string& name, const bool& d) const;
        const int* findInt(const std::string& name, int* n) const;
        int findOneInt(const std::string& name, const int& d) const;
        const float* findFloat(const std::string& name, int* n) const;
        float findOneFloat(const std::string& name, const float& d) const;
        const Vec2f* findVec2f(const std::string& name, int* n) const;
        Vec2f findOneVec2f(const std::string& name, const Vec2f& d) const;
        const Vec3f* findVec3f(const std::string& name, int* n) const;
        Vec3f findOneVec3f(const std::string& name, const Vec3f& d) const;
        const Vec4f* findVec4f(const std::string& name, int* n) const;
        Vec4f findOneVec4f(const std::string& name, const Vec4f& d) const;
        const std::string* findString(const std::string&, int* n) const;
        std::string findOneString(const std::string& name, const std::string& d) const;
    public:
        std::string name;
    private:
        std::vector<std::shared_ptr<AttribItem<bool>>> m_bools;
        std::vector<std::shared_ptr<AttribItem<int>>> m_ints;
        std::vector<std::shared_ptr<AttribItem<float>>> m_floats;
        std::vector<std::shared_ptr<AttribItem<Vec2f>>> m_Vec2fs;
        std::vector<std::shared_ptr<AttribItem<Vec3f>>> m_Vec3fs;
        std::vector<std::shared_ptr<AttribItem<Vec4f>>> m_Vec4fs;
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

} // namespace prayground