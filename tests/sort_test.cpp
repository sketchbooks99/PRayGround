#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

enum class Type {
    A = 0,
    B = 1
};

struct Base {
    virtual Type type() const = 0;
    virtual void print_members() const = 0;
};

struct DerivedA : public Base {
    DerivedA(const float a, const std::string& name) : m_a(a), m_name(name) {}

    Type type() const override { return Type::A; }
    void print_members() const override { 
        std::cout << "DerivedA: " << m_a << ' ' << m_name << std::endl;
    }

    float m_a;
    std::string m_name;
};

struct DerivedB : public Base {
    DerivedB(const float a, const double b, const std::string& name)
    : m_a(a), m_b(b), m_name(name) {}

    Type type() const override { return Type::B; }
    void print_members() const override {
        std::cout << "DerivedB: " << m_a << ' ' << m_b << ' ' << m_name << std::endl;
    }

    float m_a;
    double m_b;
    std::string m_name;
};

int main() {
    std::vector<Base*> bases;
    for (int i=0; i<10; i++) {
        // std::cout << (double)rand() / RAND_MAX << std::endl;
        if ((double)rand() / RAND_MAX < 0.5) 
            bases.push_back(new DerivedA((float)i, "derivedA" + std::to_string(i)));
        else 
            bases.push_back(new DerivedB((float)i, (double)i, "derivedB" + std::to_string(i)));
    }

    std::cout << "===== Before sort =====" << std::endl;
    for (auto& b : bases) b->print_members();

    std::sort(bases.begin(), bases.end(), [](Base* b1, Base* b2){ return (int)b1->type() < (int)b2->type(); });

    std::cout << "===== After sort =====" << std::endl;
    for (auto& b : bases) b->print_members();

    return 0;
}