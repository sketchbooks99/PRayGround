#pragma once 

#include <functional>
#include <vector>

namespace prayground
{

template <typename ReturnType, typename... Args>
class Event
{
public:
    void bindFunction(const std::function<ReturnType(Args...)>& function)
    {
        m_functions.push_back(function);
    }

    void invoke(const Args&... args)
    {
        for (const auto& func : m_functions)
            func(args...);
    }
private:
    std::vector<std::function<ReturnType(Args...)>> m_functions;
};

}