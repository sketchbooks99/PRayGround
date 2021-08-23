#pragma once 

#include <functional>

namespace prayground
{

template <typename ReturnType, typename... Args>
class Event
{
public:
    void bindFunction(const std::function<ReturnType(Args...)>& function)
    {
        m_function = function;
    }

    void invoke(const Args&... args)
    {
        m_function(args...);
    }
private:
    std::function<ReturnType(Args...)> m_function;
};

}