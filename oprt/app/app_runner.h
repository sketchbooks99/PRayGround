#pragma once 

#include "baseapp.h"
#include "window.h"

namespace oprt { 

float   oprtGetMouseX();
float   oprtGetMouseY();
float   oprtGetPreviousMouseX();
float   oprtGetpreviousMouseY();
float2  oprtGetMousePosition();
int32_t oprtGetMouseButton();
int32_t oprtGetWidth();
int32_t oprtGetHeight();
void oprtRunApp(const std::shared_ptr<BaseApp>& app, const std::shared_ptr<Window>& window);

class AppRunner
{
public:
    AppRunner(const std::shared_ptr<BaseApp>& app, const std::shared_ptr<Window>& window);

    void run() const;
    void loop() const;
    void close() const;

    std::shared_ptr<BaseApp> app() const;
    std::shared_ptr<Window> window() const;
private:
    std::shared_ptr<BaseApp> m_app;
    std::shared_ptr<Window> m_window;
};

}