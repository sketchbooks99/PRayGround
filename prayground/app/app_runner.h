#pragma once 

#include "baseapp.h"
#include "window.h"

namespace prayground { 

float   pgGetMouseX();
float   pgGetMouseY();
float   pgGetPreviousMouseX();
float   pgGetPreviousMouseY();
float2  pgGetMousePosition();
float2  pgGetPreviousMousePosition();
int32_t pgGetMouseButton();
int32_t pgGetWidth();
int32_t pgGetHeight();
int32_t pgGetFrame();
float   pgGetFrameRate();
template <typename T> T pgGetElapsedTime();
void    pgSetWindowName(const std::string& name);
void    pgRunApp(const std::shared_ptr<BaseApp>& app, const std::shared_ptr<Window>& window);
void    pgExit();

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