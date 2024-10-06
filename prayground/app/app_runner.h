#pragma once 

#include "baseapp.h"
#include "window.h"

namespace prayground { 

    float       pgGetMouseX();
    float       pgGetMouseY();
    float       pgGetPreviousMouseX();
    float       pgGetPreviousMouseY();
    Vec2f       pgGetMousePosition();
    Vec2f       pgGetPreviousMousePosition();
    int32_t     pgGetMouseButton();
    int32_t     pgGetKey();
    int32_t     pgGetWidth();
    int32_t     pgGetHeight();
    int32_t     pgGetFrame();
    float       pgGetFrameRate();
    void        pgSetFrameRate(const float fps);
    float       pgGetElapsedTimef();
    std::shared_ptr<Window> pgGetCurrentWindow();
    void        pgSetWindowName(const std::string& name);
    void        pgRunApp(const std::shared_ptr<BaseApp>& app, const std::shared_ptr<Window>& window, bool use_window=true);
    bool        pgAppWindowInitialized();
    void        pgSetAppName(const std::string& name);
    std::string pgGetAppName();
    void        pgExit();


    class AppRunner
    {
    public:
        AppRunner(const std::shared_ptr<BaseApp>& app, const std::shared_ptr<Window>& window, bool use_window=true);

        void run() const;
        void loop() const;
        void close() const;

        std::shared_ptr<BaseApp> app() const;
        std::shared_ptr<Window> window() const;
    private:
        std::shared_ptr<BaseApp> m_app;
        std::shared_ptr<Window> m_window;
        bool m_use_window;
    };

} // namespace prayground