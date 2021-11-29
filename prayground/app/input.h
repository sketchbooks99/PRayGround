#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace prayground {

    enum class MouseButton : int
    {
        One     = GLFW_MOUSE_BUTTON_1,
        Two     = GLFW_MOUSE_BUTTON_2,
        Three   = GLFW_MOUSE_BUTTON_3,
        Four    = GLFW_MOUSE_BUTTON_4,
        Five    = GLFW_MOUSE_BUTTON_5,
        Six     = GLFW_MOUSE_BUTTON_6,
        Seven   = GLFW_MOUSE_BUTTON_7,
        Eight   = GLFW_MOUSE_BUTTON_8,

        Last    = GLFW_MOUSE_BUTTON_LAST,
        Left    = GLFW_MOUSE_BUTTON_LEFT,
        Right   = GLFW_MOUSE_BUTTON_RIGHT,
        Middle  = GLFW_MOUSE_BUTTON_MIDDLE
    };

    constexpr bool operator==(int b1, MouseButton b2)           { return b1 == static_cast<int>(b2); }
    constexpr bool operator==(MouseButton b1, int b2)           { return static_cast<int>(b1) == b2; }
    constexpr bool operator!=(int b1, MouseButton b2)           { return !(b1 == b2); }
    constexpr bool operator!=(MouseButton b1, int b2)           { return !(b1 == b2); }
    constexpr int  operator| (MouseButton b1, MouseButton b2)   { return static_cast<int>(b1) | static_cast<int>(b2); }
    constexpr int  operator| (int b1, MouseButton b2)           { return                   b1 | static_cast<int>(b2); }
    constexpr int  operator& (MouseButton b1, MouseButton b2)   { return static_cast<int>(b1) & static_cast<int>(b2); }
    constexpr int  operator& (int b1, MouseButton b2)           { return                   b1 & static_cast<int>(b2); }
    constexpr int  operator~ (MouseButton b1)                   { return ~static_cast<int>(b1); }
    constexpr int  operator+ (MouseButton b1)                   { return static_cast<int>(b1); }

    enum class Key : int
    {
        Unknown         = GLFW_KEY_UNKNOWN,

        Space           = GLFW_KEY_SPACE,
        Apostrophe      = GLFW_KEY_APOSTROPHE,
        Comma           = GLFW_KEY_COMMA,
        Period          = GLFW_KEY_PERIOD,
        Slash           = GLFW_KEY_SLASH,

        // Numbers
        Zero            = GLFW_KEY_0,
        One             = GLFW_KEY_1,
        Two             = GLFW_KEY_2,
        Three           = GLFW_KEY_3,
        Four            = GLFW_KEY_4,
        Five            = GLFW_KEY_5,
        Six             = GLFW_KEY_6,
        Seven           = GLFW_KEY_7,
        Eight           = GLFW_KEY_8,
        Nine            = GLFW_KEY_9,

        Semicolon       = GLFW_KEY_SEMICOLON,
        EQUAL           = GLFW_KEY_EQUAL,

        A               = GLFW_KEY_A,
        B               = GLFW_KEY_B,
        C               = GLFW_KEY_C,
        D               = GLFW_KEY_D,
        E               = GLFW_KEY_E,
        F               = GLFW_KEY_F,
        G               = GLFW_KEY_G,
        H               = GLFW_KEY_H,
        I               = GLFW_KEY_I,
        J               = GLFW_KEY_J,
        K               = GLFW_KEY_K,
        L               = GLFW_KEY_L,
        M               = GLFW_KEY_M,
        N               = GLFW_KEY_N,
        O               = GLFW_KEY_O,
        P               = GLFW_KEY_P,
        Q               = GLFW_KEY_Q,
        R               = GLFW_KEY_R,
        S               = GLFW_KEY_S,
        T               = GLFW_KEY_T,
        U               = GLFW_KEY_U,
        V               = GLFW_KEY_V,
        W               = GLFW_KEY_W,
        X               = GLFW_KEY_X,
        Y               = GLFW_KEY_Y,
        Z               = GLFW_KEY_Z,
        LeftBracket     = GLFW_KEY_LEFT_BRACKET,
        BackSlash       = GLFW_KEY_BACKSLASH,
        RightBracket    = GLFW_KEY_RIGHT_BRACKET,
        GraveAccent     = GLFW_KEY_GRAVE_ACCENT,
        World1          = GLFW_KEY_WORLD_1,
        World2          = GLFW_KEY_WORLD_2,

        // Function keys 
        Escape          = GLFW_KEY_ESCAPE,
        Enter           = GLFW_KEY_ENTER,
        Tab             = GLFW_KEY_TAB,
        Backspace       = GLFW_KEY_BACKSPACE,
        Insert          = GLFW_KEY_INSERT,
        Delete          = GLFW_KEY_DELETE,
        Right           = GLFW_KEY_RIGHT,
        Left            = GLFW_KEY_LEFT,
        Down            = GLFW_KEY_DOWN,
        Up              = GLFW_KEY_UP,
        PageUp          = GLFW_KEY_PAGE_UP,
        PageDown        = GLFW_KEY_PAGE_DOWN,
        Home            = GLFW_KEY_HOME,
        End             = GLFW_KEY_END,
        CapsLock        = GLFW_KEY_CAPS_LOCK,
        ScrollLock      = GLFW_KEY_SCROLL_LOCK,
        NumLock         = GLFW_KEY_NUM_LOCK,
        PrintScreen     = GLFW_KEY_PRINT_SCREEN,
        KeyPause        = GLFW_KEY_PAUSE,
        F1              = GLFW_KEY_F1,
        F2              = GLFW_KEY_F2,
        F3              = GLFW_KEY_F3,
        F4              = GLFW_KEY_F4,
        F5              = GLFW_KEY_F5,
        F6              = GLFW_KEY_F6,
        F7              = GLFW_KEY_F7,
        F8              = GLFW_KEY_F8,
        F9              = GLFW_KEY_F9,
        F10             = GLFW_KEY_F10,
        F11             = GLFW_KEY_F11,
        F12             = GLFW_KEY_F12,
        F13             = GLFW_KEY_F13,
        F14             = GLFW_KEY_F14,
        F15             = GLFW_KEY_F15,
        F16             = GLFW_KEY_F16,
        F17             = GLFW_KEY_F17,
        F18             = GLFW_KEY_F18,
        F19             = GLFW_KEY_F19,
        F20             = GLFW_KEY_F20,
        F21             = GLFW_KEY_F21,
        F22             = GLFW_KEY_F22,
        F23             = GLFW_KEY_F23,
        F24             = GLFW_KEY_F24,
        F25             = GLFW_KEY_F25,
        KP0             = GLFW_KEY_KP_0,
        KP1             = GLFW_KEY_KP_1,
        KP2             = GLFW_KEY_KP_2,
        KP3             = GLFW_KEY_KP_3,
        KP4             = GLFW_KEY_KP_4,
        KP5             = GLFW_KEY_KP_5,               
        KP6             = GLFW_KEY_KP_6,
        KP7             = GLFW_KEY_KP_7,
        KP8             = GLFW_KEY_KP_8,
        KP9             = GLFW_KEY_KP_9,
        KPDicimal       = GLFW_KEY_KP_DECIMAL,
        KPDivide        = GLFW_KEY_KP_DIVIDE,
        KPMultiply      = GLFW_KEY_KP_MULTIPLY,
        KPSubtract      = GLFW_KEY_KP_SUBTRACT,
        KPAdd           = GLFW_KEY_KP_ADD,
        KPEnter         = GLFW_KEY_KP_ENTER,
        KPEqual         = GLFW_KEY_KP_EQUAL,
        LeftShift       = GLFW_KEY_LEFT_SHIFT,
        LeftControl     = GLFW_KEY_LEFT_CONTROL,
        LeftAlt         = GLFW_KEY_LEFT_ALT,
        LeftSuper       = GLFW_KEY_LEFT_SUPER,
        RightShift      = GLFW_KEY_RIGHT_SHIFT,
        RightControl    = GLFW_KEY_RIGHT_CONTROL,
        RightAlt        = GLFW_KEY_RIGHT_ALT,
        RightSuper      = GLFW_KEY_RIGHT_SUPER,
        Menu            = GLFW_KEY_MENU,

        Last            = GLFW_KEY_LAST
    };

    constexpr bool operator==(int k1, Key k2) { return k1 == static_cast<int>(k2); }
    constexpr bool operator==(Key k1, int k2) { return static_cast<int>(k1) == k2; }
    constexpr bool operator!=(int k1, Key k2) { return !(k1 == k2); }
    constexpr bool operator!=(Key k1, int k2) { return !(k1 == k2); }
    constexpr int  operator| (Key k1, Key k2) { return static_cast<int>(k1) | static_cast<int>(k2); }
    constexpr int  operator| (int k1, Key k2) { return                   k1 | static_cast<int>(k2); }
    constexpr int  operator& (Key k1, Key k2) { return static_cast<int>(k1) & static_cast<int>(k2); }
    constexpr int  operator& (int k1, Key k2) { return                   k1 & static_cast<int>(k2); }
    constexpr int  operator~ (Key k1)         { return ~static_cast<int>(k1); }
    constexpr int  operator+ (Key k1)         { return static_cast<int>(k1); }
} // ::prayground