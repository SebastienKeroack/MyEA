#include "stdafx.hpp"

// This.
#include <Capturing/Keyboard/Windows/Keyboard.hpp>

namespace MyEA::Capturing
{
    Keyboard::Keyboard(void) { }

    bool Keyboard::Trigger_Key(short const key_code)
    {
        if(GetAsyncKeyState(key_code) != 0) { return(GetConsoleWindow() == GetForegroundWindow()); }

        return(false);
    }
}