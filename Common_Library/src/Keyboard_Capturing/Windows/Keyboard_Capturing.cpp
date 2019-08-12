#include "stdafx.hpp"

#include <Keyboard_Capturing/Windows/Keyboard_Capturing.hpp>

Key_Logger::Key_Logger(void) { }

bool Key_Logger::Trigger_Key(short const key_code)
{
    if(GetAsyncKeyState(key_code) != 0) { return(GetConsoleWindow() == GetForegroundWindow()); }

    return(false);
}