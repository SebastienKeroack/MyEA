#include "stdafx.hpp"

#include <Tools/Key_Logger__Windows.hpp>

Key_Logger::Key_Logger(void) { }

bool Key_Logger::Trigger_Key(short const key_code)
{
    if(GetAsyncKeyState(key_code) != 0) { return(GetConsoleWindow() == GetForegroundWindow()); }

    return(false);
}