#pragma once

#if defined(COMPILE_WINDOWS)
    #include <Keyboard_Capturing/Windows/Keyboard_Capturing.hpp>
#elif defined(COMPILE_LINUX)
    #include <Keyboard_Capturing/Linux/Keyboard_Capturing.hpp>
#endif