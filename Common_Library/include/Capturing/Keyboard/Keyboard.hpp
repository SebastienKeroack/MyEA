#pragma once

#if defined(COMPILE_WINDOWS)
    #include <Capturing/Keyboard/Windows/Keyboard.hpp>
#elif defined(COMPILE_LINUX)
    #include <Capturing/Keyboard/Linux/Keyboard.hpp>
#endif