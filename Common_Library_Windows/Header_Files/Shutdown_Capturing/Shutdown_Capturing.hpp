#pragma once

#if defined(COMPILE_WINDOWS)
    #include <Shutdown_Capturing/Windows/Shutdown_Capturing.hpp>
#elif defined(COMPILE_LINUX)
    #include <Shutdown_Capturing/Linux/Shutdown_Capturing.hpp>
#endif