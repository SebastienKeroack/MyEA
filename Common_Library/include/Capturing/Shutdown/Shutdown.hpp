#pragma once

#if defined(COMPILE_WINDOWS)
    #include <Capturing/Shutdown/Windows/Shutdown.hpp>
#elif defined(COMPILE_LINUX)
    #include <Capturing/Shutdown/Linux/Shutdown.hpp>
#endif