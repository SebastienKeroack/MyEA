#pragma once

#if defined(COMPILE_WINDOWS)
    #include <Shutdown/Windows/Shutdown.hpp>
#elif defined(COMPILE_LINUX)
    #include <Shutdown/Linux/Shutdown.hpp>
#endif