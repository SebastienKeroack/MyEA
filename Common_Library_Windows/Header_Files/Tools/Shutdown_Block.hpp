#pragma once

#if defined(COMPILE_WINDOWS)
    #include <Tools/Shutdown_Block__Windows.hpp>
#elif defined(COMPILE_LINUX)
    #include <Tools/Shutdown_Block__Linux.hpp>
#endif