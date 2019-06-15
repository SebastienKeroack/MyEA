#pragma once

#if defined(COMPILE_WINDOWS)
    #include <Tools/Key_Logger__Windows.hpp>
#elif defined(COMPILE_LINUX)
    #include <Tools/Key_Logger__Linux.hpp>
#endif