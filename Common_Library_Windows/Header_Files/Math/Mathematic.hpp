#pragma once

namespace MyEA
{
    namespace Math
    {
    #if defined(COMPILE_ADEPT)
        template<typename T> static T PI = T(3.14159265358979323846264338327950288419716939937510L);
    #else
        template<typename T> constexpr T PI = T(3.14159265358979323846264338327950288419716939937510L);
    #endif
    }
}

#if defined(COMPILE_CUDA)
    #include <Math/CUDA_Mathematic.cuh>
#else
    #include <Math/Mathematic_cpp.hpp>
#endif
