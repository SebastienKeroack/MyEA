#pragma once

namespace MyEA::Math
{
    template<typename T> constexpr
    T PI = T(3.14159265358979323846264338327950288419716939937510L);
}

#if defined(COMPILE_CUDA)
    #include <Math/CUDA/Math.cuh>
#else
    #include <Math/CPU/Math.hpp>
#endif
