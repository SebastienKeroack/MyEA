#pragma once

#if defined(COMPILE_CUDA)
    #include <Tools/CUDA_Reallocate.cuh>
#else
    #include <Tools/Reallocate_cpp.hpp>
#endif

#if defined(COMPILE_ADEPT)
    #include <Tools/Configuration.hpp>
    
    // Adept memset compatibility.
    #define MEMSET(x, v, sz) Memory::Fill<T_>(x, x + (sz / sizeof(T_)), 0_T)
    
    // Adept memcpy compatibility.
    #define MEMCPY(d, s, sz) Memory::Copy<T_>(s, s + (sz / sizeof(T_)), d)
#else
    // Adept memset compatibility.
    #define MEMSET(x, v, sz) memset(x, v, sz)
    
    // Adept memcpy compatibility.
    #define MEMCPY(d, s, sz) memcpy(d, s, sz)
#endif
