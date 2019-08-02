#pragma once

#if defined(COMPILE_CUDA)
    #include <Reallocate/CUDA/Reallocate.cuh>
#else
    #include <Reallocate/CPU/Reallocate.hpp>
#endif

#if defined(COMPILE_ADEPT)
    #include <Configuration/Configuration.hpp>
    
    // Adept memset compatibility.
    #define MEMSET(x, v, sz) Memory::Fill<T_>(x, x + (sz / sizeof(T_)), 0_T)
    
    // Adept memcpy compatibility.
    #define MEMCPY(d, s, sz) Memory::Copy<T_>(s, s + (sz / sizeof(T_)), d)
#else
    #define MEMSET(x, v, sz) memset(x, v, sz)
    
    #define MEMCPY(d, s, sz) memcpy(d, s, sz)
#endif
