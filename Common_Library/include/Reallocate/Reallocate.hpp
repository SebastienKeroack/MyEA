/* Copyright 2020 Sébastien Kéroack. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#if defined(COMPILE_CUDA)
    #include <Reallocate/CUDA/Reallocate.cuh>
    #include <Reallocate/CUDA/Reallocate_C.cuh>
    #include <Reallocate/CUDA/Reallocate_Cpp.cuh>
#else
    #include <Reallocate/CPU/Reallocate.hpp>
    #include <Reallocate/CPU/Reallocate_C.hpp>
    #include <Reallocate/CPU/Reallocate_Cpp.hpp>
#endif

#if defined(COMPILE_AUTODIFF)
    #include <Configuration/Configuration.hpp>
    
    // Adept memset compatibility.
    #define MEMSET(x, v, sz) MyEA::Memory::Fill<T_>(x, x + (sz / sizeof(T_)), 0_T)
    
    // Adept memcpy compatibility.
    #define MEMCPY(d, s, sz) MyEA::Memory::Copy<T_>(s, s + (sz / sizeof(T_)), d)
#else
    #define MEMSET(x, v, sz) memset(x, v, sz)
    
    #define MEMCPY(d, s, sz) memcpy(d, s, sz)
#endif
