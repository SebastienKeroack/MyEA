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

// This.
#include <Configuration/Configuration.hpp>

// Forward declaration.
namespace MyEA::Memory::Cpp
{
    template<class T>
    void Fill_Nullptr(T *ptr_array_received, T const *const ptr_array_last_received);

    template<class T, bool CPY = true, bool SET = true>
    T* Reallocate(T *ptr_array_received,
                  size_t const new_size_received,
                  size_t const old_size_received);

    template<class T, bool CPY = true>
    T* Reallocate_Objects(T *ptr_array_received,
                          size_t const new_size_received,
                          size_t const old_size_received);

    template<class T, bool CPY = true, bool SET = true>
    T* Reallocate_PtOfPt(T *ptr_array_received,
                         size_t const new_size_received,
                         size_t const old_size_received);
}

// This.
#include <../src/Reallocate/CPU/Reallocate_Cpp.cpp>
