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

#include "pch.hpp"

// Standard.
#include <iostream>

// This.
#include <Configuration/Configuration.hpp>

void string_cat(char *destination_received,
                size_t const sizeof_received,
                char const *const source_received)
{
#if defined(__CUDA_ARCH__) == false
    strcat_s(destination_received, sizeof_received, source_received);
#else
    strcat(destination_received, source_received);
#endif
}

void string_copy(char *destination_received,
                 size_t const sizeof_received,
                 char const *const source_received)
{
#if defined(__CUDA_ARCH__) == false
    strcpy_s(destination_received, sizeof_received, source_received);
#else
    strcpy(destination_received, source_received);
#endif
}

size_t Get__Total_System_Memory(void)
{
    MEMORYSTATUSEX tmp_MEMORYSTATUSEX;

    tmp_MEMORYSTATUSEX.dwLength = sizeof(MEMORYSTATUSEX);
    
    GlobalMemoryStatusEx(&tmp_MEMORYSTATUSEX);

    return(static_cast<size_t>(tmp_MEMORYSTATUSEX.ullTotalPhys));
}

size_t Get__Available_System_Memory(void)
{
    MEMORYSTATUSEX tmp_MEMORYSTATUSEX;

    tmp_MEMORYSTATUSEX.dwLength = sizeof(MEMORYSTATUSEX);
    
    GlobalMemoryStatusEx(&tmp_MEMORYSTATUSEX);

    return(static_cast<size_t>(tmp_MEMORYSTATUSEX.ullAvailPhys));
}

void PAUSE_TERMINAL(void) { system("PAUSE"); }