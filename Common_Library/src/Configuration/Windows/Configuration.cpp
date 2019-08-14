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