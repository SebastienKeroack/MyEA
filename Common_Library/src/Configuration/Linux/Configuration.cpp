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
#include <unistd.h>
#include <iostream>

// This.
#include <Configuration/Configuration.hpp>
#include <Strings/String.hpp>

void string_cat(char *destination_received,
                size_t const sizeof_received,
                char const *const source_received)
{
    strcat(destination_received, source_received);
}

void string_copy(char *destination_received,
                 size_t const sizeof_received,
                 char const *const source_received)
{
    strcpy(destination_received, source_received);
}
    
size_t Get__Total_System_Memory(void)
{
    long const tmp_pages(sysconf(_SC_PHYS_PAGES)),
               tmp_page_size(sysconf(_SC_PAGE_SIZE));

    return(static_cast<size_t>(tmp_pages) * static_cast<size_t>(tmp_page_size));
}

size_t Get__Available_System_Memory(void)
{
    long const tmp_pages(sysconf(_SC_AVPHYS_PAGES)),
               tmp_page_size(sysconf(_SC_PAGE_SIZE));

    return(static_cast<size_t>(tmp_pages) * static_cast<size_t>(tmp_page_size));
}

void PAUSE_TERMINAL(void) { std::cout << "Press \"ENTER\" key to exit." NEW_LINE; std::cin.get(); }