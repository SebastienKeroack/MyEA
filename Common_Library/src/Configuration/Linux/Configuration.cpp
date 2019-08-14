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