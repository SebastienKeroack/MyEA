#include "stdafx.hpp"

#include <Tools/Configuration.hpp>
#include <Strings/String.hpp>
#include <Math/Math.hpp>

size_t Get__Remaining_Available_System_Memory(long double const reserved_bytes_percent_received, size_t const maximum_reserved_bytes_received)
{
    if(reserved_bytes_percent_received > 100.0L)
    {
        PRINT_FORMAT("%s: %s: ERROR: Number of reserved bytes in percent is greater than 100%%." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__);

        return(0_zu);
    }
    else if(reserved_bytes_percent_received < 0.0L)
    {
        PRINT_FORMAT("%s: %s: ERROR: Number of reserved bytes in percent is less than 0%%." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__);

        return(0_zu);
    }
    
    size_t const tmp_available_memory(Get__Available_System_Memory());
    size_t tmp_reserved_memory(static_cast<size_t>(floor(static_cast<long double>(Get__Total_System_Memory()) * reserved_bytes_percent_received / 100.0L)));

    tmp_reserved_memory = MyEA::Math::Minimum<size_t>(tmp_reserved_memory, maximum_reserved_bytes_received);

    return(tmp_reserved_memory > tmp_available_memory ? 0_zu : tmp_available_memory - tmp_reserved_memory);
}