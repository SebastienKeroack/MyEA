#pragma once

// Standard.
#include <string>

namespace MyEA::Time
{
    void Sleep(unsigned int const milliseconds_received);

#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    constexpr
    bool _USE_LOCAL_TIME(void) { return(true); }
#else
    constexpr
    bool _USE_LOCAL_TIME(void) { return(false); }
#endif // _DEBUG || COMPILE_DEBUG

    std::string Date_Time_Acc_Now(bool const use_local_time_received = _USE_LOCAL_TIME());

    std::string Date_Time_Now(bool const use_local_time_received = _USE_LOCAL_TIME());

    std::string Date_Now(bool const use_local_time_received = _USE_LOCAL_TIME());

    std::string Now_Format(std::string format_received = "", bool const use_local_time_received = _USE_LOCAL_TIME());

    std::string Time_Elapsed_Format(double const time_elapse_received);

}