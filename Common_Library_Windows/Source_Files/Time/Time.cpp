#include "stdafx.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif

#include <Time/Time.hpp>

#include <chrono>
#include <thread>

namespace MyEA::Time
{
    void Sleep(unsigned int const milliseconds_received)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds_received));
    }

    std::string Get__DateTimeFull(void)
    {
    #if defined(COMPILE_WINDOWS)
        SYSTEMTIME tmp_system_time;

        GetSystemTime(&tmp_system_time);

        return("[" + std::to_string(tmp_system_time.wYear) +
               "-" + std::to_string(tmp_system_time.wMonth) +
               "-" + std::to_string(tmp_system_time.wDay) +
               " " + std::to_string(tmp_system_time.wHour) +
               ":" + std::to_string(tmp_system_time.wMinute) +
               ":" + std::to_string(tmp_system_time.wSecond) +
               "." + std::to_string(tmp_system_time.wMilliseconds) + "]");
    #elif defined(COMPILE_LINUX)
        // Get actual system time.
        std::chrono::time_point const tmp_now(std::chrono::system_clock::now());

        // Get seconds since 1970/1/1 00:00:00 UTC.
        std::time_t const tmp_now_utc(std::chrono::system_clock::to_time_t(tmp_now));

        // Get time_point from `tmp_now_utc` (note: no milliseconds).
        std::chrono::time_point const tmp_now_rounded(std::chrono::system_clock::from_time_t(tmp_now_utc));

        // Get milliseconds (difference between `tmp_now` and `tmp_now_rounded`).
        int const tmp_milliseconds(std::chrono::duration<double, std::milli>(tmp_now - tmp_now_rounded).count());

        // Initialize datetime buffer.
        char tmp_datetime[20];

        // Get datetime formatted to "Y-MW-D H:M:S".
        std::strftime(tmp_datetime, 20, "%F %T", std::localtime(&tmp_now_utc));

        // Return ("[Y-MW-D H:M:S.xxx]").
        return("[" + std::string(tmp_datetime) + std::string('.' + std::to_string(tmp_milliseconds)) + "]");
    #endif
    }

    std::string Get__DateTimeStandard(void)
    {
    #if defined(COMPILE_WINDOWS)
        SYSTEMTIME tmp_system_time;

        GetSystemTime(&tmp_system_time);

        return("[" + std::to_string(tmp_system_time.wYear) +
               "-" + std::to_string(tmp_system_time.wMonth) +
               "-" + std::to_string(tmp_system_time.wDay) +
               " " + std::to_string(tmp_system_time.wHour) +
               ":" + std::to_string(tmp_system_time.wMinute) +
               ":" + std::to_string(tmp_system_time.wSecond) + "]");
    #elif defined(COMPILE_LINUX)
        std::time_t tmp_time(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
        struct std::tm *tmp_ptr_tm(localtime(&tmp_time));

        return("[" + std::to_string(tmp_ptr_tm->tm_year + 1900) +
               "-" + std::to_string(tmp_ptr_tm->tm_mon + 1) +
               "-" + std::to_string(tmp_ptr_tm->tm_mday) +
               " " + std::to_string(tmp_ptr_tm->tm_hour) +
               ":" + std::to_string(tmp_ptr_tm->tm_min) +
               ":" + std::to_string(tmp_ptr_tm->tm_sec) + "]");
    #endif
    }

    std::string Get__DateTimeMinimal(void)
    {
    #if defined(COMPILE_WINDOWS)
        SYSTEMTIME tmp_system_time;

        GetSystemTime(&tmp_system_time);

        return("[" + std::to_string(tmp_system_time.wYear) +
               "-" + std::to_string(tmp_system_time.wMonth) +
               "-" + std::to_string(tmp_system_time.wDay) + "]");
    #elif defined(COMPILE_LINUX)
        std::time_t tmp_time(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

        struct std::tm *tmp_ptr_tm(localtime(&tmp_time));

        return("[" + std::to_string(tmp_ptr_tm->tm_year + 1900) +
               "-" + std::to_string(tmp_ptr_tm->tm_mon + 1) +
               "-" + std::to_string(tmp_ptr_tm->tm_mday) + "]");
    #endif
    }
}
