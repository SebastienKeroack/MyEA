#include "stdafx.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif

#include <Tools/Time.hpp>

#include <chrono>
#include <thread>

namespace MyEA
{
    namespace Time
    {
        void Sleep(unsigned int const milliseconds_received) { std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds_received)); }

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
                     ":0." + std::to_string(tmp_system_time.wMilliseconds) + "]");
        #elif defined(COMPILE_LINUX)
            std::time_t tmp_time(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
            struct std::tm *tmp_ptr_tm(localtime(&tmp_time));

            return("[" + std::to_string(tmp_ptr_tm->tm_year + 1900) +
                     "-" + std::to_string(tmp_ptr_tm->tm_mon + 1) +
                     "-" + std::to_string(tmp_ptr_tm->tm_mday) +
                     " " + std::to_string(tmp_ptr_tm->tm_hour) +
                     ":" + std::to_string(tmp_ptr_tm->tm_min) +
                     ":" + std::to_string(tmp_ptr_tm->tm_sec) + "]");
        #else
            return("");
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
        #else
            return("");
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
        #else
            return("");
        #endif
        }
    }
}
