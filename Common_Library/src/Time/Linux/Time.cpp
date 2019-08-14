#include "pch.hpp"

// Standard.
#include <chrono>
#include <iomanip>

// This.
#include <Time/Time.hpp>

namespace MyEA::Time
{
    std::string Date_Time_Acc_Now(bool const use_local_time_received)
    {
        // Get actual system time.
        std::chrono::time_point const tmp_now(std::chrono::system_clock::now());

        // Get seconds since 1970/1/1 00:00:00 UTC.
        std::time_t const tmp_now_utc(std::chrono::system_clock::to_time_t(tmp_now));

        // Get time_point from `tmp_now_utc` (note: no milliseconds).
        std::chrono::time_point const tmp_now_rounded(std::chrono::system_clock::from_time_t(tmp_now_utc));

        // Get milliseconds (difference between `tmp_now` and `tmp_now_rounded`).
        int const tmp_milliseconds(static_cast<int>(std::chrono::duration<double, std::milli>(tmp_now - tmp_now_rounded).count()));

        std::ostringstream tmp_ostringstream;
        
        std::string tmp_time_format;

        struct tm *tmp_ptr_tm;
        
        if(use_local_time_received)
        {
            tmp_time_format = "LC:%Y/%m/%d %T";

            tmp_ptr_tm = localtime(&tmp_now_utc);
        }
        else
        {
            tmp_time_format = "GM:%Y/%m/%d %T";

            tmp_ptr_tm = gmtime   (&tmp_now_utc);
        }

        tmp_ostringstream << std::put_time(tmp_ptr_tm, tmp_time_format.c_str());

        // Return ("[Y/MW/D H:M:S.xxx]").
        return("[" + tmp_ostringstream.str() + std::string('.' + std::to_string(tmp_milliseconds)) + "]");
    }

    std::string Now_Format(std::string format_received, bool const use_local_time_received)
    {
        BOOST_ASSERT_MSG(format_received.empty() == false, "`format_received` is an empty string.");

        std::ostringstream tmp_ostringstream;

        std::string tmp_time_format;

        time_t tmp_time_t(std::time(nullptr));

        struct tm *tmp_ptr_tm;
        
        if(use_local_time_received)
        {
            tmp_time_format = "LC:" + format_received;

            tmp_ptr_tm = localtime(&tmp_time_t);
        }
        else
        {
            tmp_time_format = "GM:" + format_received;

            tmp_ptr_tm = gmtime   (&tmp_time_t);
        }

        tmp_ostringstream << std::put_time(tmp_ptr_tm, tmp_time_format.c_str());

        return(tmp_ostringstream.str());
    }
}
