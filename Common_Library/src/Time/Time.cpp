#include "stdafx.hpp"

// Standard.
#include <thread>

// This.
#include <Strings/String.hpp>
#include <Time/Time.hpp>

namespace MyEA::Time
{
    void Sleep(unsigned int const milliseconds_received)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds_received));
    }

    std::string Date_Time_Now(bool const use_local_time_received)
    {
        return("[" + Now_Format("%d/%m/%Y %T", use_local_time_received) + "]");
    }

    std::string Date_Now(bool const use_local_time_received)
    {
        return("[" + Now_Format("%d/%m/%Y", use_local_time_received) + "]");
    }

    std::string Time_Elapsed_Format(double const time_elapse_received)
    {
        std::string tmp_string;

        if(     time_elapse_received <= 0.000'000'999) { tmp_string = std::to_string(time_elapse_received * 1e+9) + "ns"; } // nanoseconds
        else if(time_elapse_received <= 0.000'999    ) { tmp_string = MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_STRING_FORMAT::FIXED>(time_elapse_received * 1e+6, 3u) + "us"; } // microseconds μs
        else if(time_elapse_received <= 0.999        ) { tmp_string = MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_STRING_FORMAT::FIXED>(time_elapse_received * 1e+3, 3u) + "ms"; } // milliseconds
        else if(time_elapse_received <= 59.0         ) { tmp_string = MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_STRING_FORMAT::FIXED>(time_elapse_received, 3u) + "s"; } // seconds
        else if(time_elapse_received <= 3599.0       )
        {
            tmp_string =  std::to_string(static_cast<unsigned int>(floor(time_elapse_received / 60.0))) + "m:"; // minute
            tmp_string += std::to_string(static_cast<unsigned int>(time_elapse_received) % 60u) + "s:"; // second
            tmp_string += MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_STRING_FORMAT::FIXED>(time_elapse_received - floor(time_elapse_received), 3u) + "ms"; // milliseconds
        }
        else if(time_elapse_received <= 86'399.0)
        {
            double const tmp_minutes(static_cast<double>(static_cast<unsigned int>(time_elapse_received) % 3600u) / 60.0);

            tmp_string =  std::to_string(static_cast<unsigned int>(floor(time_elapse_received / 3600.0))) + "h:"; // hour
            tmp_string += std::to_string(static_cast<unsigned int>(floor(tmp_minutes))) + "m:"; // minute
            tmp_string += std::to_string(static_cast<unsigned int>(tmp_minutes) % 60u) + "s:"; // second
            tmp_string += MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_STRING_FORMAT::FIXED>(time_elapse_received - floor(time_elapse_received), 3u) + "ms"; // milliseconds
        }
        else { tmp_string = MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_STRING_FORMAT::FIXED>(time_elapse_received, 3u) + "s"; } // second

        return(tmp_string);
    }
}
