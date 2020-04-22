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

        if(     time_elapse_received <=    0.000'000'999) { tmp_string = std::to_string(time_elapse_received * 1e+9) + "ns"; } // nanoseconds
        else if(time_elapse_received <=    0.000'999    ) { tmp_string = MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_STRING_FORMAT::FIXED>(time_elapse_received * 1e+6, 3u) + "us"; } // microseconds μs
        else if(time_elapse_received <=    0.999        ) { tmp_string = MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_STRING_FORMAT::FIXED>(time_elapse_received * 1e+3, 3u) + "ms"; } // milliseconds
        else if(time_elapse_received <=   59.0          ) { tmp_string = MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_STRING_FORMAT::FIXED>(time_elapse_received, 3u) + "s"; } // seconds
        else if(time_elapse_received <= 3599.0          )
        {
            tmp_string =  std::to_string(static_cast<size_t>(floor(time_elapse_received / 60.0))) + "m:"; // minute
            tmp_string += std::to_string(static_cast<size_t>(time_elapse_received) % 60_zu) + "s:"; // second
            tmp_string += MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_STRING_FORMAT::FIXED>(time_elapse_received - floor(time_elapse_received), 3u) + "ms"; // milliseconds
        }
        else if(time_elapse_received <= 86'399.0)
        {
            double const tmp_minutes(static_cast<double>(static_cast<size_t>(time_elapse_received) % 3600_zu) / 60.0);

            tmp_string =  std::to_string(static_cast<size_t>(floor(time_elapse_received / 3600.0))) + "h:"; // hour
            tmp_string += std::to_string(static_cast<size_t>(floor(tmp_minutes))) + "m:"; // minute
            tmp_string += std::to_string(static_cast<size_t>(tmp_minutes) % 60_zu) + "s:"; // second
            tmp_string += MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_STRING_FORMAT::FIXED>(time_elapse_received - floor(time_elapse_received), 3u) + "ms"; // milliseconds
        }
        else { tmp_string = MyEA::String::To_string<double, MyEA::String::ENUM_TYPE_STRING_FORMAT::FIXED>(time_elapse_received, 3u) + "s"; } // second

        return(tmp_string);
    }
}
