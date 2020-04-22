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