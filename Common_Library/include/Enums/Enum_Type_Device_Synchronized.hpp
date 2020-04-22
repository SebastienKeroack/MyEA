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
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_DEVICE_SYNCHRONIZED : unsigned int
        {
            TYPE_DEVICE_SYNCHRONIZED_NONE = 0u,
            TYPE_DEVICE_SYNCHRONIZED_THREAD = 1u,
            TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK = 2u
        };

        static std::map<enum ENUM_TYPE_DEVICE_SYNCHRONIZED, std::string> ENUM_TYPE_DEVICE_SYNCHRONIZED_NAMES = {{TYPE_DEVICE_SYNCHRONIZED_NONE, "TYPE_DEVICE_SYNCHRONIZED_NONE"},
                                                                                                                {TYPE_DEVICE_SYNCHRONIZED_THREAD, "TYPE_DEVICE_SYNCHRONIZED_THREAD"},
                                                                                                                {TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK, "TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK"}};
    }
}