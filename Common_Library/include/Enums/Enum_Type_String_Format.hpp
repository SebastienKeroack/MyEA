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

namespace MyEA::String
{
    enum ENUM_TYPE_STRING_FORMAT : unsigned int
    {
        FIXED        = 0u,
        SCIENTIFIC   = 1u,
        HEXFLOAT     = 2u,
        DEFAULTFLOAT = 3u,
        LENGTH       = 4u
    };

    static
    std::map<enum ENUM_TYPE_STRING_FORMAT, std::string> ENUM_TYPE_STRING_FORMAT_NAMES = {
            {FIXED,        "Fixed"},
            {SCIENTIFIC,   "Scientific"},
            {HEXFLOAT,     "Hex float"},
            {DEFAULTFLOAT, "Default float"},
            {LENGTH,       "Length"}
                                                                                        };
}