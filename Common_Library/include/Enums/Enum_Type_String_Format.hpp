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