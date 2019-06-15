#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_SIGNAL_PATTERN
        {
            TYPE_PATTERN_NONE = 0,
            TYPE_PATTERN_LONG_OPEN = 1,
            TYPE_PATTERN_SHORT_OPEN = 2,
            TYPE_PATTERN_LONG_OPEN_REVERSE = 3,
            TYPE_PATTERN_SHORT_OPEN_REVERSE = 4,
            TYPE_PATTERN_LONG_CLOSE = 5,
            TYPE_PATTERN_SHORT_CLOSE = 6,
            TYPE_PATTERN_LONG_CLOSE_REVERSE = 7,
            TYPE_PATTERN_SHORT_CLOSE_REVERSE = 8
        };

        static std::map<enum ENUM_TYPE_SIGNAL_PATTERN, std::string> ENUM_TYPE_SIGNAL_PATTERN_NAMES = {{TYPE_PATTERN_NONE, "TYPE_PATTERN_NONE"},
                                                                                                                                                                                {TYPE_PATTERN_LONG_OPEN, "TYPE_PATTERN_LONG_OPEN"},
                                                                                                                                                                                {TYPE_PATTERN_SHORT_OPEN, "TYPE_PATTERN_SHORT_OPEN"},
                                                                                                                                                                                {TYPE_PATTERN_LONG_OPEN_REVERSE, "TYPE_PATTERN_LONG_OPEN_REVERSE"},
                                                                                                                                                                                {TYPE_PATTERN_SHORT_OPEN_REVERSE, "TYPE_PATTERN_SHORT_OPEN_REVERSE"},
                                                                                                                                                                                {TYPE_PATTERN_LONG_CLOSE, "TYPE_PATTERN_LONG_CLOSE"},
                                                                                                                                                                                {TYPE_PATTERN_SHORT_CLOSE, "TYPE_PATTERN_SHORT_CLOSE"},
                                                                                                                                                                                {TYPE_PATTERN_LONG_CLOSE_REVERSE, "TYPE_PATTERN_LONG_CLOSE_REVERSE"},
                                                                                                                                                                                {TYPE_PATTERN_SHORT_CLOSE_REVERSE, "TYPE_PATTERN_SHORT_CLOSE_REVERSE"}};
    }
}