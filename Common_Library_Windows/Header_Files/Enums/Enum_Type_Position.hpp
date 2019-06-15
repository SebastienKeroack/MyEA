#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_POSITION
        {
            TYPE_POSITION_NONE = 0,
            TYPE_POSITION_BUY = 1,
            TYPE_POSITION_SELL = 2
        };

        static std::map<enum ENUM_TYPE_POSITION, std::string> ENUM_TYPE_POSITION_NAMES = {{TYPE_POSITION_NONE, "TYPE_POSITION_NONE"},
                                                                                                                                                        {TYPE_POSITION_BUY, "TYPE_POSITION_BUY"},
                                                                                                                                                        {TYPE_POSITION_SELL, "TYPE_POSITION_SELL"}};
    }
}