#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_SIGNAL_QUALITY
        {
            TYPE_SQ_NONE = 0,
            TYPE_SQ_LOW = 1,
            TYPE_SQ_MED = 2,
            TYPE_SQ_HIGH = 3
        };

        static std::map<enum ENUM_TYPE_SIGNAL_QUALITY, std::string> ENUM_TYPE_SIGNAL_QUALITY_NAMES = {{TYPE_SQ_NONE, "TYPE_SQ_NONE"},
                                                                                                                                                                                {TYPE_SQ_LOW, "TYPE_SQ_LOW"},
                                                                                                                                                                                {TYPE_SQ_MED, "TYPE_SQ_MED"},
                                                                                                                                                                                {TYPE_SQ_HIGH, "TYPE_SQ_HIGH"}};
    }
}