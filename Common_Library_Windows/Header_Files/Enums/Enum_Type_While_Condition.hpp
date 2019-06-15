#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_WHILE_CONDITION
        {
            TYPE_WHILE_CONDITION_INFINITY = 0u,
            TYPE_WHILE_CONDITION_ITERATION = 1u,
            TYPE_WHILE_CONDITION_EXPIRATION = 2u,
            TYPE_WHILE_CONDITION_LENGTH = 3u
        };

        static std::map<enum ENUM_TYPE_WHILE_CONDITION, std::string> ENUM_TYPE_WHILE_CONDITION_NAMES = {{TYPE_WHILE_CONDITION_INFINITY, "Infinity"},
                                                                                                                                                                                   {TYPE_WHILE_CONDITION_ITERATION, "Iteration"},
                                                                                                                                                                                   {TYPE_WHILE_CONDITION_EXPIRATION, "Expiration"},
                                                                                                                                                                                   {TYPE_WHILE_CONDITION_LENGTH, "LENGTH"}};
    }
}
