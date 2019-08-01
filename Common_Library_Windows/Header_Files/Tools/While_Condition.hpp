#pragma once

#include <Enums/Enum_Type_While_Condition.hpp>

#include <chrono>

namespace MyEA::Common
{
    struct While_Condition
    {
        While_Condition(void) { }

        enum MyEA::Common::ENUM_TYPE_WHILE_CONDITION type_while_condition = MyEA::Common::ENUM_TYPE_WHILE_CONDITION::TYPE_WHILE_CONDITION_ITERATION;

        union
        {
            unsigned long long maximum_iterations = 1ull;

            std::chrono::system_clock::time_point expiration;
        };
    };
}
