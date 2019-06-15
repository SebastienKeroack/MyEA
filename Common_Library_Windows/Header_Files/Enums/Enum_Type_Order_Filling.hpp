#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_ORDER_FILLING
        {
            ORDER_FILLING_NONE = 0,
            ORDER_FILLING_FOK = 1,
            ORDER_FILLING_IOC = 2,
            ORDER_FILLING_RETURN = 3
        };

        static std::map<enum ENUM_TYPE_ORDER_FILLING, std::string> ENUM_TYPE_ORDER_FILLING_NAMES = {{ORDER_FILLING_NONE, "ORDER_FILLING_NONE"},
                                                                                                                                                                            {ORDER_FILLING_FOK, "ORDER_FILLING_FOK"},
                                                                                                                                                                            {ORDER_FILLING_IOC, "ORDER_FILLING_IOC"},
                                                                                                                                                                            {ORDER_FILLING_RETURN, "ORDER_FILLING_RETURN"}};
    }
}