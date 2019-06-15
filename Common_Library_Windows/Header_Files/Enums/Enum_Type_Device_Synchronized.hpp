#pragma once

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