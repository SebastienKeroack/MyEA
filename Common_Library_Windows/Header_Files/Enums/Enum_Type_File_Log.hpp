#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_FILE_LOG : unsigned int
        {
            TYPE_FILE_LOG = 0u,
            TYPE_FILE_ERROR = 1u,
            TYPE_FILE_DEBUG = 2u
        };

        static std::map<enum ENUM_TYPE_FILE_LOG, std::string> ENUM_TYPE_FILE_LOG_NAMES = {{TYPE_FILE_LOG, "TYPE_FILE_LOG"},
                                                                                                                                                        {TYPE_FILE_ERROR, "TYPE_FILE_ERROR"},
                                                                                                                                                        {TYPE_FILE_DEBUG, "TYPE_FILE_DEBUG"}};
    }
}