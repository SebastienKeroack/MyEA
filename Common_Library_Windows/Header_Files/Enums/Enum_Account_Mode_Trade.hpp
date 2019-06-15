#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_ACCOUNT_MODE_TRADE
        {
            ACCOUNT_MODE_TRADE_NONE = 0,
            ACCOUNT_MODE_TRADE_CONTEST = 1,
            ACCOUNT_MODE_TRADE_REAL = 2,
            ACCOUNT_MODE_TRADE_DEMO = 3
        };

        static std::map<enum ENUM_ACCOUNT_MODE_TRADE, std::string> ENUM_ACCOUNT_MODE_TRADE_NAMES = {{ACCOUNT_MODE_TRADE_NONE, "ACCOUNT_MODE_TRADE_NONE"},
                                                                                                                                                                                    {ACCOUNT_MODE_TRADE_CONTEST, "ACCOUNT_MODE_TRADE_CONTEST"},
                                                                                                                                                                                    {ACCOUNT_MODE_TRADE_REAL, "ACCOUNT_MODE_TRADE_REAL"},
                                                                                                                                                                                    {ACCOUNT_MODE_TRADE_DEMO, "ACCOUNT_MODE_TRADE_DEMO"}};
    }
}