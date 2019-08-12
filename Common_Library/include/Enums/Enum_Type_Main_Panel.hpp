#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_MAIN_PANEL
        {
            TYPE_M_P_NONE = 0,
            TYPE_M_P_MENU = 1,
            TYPE_M_P_OUTPUT = 2,
            TYPE_M_P_SIGNAL = 3,
            TYPE_M_P_ACCOUNT = 4,
            TYPE_M_P_TRADE = 5
        };

        static std::map<enum ENUM_TYPE_MAIN_PANEL, std::string> ENUM_TYPE_MAIN_PANEL_NAMES = {{TYPE_M_P_NONE, "TYPE_M_P_NONE"},
                                                                                                                                                                  {TYPE_M_P_MENU, "TYPE_M_P_MENU"},
                                                                                                                                                                  {TYPE_M_P_OUTPUT, "TYPE_M_P_OUTPUT"},
                                                                                                                                                                  {TYPE_M_P_SIGNAL, "TYPE_M_P_SIGNAL"},
                                                                                                                                                                  {TYPE_M_P_ACCOUNT, "TYPE_M_P_ACCOUNT"},
                                                                                                                                                                  {TYPE_M_P_TRADE, "TYPE_M_P_TRADE"}};
    }
}