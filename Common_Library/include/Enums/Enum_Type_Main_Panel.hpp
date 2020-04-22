/* Copyright 2020 Sébastien Kéroack. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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