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

// Standard.
#include <string>

// This.
#include <Configuration/Configuration.hpp>
#include <Enums/Enum_Type_Dialog_Box.hpp>

namespace MyEA::UI
{
    bool Dialog_Box__Accept(std::string const &ref_text_received, std::string const &ref_title_received);

    bool Dialog_Box__OK(std::string const &ref_text_received, std::string const &ref_title_received);

    bool Dialog_Box(ENUM_TYPE_DIALOG_BOX const type_received,
                    std::string const &ref_text_received,
                    std::string const &ref_title_received);
}

#define DEBUG_BOX(text_received) MyEA::UI::Dialog_Box(MyEA::UI::ENUM_TYPE_DIALOG_BOX::OK, text_received, "DEBUG");