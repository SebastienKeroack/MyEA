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

#include "pch.hpp"

// This.
#include <Strings/String.hpp>
#include <UI/Dialog_Box.hpp>

namespace MyEA::UI
{
    bool Dialog_Box(ENUM_TYPE_DIALOG_BOX const type_received,
                    std::string const &ref_text_received,
                    std::string const &ref_title_received)
    {
        switch(type_received)
        {
            case ENUM_TYPE_DIALOG_BOX::ACCEPT: return(Dialog_Box__Accept(ref_text_received, ref_title_received));
            case ENUM_TYPE_DIALOG_BOX::OK    : return(Dialog_Box__OK    (ref_text_received, ref_title_received));
            default:
                MyEA::String::Error("The `%s` dialog box type is not supported in the switch.", ENUM_TYPE_DIALOG_BOX_NAMES[type_received].c_str());
                    return(false);
        }
    }
}
