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
#include <UI/Dialog_Box.hpp>

namespace MyEA::UI
{
    bool Dialog_Box__Accept(std::string const &ref_text_received, std::string const &ref_title_received)
    {
        int const tmp_message_box_ID = MessageBox(NULL,
                                                  ref_text_received.c_str(),
                                                  ref_title_received.c_str(),
                                                  MB_ICONINFORMATION | MB_YESNO);

        switch(tmp_message_box_ID)
        {
            case IDYES: return(true );
            case IDNO : return(false);
            default: return(false);
        }
    }

    bool Dialog_Box__OK(std::string const &ref_text_received, std::string const &ref_title_received)
    {
        int const tmp_message_box_ID = MessageBox(NULL,
                                                  ref_text_received.c_str(),
                                                  ref_title_received.c_str(),
                                                  MB_ICONINFORMATION | MB_OK);

        switch(tmp_message_box_ID)
        {
            case IDOK: return(true);
            default: return(true);
        }
    }
}
