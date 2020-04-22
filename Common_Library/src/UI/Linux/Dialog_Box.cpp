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
    std::string Which_Dialog_Box(void)
    {
        // If system can use system command.
        if(::system(NULL))
        {
            if(     ::system("which gdialog") == 0) { return("gdialog"); }
            else if(::system("which kdialog") == 0) { return("kdialog"); }
        }

        return("");
    }

    bool Dialog_Box__Accept(std::string const &ref_text_received, std::string const &ref_title_received)
    {
        static
        std::string const _dialog_command(Which_Dialog_Box());

        if(_dialog_command.empty() == false)
        {
            std::string const tmp_command(_dialog_command + " --title \""+ ref_title_received + "\" --yesno \"" + ref_text_received + "\"");

            int const tmp_result(::system(tmp_command.c_str()));

            if(     tmp_result == 0) { return(true ); }
            else if(tmp_result == 1) { return(false); }
        }

        return(false);
    }

    bool Dialog_Box__OK(std::string const &ref_text_received, std::string const &ref_title_received)
    {
        static
        std::string const _dialog_command(Which_Dialog_Box());

        if(_dialog_command.empty() == false)
        {
            std::string const tmp_command(_dialog_command + " --title \""+ ref_title_received + "\" --msgbox \"" + ref_text_received + "\"");

            int const tmp_result(::system(tmp_command.c_str()));

            if(tmp_result == 0) { return(true); }
        }

        return(false);
    }
}
