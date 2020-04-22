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

namespace MyEA::String
{
    std::string System_Command(char const *const tmp_ptr_command_received)
    {
        FILE *tmp_ptr_file_command(popen(tmp_ptr_command_received, "r"));

        if(tmp_ptr_file_command == NULL)
        {
            MyEA::String::Error("An error has been triggered from the `popen(%s)` function.", tmp_ptr_command_received);

            return("");
        }

        char tmp_buffer[1024u];

        std::string tmp_output("");

        while(fgets(tmp_buffer, sizeof(tmp_buffer), tmp_ptr_file_command) != NULL) { tmp_output += tmp_buffer; }

        if(ferror(tmp_ptr_file_command) != 0)
        {
            MyEA::String::Error("An error has been triggered from the `fgets(%s, %zu, %s)` function.",
                                tmp_buffer,
                                sizeof(tmp_buffer),
                                tmp_ptr_command_received);

            return(tmp_output);
        }

        if(pclose(tmp_ptr_file_command) == -1)
        {
            MyEA::String::Error("An error has been triggered from the `pclose(%s)` function.", tmp_ptr_command_received);

            return(tmp_output);
        }

        return(tmp_output);
    }
}
