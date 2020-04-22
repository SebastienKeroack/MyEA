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
#include <vector>
#include <functional>

// This.
#include <Enums/Enum_Type_File_Log.hpp>

namespace MyEA::File
{
    std::vector<std::pair<std::string, std::string>> Get__List_Drives(void);

    std::string Get__Full_Path(std::string const &ref_path_received);

    std::string Read_File_Info(const std::function<void(void)>& function_received);

    bool Write_Print_File(std::string const &ref_path_received,
                          std::string const &ref_text_received,
                          int const mode_received);

    bool Write_File(std::string const &ref_path_received,
                    std::string const &ref_text_received,
                    int const mode_received);
    
    // TODO: Add time.
    #define fError(text) Write_File("ERROR.log", \
                                    std::string(__FILE__) + ":" + std::to_string(__LINE__) + ", ERROR: " + text + NEW_LINE, \
                                    42)
    
    bool Retrieve_Temporary_File(std::string const &ref_path_received);

    bool Write_Temporary_File(std::string const &ref_path_received);

    bool Delete_Temporary_File(std::string const &ref_path_received);

    bool Path_Exist(std::string const &ref_path_received);

    bool File_Create(std::string const &ref_path_received);

    bool File_Remove(std::string const &ref_path_received);

    bool Directory_Create(std::string const &ref_path_received);

    bool Directory_Remove(std::string const &ref_path_received);
}