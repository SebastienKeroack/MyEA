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
#include <Files/File.hpp>
#include <Strings/String.hpp>
#include <Time/Time.hpp>

// Standard.
#include <sys/stat.h>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace MyEA::File
{
    bool Path_Exist(std::string const &ref_path_received)
    {
        struct stat tmp_buffer{};

        return(stat(ref_path_received.c_str(), &tmp_buffer) == 0);
    }

    bool File_Create(std::string const &ref_path_received)
    {
        std::ofstream tmp_file(ref_path_received, std::ofstream::out | std::ofstream::trunc);

        return(tmp_file.good());
    }

    bool File_Remove(std::string const &ref_path_received)
    {
        return(std::remove(ref_path_received.c_str()) == 0);
    }

    bool Directory_Remove(std::string const &ref_path_received)
    {
        return(std::remove(ref_path_received.c_str()) == 0);
    }

    std::string Get__Full_Path(std::string const &ref_path_received)
    {
        std::vector<std::pair<std::string, std::string>> tmp_drives(Get__List_Drives());

        std::string tmp_path;

        for(auto &tmp_drive : tmp_drives)
        {
            tmp_path += tmp_drive.second + ESCAPE_FILE + ref_path_received;

            if(MyEA::File::Path_Exist(tmp_path)) { return(tmp_path); }
        }

        return("");
    }

    bool Write_Print_File(std::string const &ref_path_received,
                          std::string const &ref_text_received,
                          int const mode_received)
    {
        std::cout << ref_text_received;

        return(MyEA::File::Write_File(ref_path_received,
                                      ref_text_received,
                                      mode_received));
    }

    bool Write_File(std::string const &ref_path_received,
                    std::string const &ref_text_received,
                    int const mode_received)
    {
        std::ofstream tmp_file(ref_path_received, static_cast<std::ios_base::openmode>(mode_received));

        if(tmp_file.is_open())
        {
            tmp_file.write(ref_text_received.c_str(), ref_text_received.size() * sizeof(char));

            tmp_file.flush();
            tmp_file.close();

            return(true);
        }

        return(false);
    }

    bool Write_Temporary_File(std::string const &ref_path_received)
    {
        std::string tmp_path_temporary(ref_path_received + ".tmp");

        if(MyEA::File::Path_Exist(ref_path_received))
        {
            if(MyEA::File::Path_Exist(tmp_path_temporary) && std::filesystem::equivalent(ref_path_received, tmp_path_temporary)) { return(true); }

            return(std::filesystem::copy_file(ref_path_received,
                                              tmp_path_temporary,
                                              std::filesystem::copy_options::overwrite_existing));
        }

        return(true);
    }
    
    bool Retrieve_Temporary_File(std::string const &ref_path_received)
    {
        std::string tmp_path_temporary(ref_path_received + ".tmp");

        if(MyEA::File::Path_Exist(tmp_path_temporary))
        {
            if(std::filesystem::equivalent(ref_path_received, tmp_path_temporary) == false
               &&
               std::filesystem::copy_file(tmp_path_temporary,
                                          ref_path_received,
                                          std::filesystem::copy_options::overwrite_existing) == false)
            {
                MyEA::String::Error("An error has been triggered from the `copy_file(%s, %s)` function.",
                                    tmp_path_temporary.c_str(),
                                    ref_path_received.c_str());

                return(false);
            }
            else if(MyEA::File::File_Remove(tmp_path_temporary) == false)
            {
                MyEA::String::Error("An error has been triggered from the `File_Remove(%s)` function.", tmp_path_temporary.c_str());

                return(false);
            }
        }

        return(true);
    }

    bool Delete_Temporary_File(std::string const &ref_path_received)
    {
        std::string tmp_path_temporary(ref_path_received + ".tmp");

        if(MyEA::File::Path_Exist(tmp_path_temporary)) { return(MyEA::File::File_Remove(tmp_path_temporary)); }

        return(true);
    }
}
