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
#include <errno.h>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace MyEA::File
{
    bool Directory_Create(std::string const &ref_path_received)
    {
        if(mkdir(ref_path_received.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) { return(errno == EEXIST); }

        return(true);
    }

    std::vector<std::pair<std::string, std::string>> Get__List_Drives(void)
    {
        std::vector<std::pair<std::string, std::string>> tmp_drives;

        FILE *tmp_ptr_file_command(popen("cat /proc/mounts | grep /dev/sd", "r"));

        if(tmp_ptr_file_command == NULL)
        {
            MyEA::String::Error("The command `cat /proc/mounts | grep /dev/sd` could not be execute.");

            return(tmp_drives);
        }

        char tmp_buffers[1024u];

        std::stringstream tmp_cat_stream("", std::ios_base::ate | std::ios_base::in | std::ios_base::out);

        while(fgets(tmp_buffers, sizeof(tmp_buffers), tmp_ptr_file_command) != NULL) { tmp_cat_stream << tmp_buffers; }

        pclose(tmp_ptr_file_command);

        size_t tmp_found_drive_at,
               tmp_found_mount_point_at;

        std::string tmp_line,
                    tmp_drive,
                    tmp_mount_point;

        while(std::getline(tmp_cat_stream, tmp_line))
        {
            if(tmp_line.find("/dev/sd") != std::string::npos)
            {
                tmp_found_drive_at = tmp_line.find_first_of(" ");
                tmp_drive = tmp_line.substr(5_zu, tmp_found_drive_at - 5_zu);

                tmp_found_mount_point_at = tmp_line.find_first_of(" ", tmp_found_drive_at + 1_zu);
                tmp_mount_point = tmp_line.substr(tmp_found_drive_at + 1_zu, tmp_found_mount_point_at - tmp_found_drive_at - 1_zu);

                if(tmp_mount_point == "/"
                   &&
                   MyEA::File::Path_Exist(tmp_mount_point + "home/" + std::string(getlogin())))
                { tmp_mount_point += "home/" + std::string(getlogin()); }

                tmp_drives.push_back(std::pair<std::string, std::string>(tmp_drive, tmp_mount_point));
            }
        }

        return(tmp_drives);
    }
}
