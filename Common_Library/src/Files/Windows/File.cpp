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
#include <tchar.h>
#include <sys/stat.h>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace MyEA::File
{
    bool Directory_Create(std::string const &ref_path_received)
    {
        return((CreateDirectory(ref_path_received.c_str(), NULL) == 0 && GetLastError() == ERROR_PATH_NOT_FOUND) ? false : true);
    }

    std::vector<std::pair<std::string, std::string>> Get__List_Drives(void)
    {
        std::vector<std::pair<std::string, std::string>> tmp_drives;

        TCHAR tmp_szDrive[](_T(" A:"));
        DWORD tmp_uDriveMask(GetLogicalDrives());

        std::string tmp_path;

        if(tmp_uDriveMask != 0)
        {
            while(tmp_uDriveMask)
            {
                // Use the bitwise AND, 1=available, 0=not available.
                if(tmp_uDriveMask & 1)
                {
                    tmp_path = tmp_szDrive;

                    tmp_path.erase(0, 1);

                    tmp_drives.push_back(std::pair<std::string, std::string>(tmp_path, tmp_path));
                }

                // Increment...
                ++tmp_szDrive[1];

                // Shift the bitmask binary right.
                tmp_uDriveMask >>= 1;
            }
        }
        else { MyEA::String::Error("An error has been triggered from the `GetLogicalDrives() -> %d` function.", GetLastError()); }

        return(tmp_drives);
    }
}
