#include "stdafx.hpp"

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

    bool Retrieve_Temporary_File(std::string const &ref_path_received)
    {
        std::string tmp_path_temporary(ref_path_received + ".tmp");

        if(MyEA::File::Path_Exist(tmp_path_temporary))
        {
            if(std::experimental::filesystem::v1::equivalent(ref_path_received, tmp_path_temporary) == false
               &&
               std::experimental::filesystem::v1::copy_file(tmp_path_temporary,
                                                            ref_path_received,
                                                            std::experimental::filesystem::v1::copy_options::overwrite_existing) == false)
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

    bool Write_Temporary_File(std::string const &ref_path_received)
    {
        std::string tmp_path_temporary(ref_path_received + ".tmp");

        if(MyEA::File::Path_Exist(ref_path_received))
        {
            if(MyEA::File::Path_Exist(tmp_path_temporary) && std::experimental::filesystem::v1::equivalent(ref_path_received, tmp_path_temporary)) { return(true); }

            return(std::experimental::filesystem::v1::copy_file(ref_path_received,
                                                                tmp_path_temporary,
                                                                std::experimental::filesystem::v1::copy_options::overwrite_existing));
        }

        return(true);
    }
}
