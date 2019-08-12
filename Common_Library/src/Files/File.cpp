#include "stdafx.hpp"

#include <Time/Time.hpp>
#include <Strings/String.hpp>

#if defined(COMPILE_WINDOWS)
    #include <tchar.h>
#elif defined(COMPILE_LINUX)
    #include <errno.h>
    #include <cstring>
    #include <unistd.h>
#endif

#include <sys/stat.h>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <Files/File.hpp>

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

    bool Directory_Create(std::string const &ref_path_received)
#if defined(COMPILE_WINDOWS)
    {
        return((CreateDirectory(ref_path_received.c_str(), NULL) == 0 && GetLastError() == ERROR_PATH_NOT_FOUND) ? false : true);
    }
#elif defined(COMPILE_LINUX)
    {
        if(mkdir(ref_path_received.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) { return(errno == EEXIST); }

        return(true);
    }
#endif

    bool Directory_Remove(std::string const &ref_path_received)
    {
        return(std::remove(ref_path_received.c_str()) == 0);
    }

    std::vector<std::pair<std::string, std::string>> Get__List_Drives(void)
    {
        std::vector<std::pair<std::string, std::string>> tmp_drives;

    #if defined(COMPILE_WINDOWS)
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
    #elif defined(COMPILE_LINUX)
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
    #endif // COMPILE_WINDOWS || COMPILE_LINUX

        return(tmp_drives);
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

    bool Retrieve_Temporary_File(std::string const &ref_path_received)
    {
        std::string tmp_path_temporary(ref_path_received + ".tmp");

        if(MyEA::File::Path_Exist(tmp_path_temporary))
        {
        #if defined(COMPILE_WINDOWS)
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
        #elif defined(COMPILE_LINUX)
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
        #endif
        }

        return(true);
    }

    bool Write_Temporary_File(std::string const &ref_path_received)
    {
        std::string tmp_path_temporary(ref_path_received + ".tmp");

        if(MyEA::File::Path_Exist(ref_path_received))
        {
        #if defined(COMPILE_WINDOWS)
            if(MyEA::File::Path_Exist(tmp_path_temporary) && std::experimental::filesystem::v1::equivalent(ref_path_received, tmp_path_temporary)) { return(true); }

            return(std::experimental::filesystem::v1::copy_file(ref_path_received,
                                                                tmp_path_temporary,
                                                                std::experimental::filesystem::v1::copy_options::overwrite_existing));
        #elif defined(COMPILE_LINUX)
            if(MyEA::File::Path_Exist(tmp_path_temporary) && std::filesystem::equivalent(ref_path_received, tmp_path_temporary)) { return(true); }

            return(std::filesystem::copy_file(ref_path_received,
                                              tmp_path_temporary,
                                              std::filesystem::copy_options::overwrite_existing));
        #endif
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
