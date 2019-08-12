#include "stdafx.hpp"

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

    bool Delete_Temporary_File(std::string const &ref_path_received)
    {
        std::string tmp_path_temporary(ref_path_received + ".tmp");

        if(MyEA::File::Path_Exist(tmp_path_temporary)) { return(MyEA::File::File_Remove(tmp_path_temporary)); }

        return(true);
    }
}
