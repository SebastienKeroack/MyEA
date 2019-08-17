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
    
    #define fError(text) Write_File("ERROR.log", \
                                    std::string(__FILE__) + ":" + std::to_string(__LINE__) + ", ERROR: " + text, \
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