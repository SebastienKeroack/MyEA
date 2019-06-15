#pragma once

#include <Enums/Enum_Type_File_Log.hpp>

#include <string>
#include <vector>
#include <functional>

#define CUDA_TOOLKITPATH_EXIST MyEA::File::Path_Exist(MyEA::File::Get__Full_Path("Program Files\\NVIDIA GPU Computing Toolkit")+"\\CUDA")

namespace MyEA
{
    namespace File
    {
        std::vector<std::pair<std::string, std::string>> Get__List_Drives(void);
        std::string Get__Full_Path(std::string const &ref_path_received);
        std::string Read_File_Info(const std::function<void(void)>& function_received);

        bool Write_Print_File(std::string const &path_received,
                                       std::string const &text_received,
                                       int const mode_received);
        bool Write_File(std::string const &path_received,
                               std::string const &text_received,
                               int const mode_received);
        bool Retrieve_Tempory_File(std::string const &path_received);
        bool Write_Tempory_File(std::string const &path_received);
        bool Delete_Tempory_File(std::string const &path_received);
        bool Path_Exist(const std::wstring& path_received);
        bool Path_Exist(std::string const &path_received);
        bool File_Create(std::string const &path_received);
        bool File_Remove(std::string const &path_received);
        bool Directory_Create(std::string const &path_received);
        bool Directory_Remove(std::string const &path_received);
    }
}