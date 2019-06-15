#include "stdafx.hpp"
#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif

#include <Tools/Animation_Waiting.hpp>

#include <Preprocessing__Merge_Dataset.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Preprocessing__Merge_Dataset(void)
{
    std::string tmp_dataset_name;
    
    std::cout << MyEA::String::Get__Time() << ": Dataset name: ";

    getline(std::cin, tmp_dataset_name);
    
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_dataset_name + " - Preprocessing, Merge").c_str());
#endif
    
    class MyEA::Neural_Network::Neural_Network_Manager tmp_Neural_Network_Manager(true, MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iNONE);
    
    if(tmp_Neural_Network_Manager.Initialize_Path(tmp_dataset_name, tmp_dataset_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_dataset_name.c_str(),
                                 tmp_dataset_name.c_str(),
                                 __LINE__);

        return(false);
    }
    
    // Dataset Manager Parameters.
    struct Dataset_Manager_Parameters tmp_Dataset_Manager_Parameters;

    tmp_Dataset_Manager_Parameters.type_storage = 0;
    tmp_Dataset_Manager_Parameters.type_training = 0;
    
    if(tmp_Neural_Network_Manager.Initialize_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_Manager__OLD()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    // |END| Dataset Manager Parameters. |END|
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Dataset type." NEW_LINE, MyEA::String::Get__Time().c_str());
    for(unsigned int tmp_dataset_type_index(0u); tmp_dataset_type_index != MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_LENGTH; ++tmp_dataset_type_index)
    {
        PRINT_FORMAT("%s:\t[%u]: %s." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_dataset_type_index,
                                 MyEA::Common::ENUM_TYPE_DATASET_FILE_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_FILE>(tmp_dataset_type_index)].c_str());
    }
    PRINT_FORMAT("%s:\tdefault=%s." NEW_LINE,
                             MyEA::String::Get__Time().c_str(),
                             MyEA::Common::ENUM_TYPE_DATASET_FILE_NAMES[MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_DATASET_SPLIT].c_str());
    
    enum MyEA::Common::ENUM_TYPE_DATASET_FILE tmp_type_dataset_file;

    if((tmp_type_dataset_file = static_cast<enum MyEA::Common::ENUM_TYPE_DATASET_FILE>(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                                                                                                                                             MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_LENGTH - 1u,
                                                                                                                                                                                                             MyEA::String::Get__Time() + ": Type: "))) >= MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_LENGTH)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 1u,
                                 MyEA::Common::ENUM_TYPE_DATASET_FILE::TYPE_DATASET_FILE_LENGTH - 1u,
                                 __LINE__);

        return(false);
    }
    
    if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Set__Type_Data_File(tmp_type_dataset_file) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Type_Data_File(%u | %s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_type_dataset_file,
                                 MyEA::Common::ENUM_TYPE_DATASET_FILE_NAMES[tmp_type_dataset_file].c_str(),
                                 __LINE__);

        return(false);
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Saving into %s... ",
                             MyEA::String::Get__Time().c_str(),
                             tmp_Neural_Network_Manager.Get__Path_Dataset_Manager().c_str());
    class MyEA::Animation::Animation_Waiting tmp_Animation_Waiting;
    tmp_Animation_Waiting.Print_While_Async();

    if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Save(tmp_Neural_Network_Manager.Get__Path_Dataset_Manager()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_Neural_Network_Manager.Get__Path_Dataset_Manager().c_str(),
                                 __LINE__);

        return(false);
    }
    
    tmp_Animation_Waiting.Join();
    PRINT_FORMAT(NEW_LINE "%s" NEW_LINE, MyEA::String::Get__Time().c_str());

    return(true);
}
