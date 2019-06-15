#include "stdafx.hpp"
#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif

#include <Tools/Animation_Waiting.hpp>

#include <Preprocessing__Unrecurrent.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Preprocessing__Unrecurrent(void)
{
    std::string tmp_dataset_name;
    
    std::cout << MyEA::String::Get__Time() << ": Dataset name: ";

    getline(std::cin, tmp_dataset_name);
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_dataset_name + " - Preprocessing, Unrecurrent").c_str());
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
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_Manager()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    // |END| Dataset Manager Parameters. |END|
    
    if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Unrecurrent() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Unrecurrent()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(tmp_Neural_Network_Manager.Initialize_Path(tmp_dataset_name, tmp_dataset_name + "_UnR") == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s_UnR)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_dataset_name.c_str(),
                                 tmp_dataset_name.c_str(),
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
