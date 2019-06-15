#include "stdafx.hpp"
#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif

#include <Tools/Animation_Waiting.hpp>

#include <Preprocessing__Concat.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Preprocessing__Concat(void)
{
    std::string tmp_dataset_name_A,
                    tmp_dataset_name_B;
    
    std::cout << MyEA::String::Get__Time() << ": Dataset A name: ";

    getline(std::cin, tmp_dataset_name_A);

    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

    std::cout << MyEA::String::Get__Time() << ": Dataset B name: ";

    getline(std::cin, tmp_dataset_name_B);
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());

#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_dataset_name_A + " / " + tmp_dataset_name_B + " - Preprocessing, Concat").c_str());
#endif
    
    class MyEA::Neural_Network::Neural_Network_Manager tmp_Neural_Network_Manager_A(true, MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iNONE),
                                                                                      tmp_Neural_Network_Manager_B(true, MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iNONE);
    
    if(tmp_Neural_Network_Manager_A.Initialize_Path(tmp_dataset_name_A, tmp_dataset_name_A) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_dataset_name_A.c_str(),
                                 tmp_dataset_name_A.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(tmp_Neural_Network_Manager_B.Initialize_Path(tmp_dataset_name_B, tmp_dataset_name_B) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_dataset_name_B.c_str(),
                                 tmp_dataset_name_B.c_str(),
                                 __LINE__);

        return(false);
    }
    
    // Dataset Manager Parameters.
    struct Dataset_Manager_Parameters tmp_Dataset_Manager_Parameters;

    tmp_Dataset_Manager_Parameters.type_storage = 0;
    tmp_Dataset_Manager_Parameters.type_training = 0;
    
    if(tmp_Neural_Network_Manager_A.Initialize_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_Manager()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
        
        return(false);
    }
    else if(tmp_Neural_Network_Manager_B.Initialize_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_Manager()\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    // |END| Dataset Manager Parameters. |END|
    
    if(tmp_Neural_Network_Manager_A.Get__Dataset_Manager()->Concat(tmp_Neural_Network_Manager_B.Get__Dataset_Manager()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Concat(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    if(tmp_Neural_Network_Manager_A.Initialize_Path(tmp_dataset_name_A, tmp_dataset_name_A + "_CONCAT") == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s_CONCAT)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_dataset_name_A.c_str(),
                                 tmp_dataset_name_A.c_str(),
                                 __LINE__);

        return(false);
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
    PRINT_FORMAT("%s: Saving into %s... ",
                             MyEA::String::Get__Time().c_str(),
                             tmp_Neural_Network_Manager_A.Get__Path_Dataset_Manager().c_str());
    class MyEA::Animation::Animation_Waiting tmp_Animation_Waiting;
    tmp_Animation_Waiting.Print_While_Async();

    if(tmp_Neural_Network_Manager_A.Get__Dataset_Manager()->Save(tmp_Neural_Network_Manager_A.Get__Path_Dataset_Manager()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_Neural_Network_Manager_A.Get__Path_Dataset_Manager().c_str(),
                                 __LINE__);

        return(false);
    }
    
    tmp_Animation_Waiting.Join();
    PRINT_FORMAT(NEW_LINE "%s" NEW_LINE, MyEA::String::Get__Time().c_str());

    return(true);
}
