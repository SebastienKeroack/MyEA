#include "stdafx.hpp"
#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif

#include <Strings/Animation_Waiting.hpp>

#include <Preprocessing__Sequential_Input.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Preprocessing__Sequential_Input(void)
{
    std::string tmp_dataset_name;
    
    std::cout << MyEA::Time::Date_Time_Now() << ": Dataset name: ";

    getline(std::cin, tmp_dataset_name);
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_dataset_name + " - Preprocessing, Sequential input").c_str());
#endif
    
    class MyEA::Neural_Network::Neural_Network_Manager tmp_Neural_Network_Manager;
    
    if(tmp_Neural_Network_Manager.Initialize_Path(tmp_dataset_name, tmp_dataset_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
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
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    // |END| Dataset Manager Parameters. |END|
    
    if(tmp_Neural_Network_Manager.Initialize_Path(tmp_dataset_name, tmp_dataset_name + "_SE") == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s_SE)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_dataset_name.c_str(),
                                 tmp_dataset_name.c_str(),
                                 __LINE__);

        return(false);
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Recurrent depth." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s:\tRange[2, %zu]." NEW_LINE,
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Inputs());
    size_t const tmp_number_recurrent_depth(MyEA::String::Cin_Number<size_t>(2_zu,
                                                                                                                       tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Inputs(),
                                                                                                                       MyEA::Time::Date_Time_Now() + ": Recurrent depth: "));
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Saving into %s... ",
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_Neural_Network_Manager.Get__Path_Dataset_Manager().c_str());
    class MyEA::String::Animation_Waiting tmp_Animation_Waiting;
    tmp_Animation_Waiting.Print_While_Async();

    if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Save__Sequential_Input(tmp_number_recurrent_depth, tmp_Neural_Network_Manager.Get__Path_Dataset_Manager()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save__Sequential_Input(%zu, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_number_recurrent_depth,
                                 tmp_Neural_Network_Manager.Get__Path_Dataset_Manager().c_str(),
                                 __LINE__);

        return(false);
    }
    
    tmp_Animation_Waiting.Join();
    PRINT_FORMAT(NEW_LINE "%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

    return(true);
}
