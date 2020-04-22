#include "stdafx.hpp"
#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif

#include <Strings/Animation_Waiting.hpp>

#include <Preprocessing__Replace_Entries.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Preprocessing__Replace_Entries(void)
{
    std::string tmp_source_dataset_name,
                    tmp_destination_dataset_name;
    
    std::cout << MyEA::Time::Date_Time_Now() << ": Dataset source name: ";

    getline(std::cin, tmp_source_dataset_name);

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

    std::cout << MyEA::Time::Date_Time_Now() << ": Dataset destination name: ";

    getline(std::cin, tmp_destination_dataset_name);
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_source_dataset_name + " / " + tmp_destination_dataset_name + " - Preprocessing, Replace entrie(s)").c_str());
#endif
    
    class MyEA::Neural_Network::Neural_Network_Manager tmp_source_Neural_Network_Manager,
                                                                                      tmp_destination_Neural_Network_Manager;
    
    if(tmp_source_Neural_Network_Manager.Initialize_Path(tmp_source_dataset_name, tmp_source_dataset_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_source_dataset_name.c_str(),
                                 tmp_source_dataset_name.c_str(),
                                 __LINE__);

        return(false);
    }
    else if(tmp_destination_Neural_Network_Manager.Initialize_Path(tmp_destination_dataset_name, tmp_destination_dataset_name) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_destination_dataset_name.c_str(),
                                 tmp_destination_dataset_name.c_str(),
                                 __LINE__);

        return(false);
    }
    
    // Dataset Manager Parameters.
    struct Dataset_Manager_Parameters tmp_Dataset_Manager_Parameters;

    tmp_Dataset_Manager_Parameters.type_storage = 0;
    tmp_Dataset_Manager_Parameters.type_training = 0;
    
    if(tmp_source_Neural_Network_Manager.Initialize_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_Manager()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(tmp_destination_Neural_Network_Manager.Initialize_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Dataset_Manager()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    // |END| Dataset Manager Parameters. |END|
    
    if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to replace input(s)?"))
    {
        if(tmp_destination_Neural_Network_Manager.Get__Dataset_Manager()->Replace_Entries(tmp_source_Neural_Network_Manager.Get__Dataset_Manager(), ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Replace_Entries(ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to replace output(s)?"))
    {
        if(tmp_destination_Neural_Network_Manager.Get__Dataset_Manager()->Replace_Entries(tmp_source_Neural_Network_Manager.Get__Dataset_Manager(), ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Replace_Entries(ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    
    if(tmp_destination_Neural_Network_Manager.Initialize_Path(tmp_destination_dataset_name, tmp_destination_dataset_name + "_RE") == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s_RE)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_destination_dataset_name.c_str(),
                                 tmp_destination_dataset_name.c_str(),
                                 __LINE__);

        return(false);
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Saving into %s... ",
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_destination_Neural_Network_Manager.Get__Path_Dataset_Manager().c_str());
    class MyEA::String::Animation_Waiting tmp_Animation_Waiting;
    tmp_Animation_Waiting.Print_While_Async();

    if(tmp_destination_Neural_Network_Manager.Get__Dataset_Manager()->Save(tmp_destination_Neural_Network_Manager.Get__Path_Dataset_Manager()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save(%s)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_destination_Neural_Network_Manager.Get__Path_Dataset_Manager().c_str(),
                                 __LINE__);

        return(false);
    }
    
    tmp_Animation_Waiting.Join();
    PRINT_FORMAT(NEW_LINE "%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

    return(true);
}
