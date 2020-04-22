#include "stdafx.hpp"
#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <Enums/Enum_Type_Chart.hpp>

    #include <windows.h>
    #include <Form.hpp>
#endif

#include <Strings/Animation_Waiting.hpp>

#include <Dataset__Datapoint.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Dataset__Datapoint(void)
{
    std::string tmp_dataset_name;
    
    std::cout << MyEA::Time::Date_Time_Now() << ": Dataset name: ";

    getline(std::cin, tmp_dataset_name);

#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_dataset_name + " - Datapoint").c_str());
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
    
#if defined(COMPILE_UI)
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    enum ENUM_TYPE_INPUT const tmp_type_input(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to datapoint input(s)?") ? ENUM_TYPE_INPUT::TYPE_INPUT_INPUT : ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT);
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: FORM: Allocate." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    MyEA::Form::API__Form__Neural_Network__Allocate();

    PRINT_FORMAT("%s: FORM: Initialize chart of type output with 2 series." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    MyEA::Form::API__Form__Neural_Network__Chart_Initialize(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_OUTPUT, 2u);

    PRINT_FORMAT("%s: FORM: Maximum ploted example(s)." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->User_Controls__Set__Maximum_Ploted_Example() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Set__Maximum_Ploted_Example()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    PRINT_FORMAT("%s: FORM: Time delay ploted." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->User_Controls__Set__Time_Delay_Ploted() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Set__Time_Delay_Ploted()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    tmp_Neural_Network_Manager.Get__Dataset_Manager()->Set__Plot__Output(true);
    
    class MyEA::String::Animation_Waiting tmp_Animation_Waiting;

    do
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: Input index." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s:\tRange[-1, %zu]." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 (tmp_type_input == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Inputs() : tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Outputs()) - 1_zu);
        PRINT_FORMAT("%s:\tdefault=-1." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

        int const tmp_input_index(MyEA::String::Cin_Number<int>(-1,
                                                                                              static_cast<int>(tmp_type_input == ENUM_TYPE_INPUT::TYPE_INPUT_INPUT ? tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Inputs() : tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Outputs()) - 1,
                                                                                              MyEA::Time::Date_Time_Now() + ": Input index: "));

        PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: FORM: Plot dataset manager... ", MyEA::Time::Date_Time_Now().c_str());
        tmp_Animation_Waiting.Print_While_Async();

        if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Plot__Dataset_Manager(tmp_input_index, tmp_type_input) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Plot__Dataset_Manager(%d, %u)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_input_index,
                                     tmp_type_input,
                                     __LINE__);

            return(false);
        }

        tmp_Animation_Waiting.Join();
        PRINT_FORMAT(NEW_LINE);
    }
    while(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": " + NEW_LINE + MyEA::Time::Date_Time_Now() + ": Do you want to datapoint again?"));
#endif
    
    PAUSE_TERMINAL();
    
#if defined(COMPILE_UI)
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: FORM: Deallocate." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    MyEA::Form::API__Form__Neural_Network__Deallocate();
#endif

    return(true);
}
