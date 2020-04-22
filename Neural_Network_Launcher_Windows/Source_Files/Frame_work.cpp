#include "stdafx.hpp"
#include "main.hpp"

#include <MODWT_SAEs_LSTM.hpp>
#include <Frame_work.hpp>

#include <Capturing/Shutdown/Shutdown.hpp>

#include <iostream>

bool Framework(class MyEA::Capturing::Shutdown &shutdown_module)
{
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string("Framework - Neural Network").c_str());
#endif
    
    while(true)
    {
        PRINT_FORMAT("%s: Options:" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[0]: MODWT-SAEs-LSTM." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[1]: Quit." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u, MyEA::Time::Date_Time_Now() + ": Choose an option: "))
        {
            case 0u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(MODWT_SAEs_LSTM(shutdown_module) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"MODWT_SAEs_LSTM(ref)\" function. At line %d." NEW_LINE,
                                                MyEA::Time::Date_Time_Now().c_str(),
                                                __FUNCTION__,
                                                __LINE__);
                }
                    break;
            case 1u: return(true);
            default: PRINT_FORMAT("%s: Invalid option." NEW_LINE, MyEA::Time::Date_Time_Now().c_str()); break;
        }

        if(shutdown_module.Get__On_Shutdown()) { break; }
    }

    return(true);
}
