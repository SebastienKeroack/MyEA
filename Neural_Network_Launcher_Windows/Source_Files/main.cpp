#include "stdafx.hpp"
#include "main.hpp"
#include "debug.hpp"

#if defined(COMPILE_AUTODIFF)
    #include <adept_source.h>
#endif

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
    #include <tchar.h>
#endif

/*
Linux:
    Library requirements:
    sudo apt install libsystemd-dev && 
    sudo apt install -y build-essential gdbserver
        --> sudo apt install -y libsystemd-dev build-essential gdbserver

    If need <sys/...h>:
        sudo apt-get install libc6-dev-amd64

    GCC 8 on 18.04:
    sudo apt -y update && sudo apt -y upgrade && sudo apt -y dist-upgrade
        sudo apt install build-essential software-properties-common -y && 
        sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y && 
        sudo apt update -y && 
        sudo apt install gcc-8 g++-8 -y && 
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 60 --slave /usr/bin/g++ g++ /usr/bin/g++-8 && 
        sudo update-alternatives --config gcc 
            OR
        sudo apt install aptitude && 
        sudo aptitude install libubsan1 libasan5 libgcc-8-dev gcc-8-base gcc-8 cpp-8 g++-8 && 
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 60 --slave /usr/bin/g++ g++ /usr/bin/g++-8 && 
        sudo update-alternatives --config gcc
 */

#include <Capturing/Shutdown/Shutdown.hpp>
#include <Files/File.hpp>

#include <Start_Neural_Network.hpp>
#include <Pre_Training.hpp>
#include <Transfer_Learning.hpp>
#include <Frame_work.hpp>
#include <Count_Identical_Outputs_Entries.hpp>
#include <Grid_Search_Optimizer.hpp>
#include <Simulate_Classification_Trading_Session.hpp>
#include <Dataset__Datapoint.hpp>
#include <User_Controls__Preprocessing.hpp>

#include <iostream>
#include <thread>

#if defined(COMPILE_WINDOWS)
bool Search_Option(unsigned int const number_arguments_received,
                   _TCHAR const *const ptr_array_arguments_received[],
                   std::string const search_option_received)
#elif defined(COMPILE_LINUX)
bool Search_Option(unsigned int const number_arguments_received,
                   char const *const ptr_array_arguments_received[],
                   std::string const search_option_received)
#endif
{
    std::string tmp_search_option("-" + search_option_received);

    for(unsigned int i(0u); i != number_arguments_received; ++i) { if(strcmp(ptr_array_arguments_received[i], tmp_search_option.c_str()) == 0) { return(true); } }

    return(false);
}

#if defined(COMPILE_WINDOWS)
int _tmain(int const number_arguments_received, _TCHAR const *const ptr_array_arguments_received[])
#elif defined(COMPILE_LINUX)
int main(int const number_arguments_received, char const *const ptr_array_arguments_received[])
#endif
{
#if defined(COMPILE_WINDOWS)
    SetConsoleTitle("Menu - Neural Network");
#endif

#if (defined(_DEBUG) || defined(COMPILE_DEBUG)) && defined(_CRTDBG_MAP_ALLOC) && defined(COMPILE_WINDOWS)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    //_clearfp();
    unsigned int control_word;
    _controlfp_s(&control_word, _EM_INEXACT | _EM_UNDERFLOW | _EM_OVERFLOW, _MCW_EM);
#endif

    PRINT_FORMAT("%s: ******************************************" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: \tCopyright Sebastien Keroack" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: ******************************************" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

    PRINT_FORMAT("%s: Current path: %s" NEW_LINE,
                 MyEA::Time::Date_Time_Now().c_str(),
                 ptr_array_arguments_received[0u]);
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

#if defined(COMPILE_WINDOWS)
    class MyEA::Capturing::Shutdown tmp_Shutdown_Block("Neural Network Launcher Windows", "CLASS_Neural_Network_Launcher_Windows");
#elif defined(COMPILE_LINUX)
    class MyEA::Capturing::Shutdown tmp_Shutdown_Block("Neural Network Launcher Windows");
#endif

    tmp_Shutdown_Block.Initialize_Static_Shutdown_Block();

#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    bool const tmp_option_debug(Search_Option(static_cast<unsigned int>(number_arguments_received),
                                              ptr_array_arguments_received,
                                              "debug"));
    bool const tmp_option_nsight(Search_Option(static_cast<unsigned int>(number_arguments_received),
                                               ptr_array_arguments_received,
                                               "nsight"));
    
    if(tmp_option_debug
       ||
       MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Are you debuging from NVIDIA visual profiler?"))
    {
        simple_debug(tmp_option_nsight);

        if(tmp_option_nsight == false) { PAUSE_TERMINAL(); }

        return(0);
    }
#endif

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

    while(true)
    {
        PRINT_FORMAT("%s: Options:" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[0]: Optimization." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[1]: Pre-training." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[2]: Transfer learning." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[3]: Framework." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[4]: Grid search." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[5]: [DEPRECATED] Random search." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[6]: Count identical output(s) entrie(s)." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[7]: Simulate classification trading session." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[8]: Preprocessing." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[9]: Dataset datapoint." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[10]: Quit." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u, MyEA::Time::Date_Time_Now() + ": Choose an option: "))
        {
            case 0u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Start_Neural_Network(tmp_Shutdown_Block) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Start_Neural_Network(ref)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                }
                    break;
            case 1u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Pre_Training(tmp_Shutdown_Block) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Pre_Training(ref)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                }
                    break;
            case 2u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Transfer_Learning() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Transfer_Learning()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                }
                    break;
            case 3u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Framework(tmp_Shutdown_Block) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Framework(ref)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                }
                    break;
            case 4u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Grid_Search_Optimizer(tmp_Shutdown_Block) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Grid_Search_Optimizer(ref)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                }
                    break;
            case 5u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                    break;
            case 6u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Count_Identical_Outputs_Entries() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Count_Identical_Outputs_Entries()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                }
                    break;
            case 7u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Simulate_Classification_Trading_Session() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Simulate_Classification_Trading_Session()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                }
                    break;
            case 8u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(User_Controls__Preprocessing() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"User_Controls__Preprocessing()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                }
                    break;
            case 9u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Dataset__Datapoint() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Dataset__Datapoint()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
                }
                    break;
            case 10u: tmp_Shutdown_Block.Query_Shutdown(); break;
            default: PRINT_FORMAT("%s: Invalid option." NEW_LINE, MyEA::Time::Date_Time_Now().c_str()); break;
        }
        
        PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

        if(tmp_Shutdown_Block.Get__On_Shutdown()) { break; }
    }
    
    return(0);
}
