#include "stdafx.hpp"
#include "main.hpp"

#include <User_Controls__Preprocessing.hpp>
#include <Preprocessing__Autoencoder.hpp>
#include <Preprocessing__Concat.hpp>
#include <Preprocessing__Custom.hpp>
#include <Preprocessing__Input_To_Output.hpp>
#include <Preprocessing__Merge_Dataset.hpp>
#include <Preprocessing__Minimum_Maximum.hpp>
#include <Preprocessing__MODWT.hpp>
#include <Preprocessing__Remove_Duplicate_Entries_Dataset.hpp>
#include <Preprocessing__Remove_IO.hpp>
#include <Preprocessing__Replace_Entries.hpp>
#include <Preprocessing__Sequential_Input.hpp>
#include <Preprocessing__Shift_Entries.hpp>
#include <Preprocessing__Spliting_Dataset.hpp>
#include <Preprocessing__Time_Direction.hpp>
#include <Preprocessing__Unrecurrent.hpp>
#include <Preprocessing__Zero_Centered.hpp>

#include <Capturing/Shutdown/Shutdown.hpp>

#include <iostream>

bool User_Controls__Preprocessing(void)
{
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string("Preprocessing - Neural Network").c_str());
#endif
    
    while(true)
    {
        PRINT_FORMAT("%s: Options:" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[0]: Autoencoder dataset." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[1]: Concat." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[2]: Custom." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[3]: Input(s) to output(s)." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[4]: Merge dataset." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[5]: Minimum maximum (MinMax)." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[6]: Maximum overlap discrete wavelet transform (MODWT)." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[7]: Remove duplicate entrie(s)." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[8]: Remove IO." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[9]: Replace entrie(s)." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[10]: Sequential input." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[11]: Shift entrie(s)." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[12]: Spliting dataset." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[13]: Time-direction." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[14]: Unrecurrent." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[15]: Zero centered." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: \t[16]: Quit." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

        switch(MyEA::String::Cin_Number<unsigned int>(0u,
                                                                                16u,
                                                                                MyEA::Time::Date_Time_Now() + ": Choose an option: "))
        {
            case 0u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Autoencoder() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Autoencoder()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 1u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Concat() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Concat()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 2u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Custom() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Custom()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 3u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Input_To_Output() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Input_To_Output()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 4u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Merge_Dataset() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Input_To_Output()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 5u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Minimum_Maximum() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 6u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__MODWT() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 7u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Remove_Duplicate_Entries_Dataset() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Remove_Duplicate_Entries_Dataset()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 8u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Remove_IO() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Remove_IO()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 9u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Replace_Entries() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Replace_Entries()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 10u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Sequential_Input() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Sequential_Input()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 11u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Shift_Entries() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Shift_Entries()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 12u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Spliting_Dataset() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Spliting_Dataset()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 13u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Time_Direction() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Time_Direction()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 14u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Unrecurrent() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Unrecurrent()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 15u:
                PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
                if(Preprocessing__Zero_Centered() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                }
                    break;
            case 16u: return(true);
            default:
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Cin_Number<unsigned int>(%u, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         0u,
                                         16u,
                                         __LINE__);
                    break;
        }
    }

    return(true);
}
