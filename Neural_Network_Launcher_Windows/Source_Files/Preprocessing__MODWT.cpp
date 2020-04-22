/* Copyright 2020 Sébastien Kéroack. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "stdafx.hpp"
#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif

#include <Strings/Animation_Waiting.hpp>

#include <Preprocessing__MODWT.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Preprocessing__MODWT(void)
{
    std::string tmp_dataset_name;
    
    std::cout << MyEA::Time::Date_Time_Now() << ": Dataset name: ";

    getline(std::cin, tmp_dataset_name);
    
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_dataset_name + " - Preprocessing, MODWT").c_str());
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
    
    size_t const tmp_J_level_maximum(tmp_Neural_Network_Manager.Get__Dataset_Manager()->MODWT__J_Level_Maximum());
    size_t tmp_J_level,
              tmp_input_index,
              tmp_shift_index;

    if(tmp_J_level_maximum == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Can not preprocess the dataset. No enough data available. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    bool const tmp_merge_modwt(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to merge MODWT?"));

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: J level." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s:\tRange[1, %zu]." NEW_LINE,
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_J_level_maximum);
    PRINT_FORMAT("%s:\tdefault=%zu." NEW_LINE,
                             MyEA::Time::Date_Time_Now().c_str(),
                             MyEA::Math::Minimum<size_t>(tmp_J_level_maximum, 3_zu));
    tmp_J_level = MyEA::String::Cin_Number<size_t>(1_zu,
                                                                              tmp_J_level_maximum,
                                                                              MyEA::Time::Date_Time_Now() + ": J level: ");

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to preprocess input(s)?"))
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Inputs() == 1_zu || MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to preprocess all inputs?"))
        {
            if(tmp_merge_modwt)
            {
                if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Preprocessing__Merge__MODWT(tmp_J_level, ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Merge__MODWT(%zu, %u)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             tmp_J_level,
                                             ENUM_TYPE_INPUT::TYPE_INPUT_INPUT,
                                             __LINE__);

                    return(false);
                }
            }
            else
            {
                if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Preprocessing__MODWT(tmp_J_level, ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT(%zu, %u)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             tmp_J_level,
                                             ENUM_TYPE_INPUT::TYPE_INPUT_INPUT,
                                             __LINE__);

                    return(false);
                }
            }
        }
        else
        {
            if(tmp_merge_modwt) { tmp_shift_index = 0_zu; }

            PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            for(tmp_input_index = 0_zu; tmp_input_index != tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Inputs(); ++tmp_input_index)
            {
                if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to preprocess input " + std::to_string(tmp_input_index) + "?"))
                {
                    if(tmp_merge_modwt)
                    {
                        if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Preprocessing__Merge__MODWT(tmp_input_index + tmp_shift_index,
                                                                                                                                                                 tmp_J_level,
                                                                                                                                                                 ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Merge__MODWT(%zu, %zu, %u)\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_index + tmp_shift_index,
                                                     tmp_J_level,
                                                     ENUM_TYPE_INPUT::TYPE_INPUT_INPUT,
                                                     __LINE__);

                            return(false);
                        }

                        tmp_shift_index += tmp_J_level;
                    }
                    else
                    {
                        if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Preprocessing__MODWT(tmp_input_index,
                                                                                                                                                     tmp_J_level,
                                                                                                                                                     ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT(%zu, %zu, %u)\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_index,
                                                     tmp_J_level,
                                                     ENUM_TYPE_INPUT::TYPE_INPUT_INPUT,
                                                     __LINE__);

                            return(false);
                        }
                    }
                }
            }
        }
    }

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to preprocess output(s)?"))
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Outputs() == 1_zu || MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to preprocess all outputs?"))
        {
            if(tmp_merge_modwt)
            {
                if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Preprocessing__Merge__MODWT(tmp_J_level, ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Merge__MODWT(%zu, %u)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             tmp_J_level,
                                             ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT,
                                             __LINE__);

                    return(false);
                }
            }
            else
            {
                if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Preprocessing__MODWT(tmp_J_level, ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT(%zu, %u)\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             tmp_J_level,
                                             ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT,
                                             __LINE__);

                    return(false);
                }
            }
        }
        else
        {
            if(tmp_merge_modwt) { tmp_shift_index = 0_zu; }

            PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            for(tmp_input_index = 0_zu; tmp_input_index != tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Outputs(); ++tmp_input_index)
            {
                if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to preprocess output " + std::to_string(tmp_input_index) + "?"))
                {
                    if(tmp_merge_modwt)
                    {
                        if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Preprocessing__Merge__MODWT(tmp_input_index + tmp_shift_index,
                                                                                                                                                                 tmp_J_level,
                                                                                                                                                                 ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Merge__MODWT(%zu, %zu, %u)\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_index + tmp_shift_index,
                                                     tmp_J_level,
                                                     ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT,
                                                     __LINE__);

                            return(false);
                        }

                        tmp_shift_index += tmp_J_level;
                    }
                    else
                    {
                        if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Preprocessing__MODWT(tmp_input_index,
                                                                                                                                                     tmp_J_level,
                                                                                                                                                     ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__MODWT(%zu, %zu, %u)\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     tmp_input_index,
                                                     tmp_J_level,
                                                     ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT,
                                                     __LINE__);

                            return(false);
                        }
                    }
                }
            }
            PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        }
    }

    tmp_Neural_Network_Manager.Initialize_Path(tmp_dataset_name, tmp_dataset_name + "_MODWT");
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Saving into %s... ",
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_Neural_Network_Manager.Get__Path_Dataset_Manager().c_str());
    class MyEA::String::Animation_Waiting tmp_Animation_Waiting;
    tmp_Animation_Waiting.Print_While_Async();

    if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Save(tmp_Neural_Network_Manager.Get__Path_Dataset_Manager(), true) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save(%s, true)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_Neural_Network_Manager.Get__Path_Dataset_Manager().c_str(),
                                 __LINE__);

        return(false);
    }
    
    tmp_Animation_Waiting.Join();
    PRINT_FORMAT(NEW_LINE "%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());

    return(true);
}
