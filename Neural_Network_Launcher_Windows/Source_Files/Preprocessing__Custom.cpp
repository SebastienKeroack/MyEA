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

#include <Preprocessing__Custom.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Preprocessing__Custom(void)
{
    std::string tmp_dataset_name;
    
    std::cout << MyEA::Time::Date_Time_Now() << ": Dataset name: ";

    getline(std::cin, tmp_dataset_name);
    
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_dataset_name + " - Preprocessing, Custom").c_str());
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
    
    tmp_Dataset_Manager_Parameters.type_storage = 2;
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
    
    class Dataset_Manager<T_> *const tmp_ptr_Dataset_Manager(tmp_Neural_Network_Manager.Get__Dataset_Manager());
    
    // Validate input(s)/output(s) size.
    size_t tmp_J_level(0),
              tmp_shift_index;
    
    if(tmp_ptr_Dataset_Manager->Get__Number_Inputs() == 6_zu)
    {
        size_t const tmp_J_level_maximum(tmp_ptr_Dataset_Manager->MODWT__J_Level_Maximum());
        PRINT_FORMAT("%s: J level inverse." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s:\tRange[0, %zu]." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 tmp_J_level_maximum);
        PRINT_FORMAT("%s:\tdefault=%zu." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 MyEA::Math::Minimum<size_t>(tmp_J_level_maximum, 3_zu));
        tmp_J_level = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                  tmp_J_level_maximum,
                                                                                  MyEA::Time::Date_Time_Now() + ": J level inverse: ");

    }
    // |END| Validate input(s)/output(s) size. |END|
    
    if(tmp_Neural_Network_Manager.Initialize_Path(tmp_dataset_name, tmp_dataset_name + "_C") == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s_C)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_dataset_name.c_str(),
                                 tmp_dataset_name.c_str(),
                                 __LINE__);

        return(false);
    }
    
    // Preprocessing.

    T_ tmp_minimum_input,
         tmp_maximum_input;
    
    class Dataset<T_> *tmp_ptr_TrainingSet(tmp_ptr_Dataset_Manager->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Preprocessing... ", MyEA::Time::Date_Time_Now().c_str());
    class MyEA::String::Animation_Waiting tmp_Animation_Waiting;
    tmp_Animation_Waiting.Print_While_Async();

    if(tmp_ptr_Dataset_Manager->Get__Number_Inputs() == 6_zu)
    {
        // Zero centered.
        //  Price, high, low, close.
        for(size_t tmp_index(0_zu); tmp_index != 3_zu; ++tmp_index)
        {
            tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                             tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                             tmp_index,
                                                                                                             ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
            
            tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                               tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                               tmp_index,
                                                                                                               ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

            if(tmp_ptr_Dataset_Manager->Preprocessing__Zero_Centered(0_zu,
                                                                                                      tmp_ptr_Dataset_Manager->Get__Number_Examples(),
                                                                                                      tmp_index,
                                                                                                      1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                      ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }

        
        //  |END| Price, high, low, close. |END|
        // |END| Zero centered. |END|

        // ATR.
        tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                            tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                            4_zu,
                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                        
        tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                            tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                            4_zu,
                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

        if(tmp_ptr_Dataset_Manager->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                          tmp_ptr_Dataset_Manager->Get__Number_Examples(),
                                                                                                          4_zu,
                                                                                                          tmp_minimum_input,
                                                                                                          tmp_maximum_input,
                                                                                                          0_T,
                                                                                                          1_T,
                                                                                                          ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                        MyEA::Time::Date_Time_Now().c_str(),
                                        __FUNCTION__,
                                        __LINE__);

            return(false);
        }
        // |END| ATR. |END|

        // Momentum.
        tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                            tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                            5_zu,
                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                        
        tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                            tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                            5_zu,
                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

        if(tmp_ptr_Dataset_Manager->Preprocessing__Zero_Centered(0_zu,
                                                                                                        tmp_ptr_Dataset_Manager->Get__Number_Examples(),
                                                                                                        5_zu,
                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                        MyEA::Time::Date_Time_Now().c_str(),
                                        __FUNCTION__,
                                        __LINE__);

            return(false);
        }
        // |END| Momentum. |END|

        if(tmp_J_level != 0_zu)
        {
            // Merge MODWT.
            //  Price, high, low, close.
            tmp_shift_index = 0_zu;
            
            for(size_t tmp_index(0_zu); tmp_index != 3_zu; ++tmp_index)
            {
                if(tmp_ptr_Dataset_Manager->Preprocessing__Merge__MODWT(tmp_index + tmp_shift_index,
                                                                                                              tmp_J_level,
                                                                                                              ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Merge__MODWT()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }

                tmp_shift_index += tmp_J_level;
            }

            size_t tmp_J_level2(5_zu);

            for(size_t tmp_index(3_zu); tmp_index != 6_zu; ++tmp_index)
            {
                if(tmp_ptr_Dataset_Manager->Preprocessing__Merge__MODWT(tmp_index + tmp_shift_index,
                                                                                                              tmp_J_level2,
                                                                                                              ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Merge__MODWT()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);

                    return(false);
                }

                tmp_shift_index += tmp_J_level2;
            }
            //  |END| Price, high, low, close. |END|
            // |END| Merge MODWT. |END|
        }
    }
    // |END| Preprocessing. |END|
    
    tmp_Animation_Waiting.Join();
    PRINT_FORMAT(NEW_LINE "%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Saving into %s... ",
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_Neural_Network_Manager.Get__Path_Dataset_Manager().c_str());
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
