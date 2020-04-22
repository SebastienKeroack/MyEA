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
﻿
#include "stdafx.hpp"

#include "main.hpp"

#if defined(COMPILE_WINDOWS)
    #include <windows.h>
#endif

#include <Strings/Animation_Waiting.hpp>

#include <Preprocessing__Minimum_Maximum.hpp>

#include <Neural_Network/Neural_Network_Manager.hpp>

#include <iostream>

bool Preprocessing__Minimum_Maximum(void)
{
    std::string tmp_dataset_name;
    
    std::cout << MyEA::Time::Date_Time_Now() << ": Dataset name: ";

    getline(std::cin, tmp_dataset_name);
    
#if defined(COMPILE_WINDOWS)
    // TODO: Make the application Unicode with macro controlling wstring for windows and string for linux.
    SetConsoleTitle(std::string(tmp_dataset_name + " - Preprocessing, Scaler - Minimum Maximum").c_str());
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
    
    size_t const tmp_number_examples(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Examples());
    size_t tmp_data_start_index,
              tmp_data_end_index,
              tmp_input_index;

    T_ tmp_minimum,
         tmp_maximum,
         tmp_minimum_value,
         tmp_maximum_value,
         tmp_minimum_range,
         tmp_maximum_range;

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Start index." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s:\tRange[0, %zu]." NEW_LINE,
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_number_examples);
    PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    tmp_data_start_index = MyEA::String::Cin_Number<size_t>(0_zu,
                                                                                            tmp_number_examples,
                                                                                            MyEA::Time::Date_Time_Now() + ": Start index: ");
            
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: End index." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s:\tRange[%zu, %zu]." NEW_LINE,
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_data_start_index,
                             tmp_number_examples);
    PRINT_FORMAT("%s:\tdefault=%zu." NEW_LINE,
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_number_examples);
    tmp_data_end_index = MyEA::String::Cin_Number<size_t>(tmp_data_start_index,
                                                                                           tmp_number_examples,
                                                                                           MyEA::Time::Date_Time_Now() + ": End index: ");
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Minimum range." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s:\tdefault=0." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    tmp_minimum_range = MyEA::String::Cin_Real_Number<T_>(-(std::numeric_limits<ST_>::max)(),
                                                                                              (std::numeric_limits<ST_>::max)(),
                                                                                              MyEA::Time::Date_Time_Now() + ": Minimum range: ");

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Maximum range." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s:\tRange[%f, inf]." NEW_LINE,
                             MyEA::Time::Date_Time_Now().c_str(),
                             Cast_T(tmp_minimum_range));
    PRINT_FORMAT("%s:\tdefault=1." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    tmp_maximum_range = MyEA::String::Cin_Real_Number<T_>(tmp_minimum_range,
                                                                                               (std::numeric_limits<ST_>::max)(),
                                                                                               MyEA::Time::Date_Time_Now() + ": Maximum range: ");
    
    tmp_minimum_value = (std::numeric_limits<ST_>::max)();

    tmp_maximum_value = -(std::numeric_limits<ST_>::max)();
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to get min/max from input(s)?"))
    {
        if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Inputs() == 1_zu || MyEA::String::Accept(MyEA::Time::Date_Time_Now() + NEW_LINE + MyEA::Time::Date_Time_Now() + ": Do you want to get min/max from all inputs?"))
        {
            tmp_minimum = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Minimum_Input(tmp_data_start_index,
                                                                                                                                                         tmp_data_end_index,
                                                                                                                                                         ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

            tmp_maximum = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Maximum_Input(tmp_data_start_index,
                                                                                                                                                           tmp_data_end_index,
                                                                                                                                                           ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

            tmp_minimum_value = tmp_minimum < tmp_minimum_value ? tmp_minimum : tmp_minimum_value;

            tmp_maximum_value = tmp_maximum > tmp_maximum_value ? tmp_maximum : tmp_maximum_value;
        }
        else
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            for(tmp_input_index = 0_zu; tmp_input_index != tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Inputs(); ++tmp_input_index)
            {
                if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to get min/max from input " + std::to_string(tmp_input_index) + "?"))
                {
                    tmp_minimum = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Minimum_Input(tmp_data_start_index,
                                                                                                                                                                 tmp_data_end_index,
                                                                                                                                                                 tmp_input_index,
                                                                                                                                                                 ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                    tmp_maximum = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Maximum_Input(tmp_data_start_index,
                                                                                                                                                                   tmp_data_end_index,
                                                                                                                                                                   tmp_input_index,
                                                                                                                                                                   ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                    tmp_minimum_value = tmp_minimum < tmp_minimum_value ? tmp_minimum : tmp_minimum_value;

                    tmp_maximum_value = tmp_maximum > tmp_maximum_value ? tmp_maximum : tmp_maximum_value;
                }
            }
        }
    }
        
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to get min/max from output(s)?"))
    {
        if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Outputs() == 1_zu || MyEA::String::Accept(MyEA::Time::Date_Time_Now() + NEW_LINE + MyEA::Time::Date_Time_Now() + ": Do you want to get min/max from all outputs?"))
        {
            tmp_minimum = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Minimum_Input(tmp_data_start_index,
                                                                                                                                                         tmp_data_end_index,
                                                                                                                                                         ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT);

            tmp_maximum = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Maximum_Input(tmp_data_start_index,
                                                                                                                                                           tmp_data_end_index,
                                                                                                                                                           ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT);

            tmp_minimum_value = tmp_minimum < tmp_minimum_value ? tmp_minimum : tmp_minimum_value;

            tmp_maximum_value = tmp_maximum > tmp_maximum_value ? tmp_maximum : tmp_maximum_value;
        }
        else
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            for(tmp_input_index = 0_zu; tmp_input_index != tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Outputs(); ++tmp_input_index)
            {
                if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to get min/max from output " + std::to_string(tmp_input_index) + "?"))
                {
                    tmp_minimum = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Minimum_Input(tmp_data_start_index,
                                                                                                                                                                 tmp_data_end_index,
                                                                                                                                                                 tmp_input_index,
                                                                                                                                                                 ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT);

                    tmp_maximum = tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Maximum_Input(tmp_data_start_index,
                                                                                                                                                                   tmp_data_end_index,
                                                                                                                                                                   tmp_input_index,
                                                                                                                                                                   ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT);

                    tmp_minimum_value = tmp_minimum < tmp_minimum_value ? tmp_minimum : tmp_minimum_value;

                    tmp_maximum_value = tmp_maximum > tmp_maximum_value ? tmp_maximum : tmp_maximum_value;
                }
            }
        }
    }

    if(tmp_minimum_value == (std::numeric_limits<ST_>::max)())
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: Minimum value." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s:\tRange[-inf , inf]." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        tmp_minimum_value = MyEA::String::Cin_Real_Number<T_>(-(std::numeric_limits<ST_>::max)(),
                                                                                                  (std::numeric_limits<ST_>::max)(),
                                                                                                  MyEA::Time::Date_Time_Now() + ": Minimum value: ");

    }
    
    if(tmp_maximum_value == -(std::numeric_limits<ST_>::max)())
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s: Maximum value." NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        PRINT_FORMAT("%s:\tRange[%f, inf]." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 Cast_T(tmp_minimum_value));

        tmp_maximum_value = MyEA::String::Cin_Real_Number<T_>(tmp_minimum_value,
                                                                                                  (std::numeric_limits<ST_>::max)(),
                                                                                                  MyEA::Time::Date_Time_Now() + ": Maximum value: ");
    }

    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to preprocess input(s)?"))
    {
        PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
        if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Inputs() == 1_zu || MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to preprocess all inputs?"))
        {
            if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                                                                          tmp_number_examples,
                                                                                                                                                          tmp_minimum_value,
                                                                                                                                                          tmp_maximum_value,
                                                                                                                                                          tmp_minimum_range,
                                                                                                                                                          tmp_maximum_range,
                                                                                                                                                          ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum(%zu, %zu, %f, %f, %f, %f, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         0_zu,
                                         tmp_number_examples,
                                         Cast_T(tmp_minimum_value),
                                         Cast_T(tmp_maximum_value),
                                         Cast_T(tmp_minimum_range),
                                         Cast_T(tmp_maximum_range),
                                         ENUM_TYPE_INPUT::TYPE_INPUT_INPUT,
                                         __LINE__);

                return(false);
            }
        }
        else
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            for(tmp_input_index = 0_zu; tmp_input_index != tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Inputs(); ++tmp_input_index)
            {
                if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to preprocess input " + std::to_string(tmp_input_index) + "?"))
                {
                    if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                                                                                  tmp_number_examples,
                                                                                                                                                                  tmp_input_index,
                                                                                                                                                                  tmp_minimum_value,
                                                                                                                                                                  tmp_maximum_value,
                                                                                                                                                                  tmp_minimum_range,
                                                                                                                                                                  tmp_maximum_range,
                                                                                                                                                                  ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum(%zu, %zu, %zu, %f, %f, %f, %f, %u)\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 0_zu,
                                                 tmp_number_examples,
                                                 tmp_input_index,
                                                 Cast_T(tmp_minimum_value),
                                                 Cast_T(tmp_maximum_value),
                                                 Cast_T(tmp_minimum_range),
                                                 Cast_T(tmp_maximum_range),
                                                 ENUM_TYPE_INPUT::TYPE_INPUT_INPUT,
                                                 __LINE__);

                        return(false);
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
            if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                                                                          tmp_number_examples,
                                                                                                                                                          tmp_minimum_value,
                                                                                                                                                          tmp_maximum_value,
                                                                                                                                                          tmp_minimum_range,
                                                                                                                                                          tmp_maximum_range,
                                                                                                                                                          ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum(%zu, %zu, %f, %f, %f, %f, %u)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         0_zu,
                                         tmp_number_examples,
                                         Cast_T(tmp_minimum_value),
                                         Cast_T(tmp_maximum_value),
                                         Cast_T(tmp_minimum_range),
                                         Cast_T(tmp_maximum_range),
                                         ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT,
                                         __LINE__);

                return(false);
            }
        }
        else
        {
            PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
            for(tmp_input_index = 0_zu; tmp_input_index != tmp_Neural_Network_Manager.Get__Dataset_Manager()->Get__Number_Outputs(); ++tmp_input_index)
            {
                if(MyEA::String::Accept(MyEA::Time::Date_Time_Now() + ": Do you want to preprocess output " + std::to_string(tmp_input_index) + "?"))
                {
                    if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                                                                                  tmp_number_examples,
                                                                                                                                                                  tmp_input_index,
                                                                                                                                                                  tmp_minimum_value,
                                                                                                                                                                  tmp_maximum_value,
                                                                                                                                                                  tmp_minimum_range,
                                                                                                                                                                  tmp_maximum_range,
                                                                                                                                                                  ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                    {
                        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum(%zu, %zu, %zu, %f, %f, %f, %f, %u)\" function. At line %d." NEW_LINE,
                                                 MyEA::Time::Date_Time_Now().c_str(),
                                                 __FUNCTION__,
                                                 0_zu,
                                                 tmp_number_examples,
                                                 tmp_input_index,
                                                 Cast_T(tmp_minimum_value),
                                                 Cast_T(tmp_maximum_value),
                                                 Cast_T(tmp_minimum_range),
                                                 Cast_T(tmp_maximum_range),
                                                 ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT,
                                                 __LINE__);

                        return(false);
                    }
                }
            }
        }
    }

    if(tmp_Neural_Network_Manager.Initialize_Path(tmp_dataset_name, tmp_dataset_name + "_ScalerMinMax") == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_Directory(%s, %s_ScalerMinMax)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 tmp_dataset_name.c_str(),
                                 tmp_dataset_name.c_str(),
                                 __LINE__);

        return(false);
    }
    
    PRINT_FORMAT("%s" NEW_LINE, MyEA::Time::Date_Time_Now().c_str());
    PRINT_FORMAT("%s: Saving into %s... ",
                             MyEA::Time::Date_Time_Now().c_str(),
                             tmp_Neural_Network_Manager.Get__Path_Dataset_Manager().c_str());
    class MyEA::String::Animation_Waiting tmp_Animation_Waiting;
    tmp_Animation_Waiting.Print_While_Async();

    if(tmp_Neural_Network_Manager.Get__Dataset_Manager()->Save(tmp_Neural_Network_Manager.Get__Path_Dataset_Manager()) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Save(%s)\" function. At line %d." NEW_LINE,
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
