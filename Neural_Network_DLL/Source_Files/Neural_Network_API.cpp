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

#include <Enums/Enum_Time_Frames.hpp>

#include <UI/Dialog_Box.hpp>

#include <Neural_Network_API.hpp>

#define PREPROCESSING true
#define MODWT false

namespace MyEA
{
    namespace Neural_Network
    {
        bool Preprocessing__Post__Indicator(enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicators_received, class Dataset_Manager<T_> *const ptr_Dataset_Manager_received)
        {
            if(PREPROCESSING == false) { return(true); }
            
            T_ tmp_minimum_input,
                 tmp_maximum_input;
            
            class Dataset<T_> *tmp_ptr_TrainingSet;
            
            switch(type_indicators_received)
            {
                case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_AUTOENCODER:
                    tmp_ptr_TrainingSet = ptr_Dataset_Manager_received->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING);
                    
                    if(tmp_ptr_TrainingSet == nullptr) { return(false); }
                    
                    if(ptr_Dataset_Manager_received->Get__Number_Inputs() == 12_zu || ptr_Dataset_Manager_received->Get__Number_Inputs() == 16_zu)
                    {
                        // Momentum.
                        tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                         tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                         0_zu,
                                                                                                                         ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                        
                        tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                           tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                           0_zu,
                                                                                                                           ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        0_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        0_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                        // |END| Momentum. |END|

                        // BearsPower.
                        tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                         tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                         4_zu,
                                                                                                                         ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                        
                        tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                           tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                           4_zu,
                                                                                                                           ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        4_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        4_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                        // |END| BearsPower. |END|

                        // BullsPower.
                        tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                         tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                         5_zu,
                                                                                                                         ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                        
                        tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                           tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                           5_zu,
                                                                                                                           ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
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

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        5_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                        // |END| BullsPower. |END|
                        
                        // ATR.
                        tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                         tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                         7_zu,
                                                                                                                         ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                        
                        tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                           tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                           7_zu,
                                                                                                                           ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                        if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                                                ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                                7_zu,
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

                        if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                                                ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                                7_zu,
                                                                                                                                tmp_minimum_input,
                                                                                                                                tmp_maximum_input,
                                                                                                                                0_T,
                                                                                                                                1_T,
                                                                                                                                ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                        // |END| ATR. |END|
                        
                        // StdDev.
                        tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                         tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                         8_zu,
                                                                                                                         ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                        
                        tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                           tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                           8_zu,
                                                                                                                           ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                        if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                                                ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                                8_zu,
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

                        if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                                                ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                                8_zu,
                                                                                                                                tmp_minimum_input,
                                                                                                                                tmp_maximum_input,
                                                                                                                                0_T,
                                                                                                                                1_T,
                                                                                                                                ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Minimum_Maximum()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                        // |END| StdDev. |END|
                        
                        // MovingAverageOfOscillator.
                        tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                         tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                         9_zu,
                                                                                                                         ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                        
                        tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                           tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                           9_zu,
                                                                                                                           ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        9_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        9_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                        // |END| MovingAverageOfOscillator. |END|

                        // AcceleratorOscillator.
                        tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                         tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                         10_zu,
                                                                                                                         ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                        
                        tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                           tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                           10_zu,
                                                                                                                           ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        10_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        10_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                        // |END| AcceleratorOscillator. |END|

                        // AwesomeOscillator.
                        tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                         tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                         11_zu,
                                                                                                                         ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                        
                        tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                           tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                           11_zu,
                                                                                                                           ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        11_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        11_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                        // |END| AwesomeOscillator. |END|

                        // OHLC.
                        if(ptr_Dataset_Manager_received->Get__Number_Inputs() == 16_zu)
                        {
                            // OHLC, open.
                            tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                             tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                             12_zu,
                                                                                                                             ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                            
                            tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                               tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                               12_zu,
                                                                                                                               ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                            12_zu,
                                                                                                                            1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                            12_zu,
                                                                                                                            1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                            // |END| OHLC, open. |END|
                            
                            // OHLC, high.
                            tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                             tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                             13_zu,
                                                                                                                             ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                            
                            tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                               tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                               13_zu,
                                                                                                                               ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                            13_zu,
                                                                                                                            1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                            13_zu,
                                                                                                                            1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                            // |END| OHLC, high. |END|
                            
                            // OHLC, low.
                            tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                             tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                             14_zu,
                                                                                                                             ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                            
                            tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                               tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                               14_zu,
                                                                                                                               ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                            14_zu,
                                                                                                                            1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                            14_zu,
                                                                                                                            1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                            // |END| OHLC, low. |END|
                            
                            // OHLC, close.
                            tmp_minimum_input = tmp_ptr_TrainingSet->Get__Minimum_Input(0_zu,
                                                                                                                             tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                             15_zu,
                                                                                                                             ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);
                            
                            tmp_maximum_input = tmp_ptr_TrainingSet->Get__Maximum_Input(0_zu,
                                                                                                                               tmp_ptr_TrainingSet->Dataset<T_>::Get__Number_Examples(),
                                                                                                                               15_zu,
                                                                                                                               ENUM_TYPE_INPUT::TYPE_INPUT_INPUT);

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                            15_zu,
                                                                                                                            1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                            15_zu,
                                                                                                                            1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                            // |END| OHLC, close. |END|
                        }
                        // |END| OHLC. |END|
                    }
                    else if(ptr_Dataset_Manager_received->Get__Number_Inputs() == 6_zu)
                    {
                        // Zero centered.
                        //  Price.
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

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
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

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                            tmp_index,
                                                                                                                            1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                        }
                        //  |END| Price. |END|
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

                        if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                                                ptr_Dataset_Manager_received->Get__Number_Examples(),
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

                        if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(0_zu,
                                                                                                                                ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                                4_zu,
                                                                                                                                tmp_minimum_input,
                                                                                                                                tmp_maximum_input,
                                                                                                                                0_T,
                                                                                                                                1_T,
                                                                                                                                ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
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

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
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

                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                        5_zu,
                                                                                                                        1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                        ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                        {
                            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                     MyEA::Time::Date_Time_Now().c_str(),
                                                     __FUNCTION__,
                                                     __LINE__);

                            return(false);
                        }
                        // |END| Momentum. |END|
                    }
                    else if(ptr_Dataset_Manager_received->Get__Number_Inputs() == 3_zu)
                    {
                        // Zero centered.
                        //  Price.
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

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
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

                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                            ptr_Dataset_Manager_received->Get__Number_Examples(),
                                                                                                                            tmp_index,
                                                                                                                            1_T / MyEA::Math::Maximum<T_>(MyEA::Math::Absolute<T_>(tmp_minimum_input), tmp_maximum_input),
                                                                                                                            ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Zero_Centered()\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                        }
                        //  |END| Price. |END|
                        // |END| Zero centered. |END|
                        
                        if(MODWT)
                        {
                            // Merge MODWT.
                            //  Price.
                            size_t const tmp_J_level_inverse(MyEA::Math::Minimum<size_t>(8_zu, ptr_Dataset_Manager_received->MODWT__J_Level_Maximum()));
                            size_t tmp_shift_index(0_zu);
                            
                            for(size_t tmp_index(0_zu); tmp_index != 3_zu; ++tmp_index)
                            {
                                if(ptr_Dataset_Manager_received->Preprocessing__Merge__MODWT(tmp_index + tmp_shift_index,
                                                                                                                                    tmp_J_level_inverse,
                                                                                                                                    ENUM_TYPE_INPUT::TYPE_INPUT_INPUT) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Merge__MODWT()\" function. At line %d." NEW_LINE,
                                                             MyEA::Time::Date_Time_Now().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);

                                    return(false);
                                }

                                if(ptr_Dataset_Manager_received->Preprocessing__Merge__MODWT(tmp_index + tmp_shift_index,
                                                                                                                                    tmp_J_level_inverse,
                                                                                                                                    ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Preprocessing__Merge__MODWT()\" function. At line %d." NEW_LINE,
                                                             MyEA::Time::Date_Time_Now().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);

                                    return(false);
                                }
                                
                                tmp_shift_index += tmp_J_level_inverse;
                            }
                            //  |END| Price. |END|
                            
                            if(ptr_Dataset_Manager_received->Reallocate_Internal_Storage() == false)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate_Internal_Storage()\" function. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         __LINE__);

                                return(false);
                            }
                            // |END| Merge MODWT. |END|
                        }
                    }
                        break;
                default: break;
            }

            return(true);
        }

        std::vector<class Threaded_Neural_Network_Manager *> *global_ptr_vector_ptr_Threaded_Neural_Network_Manager = nullptr;
        
        Threaded_Neural_Network_Manager::Threaded_Neural_Network_Manager(bool const is_type_position_long_received, enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicator_received) : neural_network_manager(is_type_position_long_received, type_indicator_received)
        { }

        class Threaded_Neural_Network_Manager* Get__Threaded_Neural_Network_Manager(std::vector<class Threaded_Neural_Network_Manager*> *const ptr_vector_Threaded_Neural_Network_Manager_received,
                                                                                                                                     bool const is_type_position_long_received,
                                                                                                                                     unsigned int const type_indicator_received)
        {
            if(ptr_vector_Threaded_Neural_Network_Manager_received == nullptr) { return(nullptr); }

            for(auto const &iterator: *ptr_vector_Threaded_Neural_Network_Manager_received)
            {
                if(iterator->neural_network_manager.Get__Type_Indicator() == static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received)
                  &&
                  iterator->neural_network_manager.Get__Is_Type_Position_Long() == is_type_position_long_received)
                { return(iterator); }
            }

            return(nullptr);
        }

        DLL_API bool API__Neural_Network__Is_Loaded(void) { return(true); }

        DLL_API bool API__Neural_Network__Initialize(bool const is_type_position_long_received,
                                                                                                     unsigned int const type_indicator_received,
                                                                                                     unsigned int const time_frames_received)
        {
            class Threaded_Neural_Network_Manager *tmp_ptr_Threaded_Neural_Network_Manager;

            if((tmp_ptr_Threaded_Neural_Network_Manager = new class Threaded_Neural_Network_Manager(is_type_position_long_received, static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received))) == nullptr) { return(false); }
            
            if(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Initialize_Path(MyEA::Common::ENUM_TYPE_INDICATORS_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received)], MyEA::Common::ENUM_TYPE_INDICATORS_NAMES[static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received)] + "-[" + MyEA::Common::ENUM_TIME_FRAMES_NAMES[static_cast<enum MyEA::Common::ENUM_TIME_FRAMES>(time_frames_received)] + "]-" + (is_type_position_long_received ? "L" : "S")) == false)
            {
                SAFE_DELETE(tmp_ptr_Threaded_Neural_Network_Manager);

                return(false);
            }
            
            tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Set__Auto_Save_Dataset(false); // WARNING
            
            tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Set__Optimization_Auto_Save_Trainer(true);

            tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Set__Optimization_Auto_Save_Trained(true);

            global_ptr_vector_ptr_Threaded_Neural_Network_Manager->push_back(tmp_ptr_Threaded_Neural_Network_Manager);

            return(true);
        }
        
        DLL_API bool API__Neural_Network__Initialize_Dataset_Manager(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                if(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Dataset_Manager() != nullptr) { return(true); }

                tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Set__Comparison_Expiration(60_zu * 60_zu * 48_zu);

                struct Dataset_Manager_Parameters tmp_Dataset_Manager_Parameters;
                
                switch(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received))
                {
                    case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_AUTOENCODER:
                        tmp_Dataset_Manager_Parameters.type_storage = 2;
                        tmp_Dataset_Manager_Parameters.type_training = 1;

                        tmp_Dataset_Manager_Parameters.percent_training_size = 96.0;
                        tmp_Dataset_Manager_Parameters.percent_validation_size = 3.0;

                        tmp_Dataset_Manager_Parameters.training_parameters.value_0 = true;
                        tmp_Dataset_Manager_Parameters.training_parameters.value_1 = 32;
                        tmp_Dataset_Manager_Parameters.training_parameters.value_2 = 0;
                            break;
                    case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iMA_RNN_SIGN:
                    case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iMA_RNN_PRICE:
                    case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iMA_RNN_WPRICE:
                        tmp_Dataset_Manager_Parameters.type_storage = 2;
                        tmp_Dataset_Manager_Parameters.type_training = 1;

                        tmp_Dataset_Manager_Parameters.percent_training_size = 96.0;
                        tmp_Dataset_Manager_Parameters.percent_validation_size = 3.0;

                        tmp_Dataset_Manager_Parameters.training_parameters.value_0 = true;
                        tmp_Dataset_Manager_Parameters.training_parameters.value_1 = 32;
                        tmp_Dataset_Manager_Parameters.training_parameters.value_2 = 0;
                            break;
                    default:
                        tmp_Dataset_Manager_Parameters.type_storage = 1;
                        tmp_Dataset_Manager_Parameters.type_training = 1;

                        tmp_Dataset_Manager_Parameters.percent_training_size = 80.0;

                        tmp_Dataset_Manager_Parameters.training_parameters.value_0 = true;
                        tmp_Dataset_Manager_Parameters.training_parameters.value_1 = 100;
                        tmp_Dataset_Manager_Parameters.training_parameters.value_2 = 0;
                            break;
                }

                if(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Initialize_Dataset_Manager(&tmp_Dataset_Manager_Parameters) == false) { return(false); }

                class Dataset_Manager<T_> *const tmp_ptr_Dataset_Manager(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Dataset_Manager());

                if(tmp_ptr_Dataset_Manager == nullptr) { return(false); }

                if(Preprocessing__Post__Indicator(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received), tmp_ptr_Dataset_Manager) == false) { return(false); }
            }
            else { return(false); }

            return(true);
        }

        DLL_API int API__Neural_Network__Set__Output_Mode(bool const is_type_position_long_received,
                                                                                                                   unsigned int const type_indicator_received,
                                                                                                                   unsigned int const type_neural_network_use_received,
                                                                                                                   bool const use_last_layer_as_output_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                if(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Set__Output_Mode(use_last_layer_as_output_received, static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received)) == false)
                { return(-2); }
            }
            else { return(-1); }

            return(0);
        }
        
        DLL_API int API__Neural_Network__Set__Number_Inputs(bool const is_type_position_long_received,
                                                                                                                     unsigned int const type_indicator_received,
                                                                                                                     unsigned int const number_inputs_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Set__Number_Inputs(number_inputs_received); }
            else { return(-1); }

            return(0);
        }

        DLL_API int API__Neural_Network__Set__Number_Outputs(bool const is_type_position_long_received,
                                                                                                                        unsigned int const type_indicator_received,
                                                                                                                        unsigned int const number_outputs_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Set__Number_Outputs(number_outputs_received); }
            else { return(-1); }

            return(0);
        }
        
        DLL_API int API__Neural_Network__Set__Number_Time_Predictions(bool const is_type_position_long_received,
                                                                                                                                     unsigned int const type_indicator_received,
                                                                                                                                     unsigned int const number_recurrent_depth_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Set__Number_Recurrent_Depth(number_recurrent_depth_received); }
            else { return(-1); }

            return(0);
        }
        
        DLL_API int API__Neural_Network__Get__Is_Output_Symmetric(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { return(static_cast<int>(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Is_Output_Symmetric())); }
            else { return(-1); }
        }

        DLL_API int API__Neural_Network__Get__Path_Neural_Network_Exist(bool const is_type_position_long_received,
                                                                                                                                        unsigned int const type_indicator_received,
                                                                                                                                        unsigned int const type_neural_network_use_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { return(static_cast<int>(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Path_Neural_Network_Exist(static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received)))); }
            else { return(-1); }
        }

        DLL_API int API__Neural_Network__Get__Number_Inputs(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { return(static_cast<int>(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Number_Inputs(Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED))); }
            else { return(-1); }
        }

        DLL_API int API__Neural_Network__Get__Number_Outputs(bool const is_type_position_long_received,
                                                                                                                        unsigned int const type_indicator_received,
                                                                                                                        unsigned int const type_neural_network_use_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { return(static_cast<int>(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Number_Outputs(static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received)))); }
            else { return(-1); }
        }
        
        DLL_API int API__Neural_Network__Get__Number_Time_Predictions(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { return(static_cast<int>(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Number_Recurrent_Depth(Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED))); }
            else { return(-1); }
        }
        
        DLL_API float API__Neural_Network__Get__Loss(bool const is_type_position_long_received,
                                                                                                         unsigned int const type_indicator_received,
                                                                                                         unsigned int const type_neural_network_use_received,
                                                                                                         unsigned int const type_loss_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { return(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Loss(static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received), static_cast<enum MyEA::Common::ENUM_TYPE_DATASET>(type_loss_received))); }
            else { return(1.0f); }
        }
        
        DLL_API float API__Neural_Network__Get__Accuracy(bool const is_type_position_long_received,
                                                                                                                unsigned int const type_indicator_received,
                                                                                                                unsigned int const type_neural_network_use_received,
                                                                                                                unsigned int const type_accuracy_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { return(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Accuracy(static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received), static_cast<enum MyEA::Common::ENUM_TYPE_DATASET>(type_accuracy_received))); }
            else { return(0.0f); }
        }
        
        DLL_API T_ API__Neural_Network__Get__Output(bool const is_type_position_long_received,
                                                                                                          unsigned int const type_indicator_received,
                                                                                                          unsigned int const type_neural_network_use_received,
                                                                                                          unsigned int const output_index_received,
                                                                                                          unsigned int const time_step_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                class Neural_Network *const tmp_ptr_neural_network(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Neural_Network(static_cast<MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received)));

                if(tmp_ptr_neural_network == nullptr) { return((std::numeric_limits<ST_>::max)()); }

                if(output_index_received >= tmp_ptr_neural_network->Get__Output_Size()) { return((std::numeric_limits<ST_>::max)()); }
                
                if(time_step_received >= tmp_ptr_neural_network->number_recurrent_depth) { return((std::numeric_limits<ST_>::max)()); }
                
                return(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Output(time_step_received, static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received))[output_index_received]);
            }
            else { return((std::numeric_limits<ST_>::max)()); }
        }
        
        bool Preprocessing__Pre(enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicators_received,
                                             enum ENUM_TYPE_INPUT const type_input_received,
                                             T_ *const ptr_array_source_inputs_received,
                                             T_ *&ptr_array_inputs_received,
                                             class Dataset_Manager<T_> *const ptr_Dataset_Manager_received)
        {
            if(PREPROCESSING == false)
            {
                ptr_array_inputs_received = ptr_array_source_inputs_received;
                return(true);
            }
            
            switch(type_indicators_received)
            {
                case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_AUTOENCODER:
                    if(ptr_Dataset_Manager_received->Get__Number_Inputs() == 12_zu || ptr_Dataset_Manager_received->Get__Number_Inputs() == 16_zu)
                    {
                        // Momentum.
                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(0_zu,
                                                                                                                        ptr_array_source_inputs_received,
                                                                                                                        type_input_received) == false) { return(false); }
                        // |END| Momentum. |END|
                        
                        // BearsPower.
                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(4_zu,
                                                                                                                        ptr_array_source_inputs_received,
                                                                                                                        type_input_received) == false) { return(false); }
                        // |END| BearsPower. |END|

                        // BullsPower.
                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(5_zu,
                                                                                                                        ptr_array_source_inputs_received,
                                                                                                                        type_input_received) == false) { return(false); }
                        // |END| BullsPower. |END|

                        // ATR.
                        if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(7_zu,
                                                                                                                                ptr_array_source_inputs_received,
                                                                                                                                type_input_received) == false) { return(false); }
                        // |END| ATR. |END|

                        // StdDev.
                        if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(8_zu,
                                                                                                                                ptr_array_source_inputs_received,
                                                                                                                                type_input_received) == false) { return(false); }
                        // |END| StdDev. |END|

                        // MovingAverageOfOscillator.
                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(9_zu,
                                                                                                                        ptr_array_source_inputs_received,
                                                                                                                        type_input_received) == false) { return(false); }
                        // |END| MovingAverageOfOscillator. |END|

                        // AcceleratorOscillator.
                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(10_zu,
                                                                                                                        ptr_array_source_inputs_received,
                                                                                                                        type_input_received) == false) { return(false); }
                        // |END| AcceleratorOscillator. |END|

                        // AwesomeOscillator.
                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(11_zu,
                                                                                                                        ptr_array_source_inputs_received,
                                                                                                                        type_input_received) == false) { return(false); }
                        // |END| AwesomeOscillator. |END|

                        // OHLC.
                        if(ptr_Dataset_Manager_received->Get__Number_Inputs() == 16_zu)
                        {
                            // OHLC, open.
                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(12_zu,
                                                                                                                            ptr_array_source_inputs_received,
                                                                                                                            type_input_received) == false) { return(false); }
                            // |END| OHLC, open. |END|

                            // OHLC, high.
                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(13_zu,
                                                                                                                            ptr_array_source_inputs_received,
                                                                                                                            type_input_received) == false) { return(false); }
                            // |END| OHLC, high. |END|

                            // OHLC, low.
                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(14_zu,
                                                                                                                            ptr_array_source_inputs_received,
                                                                                                                            type_input_received) == false) { return(false); }
                            // |END| OHLC, low. |END|

                            // OHLC, close.
                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(15_zu,
                                                                                                                            ptr_array_source_inputs_received,
                                                                                                                            type_input_received) == false) { return(false); }
                            // |END| OHLC, close. |END|
                        }

                        ptr_array_inputs_received = ptr_array_source_inputs_received;
                        // |END| OHLC. |END|
                    }
                    else if(ptr_Dataset_Manager_received->Get__Number_Inputs() == 6_zu)
                    {
                        // Price.
                        for(size_t tmp_index(0_zu); tmp_index != 3_zu; ++tmp_index)
                        {
                            if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(tmp_index,
                                                                                                                            ptr_array_source_inputs_received,
                                                                                                                            type_input_received) == false) { return(false); }
                        }
                        // |END| Price. |END|

                        // ATR.
                        if(ptr_Dataset_Manager_received->Preprocessing__Minimum_Maximum(4_zu,
                                                                                                                                ptr_array_source_inputs_received,
                                                                                                                                type_input_received) == false) { return(false); }
                        // |END| ATR. |END|

                        // Momentum.
                        if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(5_zu,
                                                                                                                        ptr_array_source_inputs_received,
                                                                                                                        type_input_received) == false) { return(false); }
                        // |END| Momentum. |END|
                        
                        ptr_array_inputs_received = ptr_array_source_inputs_received;
                    }
                    else if((MODWT && ptr_Dataset_Manager_received->Get__Number_Inputs() == 27_zu)
                              ||
                              (MODWT == false && ptr_Dataset_Manager_received->Get__Number_Inputs() == 3_zu))
                    {
                        if(MODWT)
                        {
                            size_t const tmp_J_level(MyEA::Math::Minimum<size_t>(8_zu, ptr_Dataset_Manager_received->MODWT__J_Level_Maximum())),
                                               tmp_number_recurrent_depth(ptr_Dataset_Manager_received->Get__Number_Recurrent_Depth()),
                                               tmp_old_input_size(ptr_Dataset_Manager_received->Get__Number_Inputs() / (tmp_J_level + 1_zu)),
                                               tmp_new_input_size(ptr_Dataset_Manager_received->Get__Number_Inputs());
                            size_t tmp_shift_index,
                                      tmp_time_step_index,
                                      tmp_input_index;
                            
                            // Allocated merged inputs.
                            //  Initialize shifted index input position.
                            tmp_shift_index = 0_zu;

                            if((ptr_array_inputs_received = new T_[tmp_number_recurrent_depth * tmp_new_input_size]) == nullptr)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                                            MyEA::Time::Date_Time_Now().c_str(),
                                                            __FUNCTION__,
                                                            tmp_number_recurrent_depth * tmp_new_input_size * sizeof(T_),
                                                            __LINE__);

                                return(false);
                            }
                            MEMSET(ptr_array_inputs_received,
                                           0,
                                           tmp_number_recurrent_depth * tmp_new_input_size * sizeof(T_));

                            //  Convert small inputs dimensions into the large inputs dimensions.
                            for(tmp_time_step_index = 0_zu; tmp_time_step_index != tmp_number_recurrent_depth; ++tmp_time_step_index)
                            {
                                for(tmp_input_index = 0_zu; tmp_input_index != tmp_old_input_size; ++tmp_input_index)
                                {
                                    ptr_array_inputs_received[tmp_time_step_index * tmp_new_input_size + (tmp_input_index * (tmp_J_level + 1_zu))] = ptr_array_source_inputs_received[tmp_time_step_index * tmp_old_input_size + tmp_input_index];
                                }
                            }
                            //  |END| Convert small inputs dimensions into the large inputs dimensions. |END|
                            // |END| Allocated merged inputs. |END|
                        
                            // Price.
                            for(size_t tmp_index(0_zu); tmp_index != 3_zu; ++tmp_index)
                            {
                                if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(tmp_index + tmp_shift_index,
                                                                                                                                ptr_array_inputs_received,
                                                                                                                                type_input_received) == false)
                                {
                                    delete[](ptr_array_inputs_received);

                                    return(false);
                                }

                                tmp_shift_index += tmp_J_level;
                            }
                            // |END| Price. |END|
                        
                            // Allocated merged inputs.
                            //  Initialize shifted index input position.
                            tmp_shift_index = 0_zu;

                            //  Convert large inputs dimensions into the small inputs dimensions.
                            for(tmp_time_step_index = 0_zu; tmp_time_step_index != tmp_number_recurrent_depth; ++tmp_time_step_index)
                            {
                                for(tmp_input_index = 0_zu; tmp_input_index != tmp_old_input_size; ++tmp_input_index)
                                {
                                    ptr_array_source_inputs_received[tmp_time_step_index * tmp_old_input_size + tmp_input_index] = ptr_array_inputs_received[tmp_time_step_index * tmp_new_input_size + (tmp_input_index * (tmp_J_level + 1_zu))];
                                }
                            }
                            //  |END| Convert large inputs dimensions into the small inputs dimensions. |END|
                            
                            //  Change the large inputs dimensions into a small inputs dimensions.
                            delete[](ptr_array_inputs_received);

                            if((ptr_array_inputs_received = new T_[tmp_number_recurrent_depth * tmp_old_input_size]) == nullptr)
                            {
                                PRINT_FORMAT("%s: %s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                                         MyEA::Time::Date_Time_Now().c_str(),
                                                         __FUNCTION__,
                                                         tmp_number_recurrent_depth * tmp_old_input_size * sizeof(T_),
                                                         __LINE__);

                                return(false);
                            }
                        
                            MEMCPY(ptr_array_inputs_received,
                                            ptr_array_source_inputs_received,
                                            tmp_number_recurrent_depth * tmp_old_input_size * sizeof(T_));
                            //  |END| Change the large inputs dimensions into a small inputs dimensions. |END|
                            // |END| Allocated merged inputs. |END|
                        
                            // Price.
                            for(size_t tmp_index(0_zu); tmp_index != 3_zu; ++tmp_index)
                            {
                                if(ptr_Dataset_Manager_received->Preprocessing__Merge__MODWT(tmp_index + tmp_shift_index,
                                                                                                                                    tmp_old_input_size,
                                                                                                                                    ptr_array_inputs_received,
                                                                                                                                    type_input_received) == false)
                                {
                                    delete[](ptr_array_inputs_received);

                                    return(false);
                                }

                                tmp_shift_index += tmp_J_level;
                            }
                            // |END| Price. |END|
                        }
                        else
                        {
                            // Price.
                            for(size_t tmp_index(0_zu); tmp_index != 3_zu; ++tmp_index)
                            {
                                if(ptr_Dataset_Manager_received->Preprocessing__Zero_Centered(tmp_index,
                                                                                                                                ptr_array_source_inputs_received,
                                                                                                                                type_input_received) == false) { return(false); }
                            }
                            // |END| Price. |END|

                            ptr_array_inputs_received = ptr_array_source_inputs_received;
                        }
                    }
                        break;
                default: break;
            }
            
            return(true);
        }
        
        bool Preprocessing__Pre(enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicators_received,
                                             T_ *const ptr_array_source_inputs_received,
                                             T_ *const ptr_array_source_outputs_received,
                                             T_ *&ptr_array_inputs_received,
                                             T_ *&ptr_array_outputs_received,
                                             class Dataset_Manager<T_> *const ptr_Dataset_Manager_received)
        {
            if(ptr_array_source_inputs_received != nullptr)
            {
                if(Preprocessing__Pre(type_indicators_received,
                                                 ENUM_TYPE_INPUT::TYPE_INPUT_INPUT,
                                                 ptr_array_source_inputs_received,
                                                 ptr_array_inputs_received,
                                                 ptr_Dataset_Manager_received) == false) { return(false); }
            }
            
            if(ptr_array_source_outputs_received != nullptr)
            {
                if(Preprocessing__Pre(type_indicators_received,
                                                 ENUM_TYPE_INPUT::TYPE_INPUT_OUTPUT,
                                                 ptr_array_source_outputs_received,
                                                 ptr_array_outputs_received,
                                                 ptr_Dataset_Manager_received) == false) { return(false); }
            }

            return(true);
        }
        
        bool Preprocessing__Post(enum MyEA::Common::ENUM_TYPE_INDICATORS const type_indicators_received,
                                               T_ *&ptr_array_inputs_received,
                                               T_ *&ptr_array_outputs_received,
                                               class Dataset_Manager<T_> *const ptr_Dataset_Manager_received)
        {
            if(PREPROCESSING == false) { return(true); }
            
            switch(type_indicators_received)
            {
                case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_AUTOENCODER:
                    if(ptr_Dataset_Manager_received->Get__Number_Inputs() == 27_zu)
                    {
                        if(MODWT)
                        {
                            SAFE_DELETE_ARRAY(ptr_array_inputs_received);
                            SAFE_DELETE_ARRAY(ptr_array_outputs_received);
                        }
                    }
                        break;
                default: break;
            }

            return(true);
        }

        DLL_API bool API__Neural_Network__Forward_Pass(bool const is_type_position_long_received,
                                                                                                              unsigned int const type_indicator_received,
                                                                                                              unsigned int const type_neural_network_use_received,
                                                                                                              T_ *const ptr_array_inputs_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                class Neural_Network *const tmp_ptr_neural_network(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Neural_Network(static_cast<MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received)));

                if(tmp_ptr_neural_network == nullptr) { return(false); }
                
                class Dataset_Manager<T_> *const tmp_ptr_Dataset_Manager(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Dataset_Manager());

                if(tmp_ptr_Dataset_Manager == nullptr) { return(false); }

                T_ const **tmp_ptr_array_inputs_array(new T_ const *[1u]);
                T_ *tmp_ptr_array_inputs(nullptr),
                     *tmp_ptr_array_outputs(nullptr);

                if(tmp_ptr_array_inputs_array == nullptr) { return(false); }

                if(Preprocessing__Pre(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received),
                                                ptr_array_inputs_received,
                                                nullptr,
                                                tmp_ptr_array_inputs,
                                                tmp_ptr_array_outputs,
                                                tmp_ptr_Dataset_Manager) == false)
                {
                    delete[](tmp_ptr_array_inputs_array);

                    return(false);
                }

                tmp_ptr_array_inputs_array[0u] = tmp_ptr_array_inputs;
                
                tmp_ptr_neural_network->Forward_Pass(1_zu, tmp_ptr_array_inputs_array);
                
                if(Preprocessing__Post(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received),
                                                  tmp_ptr_array_inputs,
                                                  tmp_ptr_array_outputs,
                                                  tmp_ptr_Dataset_Manager) == false) { return(false); }

                delete[](tmp_ptr_array_inputs_array);

                return(true);
            }
            else { return(false); }
        }

        DLL_API bool API__Neural_Network__Append_To_Dataset_File(bool const is_type_position_long_received,
                                                                                                                              unsigned int const type_indicator_received,
                                                                                                                              T_ *ptr_array_inputs_received,
                                                                                                                              T_ *ptr_array_outputs_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { return(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Append_To_Dataset_History(ptr_array_inputs_received, ptr_array_outputs_received)); }
            else { return(false); }
        }

        DLL_API bool API__Neural_Network__Append_To_Dataset(bool const is_type_position_long_received,
                                                                                                                      unsigned int const type_indicator_received,
                                                                                                                      T_ *const ptr_array_inputs_received,
                                                                                                                      T_ *const ptr_array_outputs_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                class Dataset_Manager<T_> *const tmp_ptr_Dataset_Manager(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Dataset_Manager());

                if(tmp_ptr_Dataset_Manager == nullptr) { return(false); }
                
                T_ *tmp_ptr_array_inputs(nullptr),
                     *tmp_ptr_array_outputs(nullptr);
                
                if(Preprocessing__Pre(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received),
                                                ptr_array_inputs_received,
                                                ptr_array_outputs_received,
                                                tmp_ptr_array_inputs,
                                                tmp_ptr_array_outputs,
                                                tmp_ptr_Dataset_Manager) == false) { return(false); }

                // Append to dataset.
                if(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Append_To_Dataset(tmp_ptr_array_inputs, tmp_ptr_array_outputs) == false) { return(false); }
                
                if(Preprocessing__Post(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received),
                                                  tmp_ptr_array_inputs,
                                                  tmp_ptr_array_outputs,
                                                  tmp_ptr_Dataset_Manager) == false) { return(false); }

                return(true);
            }
            else { return(false); }
        }
        
        DLL_API bool API__Neural_Network__Join(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                if(tmp_ptr_Threaded_Neural_Network_Manager->optimizer_thread.joinable())
                {
                    tmp_ptr_Threaded_Neural_Network_Manager->optimizer_thread.join();
                    
                    // Compare.
                    switch(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received))
                    {
                        case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_AUTOENCODER:
                            // SAE compare trained. If success, update the LSTM dataset input.
                            if(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Compare_Trained__Pre_Training())
                            {
                                // Join LSTM before replacing entries into his dataset.
                                if(API__Neural_Network__Join(is_type_position_long_received, MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iMA) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"API__Neural_Network__Join(%s, %d)\" function. At line %d." NEW_LINE,
                                                             MyEA::Time::Date_Time_Now().c_str(),
                                                             __FUNCTION__,
                                                             is_type_position_long_received ? "true" : "false",
                                                             MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iMA,
                                                             __LINE__);

                                    return(false);
                                }
                                
                                class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_LSTM_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                         is_type_position_long_received,
                                                                                                                                                                                                                         MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_iMA));

                                // Replace LSTM dataset.
                                if(tmp_ptr_Threaded_LSTM_Manager->neural_network_manager.Get__Dataset_Manager()->Replace_Entries(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Dataset_Manager(), tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Neural_Network(MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED)) == false)
                                {
                                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Replace_Entries(ptr, ptr)\" function. At line %d." NEW_LINE,
                                                             MyEA::Time::Date_Time_Now().c_str(),
                                                             __FUNCTION__,
                                                             __LINE__);

                                    return(false);
                                }
                            }
                                break;
                        default: tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Compare_Trained(); break;
                    }
                }

                return(true);
            }
            else { return(false); }
        }
        
        DLL_API bool API__Neural_Network__Optimization(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                if(tmp_ptr_Threaded_Neural_Network_Manager->optimizer_thread.joinable() == false)
                {
                    struct MyEA::Common::While_Condition tmp_While_Condition;

                    tmp_While_Condition.type_while_condition = MyEA::Common::ENUM_TYPE_WHILE_CONDITION::TYPE_WHILE_CONDITION_EXPIRATION;
                    
                    // Current_Time = Now + 15s(delay). Sometime the tick return at 11h:59m:56s and not at 12h:00m:00s.
                    std::time_t const tmp_time_t(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) + 15);
                    std::tm *const tmp_ptr_tm(std::localtime(&tmp_time_t));
                    
                    tmp_ptr_tm->tm_sec = 0;
                    tmp_ptr_tm->tm_min = 0;
                    
                    //tmp_ptr_tm->tm_min -= tmp_ptr_tm->tm_min % 15;
                    
                    // Time normalized + 1h(tick) - 60s(gaps);
                    tmp_While_Condition.expiration = std::chrono::system_clock::from_time_t(mktime(tmp_ptr_tm) + MyEA::Common::ENUM_TIME_FRAMES::TIME_FRAMES_PERIOD_H1 - 60);
                    
                    if(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Set__While_Condition_Optimization(tmp_While_Condition) == false) { return(false); }
                    
                    // Optimize and test if require.
                    switch(static_cast<enum MyEA::Common::ENUM_TYPE_INDICATORS>(type_indicator_received))
                    {
                        case MyEA::Common::ENUM_TYPE_INDICATORS::TYPE_AUTOENCODER:
                            if(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Testing_If_Require__Pre_Training() == false) { return(false); }
                            
                            tmp_ptr_Threaded_Neural_Network_Manager->optimizer_thread = std::thread([&tmp_neural_network_manager = tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager]() { tmp_neural_network_manager.Pre_Training(); } );
                                break;
                        default:
                            if(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Testing_If_Require() == false) { return(false); }
                            
                            tmp_ptr_Threaded_Neural_Network_Manager->optimizer_thread = std::thread(&MyEA::Neural_Network::Neural_Network_Manager::Optimization, &tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager);
                                break;
                    }
                }

                return(true);
            }
            else { return(false); }
        }
        
        DLL_API bool API__Neural_Network__Write_File(bool const is_type_position_long_received,
                                                                                                        unsigned int const type_indicator_received,
                                                                                                        unsigned int const type_file_log_received,
                                                                                                        wchar_t const *const log_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                std::wstring const tmp_wchar_to_wstring(log_received);

                return(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Write_File(static_cast<enum MyEA::Common::ENUM_TYPE_FILE_LOG>(type_file_log_received), MyEA::Time::Date_Time_Now() + ": " + std::string(tmp_wchar_to_wstring.begin(), tmp_wchar_to_wstring.end())));
            }
            else { return(false); }
        }
        
        DLL_API bool API__Neural_Network__Load_Neural_Network(bool const is_type_position_long_received,
                                                                                                                         unsigned int const type_indicator_received,
                                                                                                                         unsigned int const type_neural_network_use_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));
            
        #if defined(COMPILE_CUDA)
            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                if(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Neural_Network(static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received)) != nullptr) { return(true); }

                return(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Load_Neural_Network(static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received),
                                                                                                                                                                   256_zu * KILOBYTE * KILOBYTE,
                                                                                                                                                                   256_zu * KILOBYTE * KILOBYTE,
                                                                                                                                                                   static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received) == MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED));
            }
        #else
            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                if(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Get__Neural_Network(static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received)) != nullptr) { return(true); }

                return(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Load_Neural_Network(static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received),
                                                                                                                                                                  256_zu * KILOBYTE * KILOBYTE,
                                                                                                                                                                  static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received) == MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE::TYPE_NEURAL_NETWORK_TRAINED));
            }
        #endif
            else { return(false); }
        }

        DLL_API bool API__Neural_Network__Save_Neural_Network(bool const is_type_position_long_received,
                                                                                                                         unsigned int const type_indicator_received,
                                                                                                                         unsigned int const type_neural_network_use_received)
        {
            class Threaded_Neural_Network_Manager *const tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                                    is_type_position_long_received,
                                                                                                                                                                                                                    type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr) { return(tmp_ptr_Threaded_Neural_Network_Manager->neural_network_manager.Save_Neural_Network(static_cast<enum MyEA::Common::ENUM_TYPE_NEURAL_NETWORK_USE>(type_neural_network_use_received))); }
            else { return(false); }
        }

        DLL_API bool API__Neural_Network__Deinitialize(bool const is_type_position_long_received, unsigned int const type_indicator_received)
        {
            if(global_ptr_vector_ptr_Threaded_Neural_Network_Manager == nullptr) { return(false); }

            class Threaded_Neural_Network_Manager *tmp_ptr_Threaded_Neural_Network_Manager(Get__Threaded_Neural_Network_Manager(global_ptr_vector_ptr_Threaded_Neural_Network_Manager,
                                                                                                                                                                                                           is_type_position_long_received,
                                                                                                                                                                                                           type_indicator_received));

            if(tmp_ptr_Threaded_Neural_Network_Manager != nullptr)
            {
                // Find from vector.
                auto tmp_ptr_iteration_Threaded_Neural_Network_Manager(std::find(global_ptr_vector_ptr_Threaded_Neural_Network_Manager->begin(),
                                                                                                                   global_ptr_vector_ptr_Threaded_Neural_Network_Manager->end(),
                                                                                                                   tmp_ptr_Threaded_Neural_Network_Manager));

                // Remove from vector.
                if(tmp_ptr_iteration_Threaded_Neural_Network_Manager != global_ptr_vector_ptr_Threaded_Neural_Network_Manager->end()) { global_ptr_vector_ptr_Threaded_Neural_Network_Manager->erase(tmp_ptr_iteration_Threaded_Neural_Network_Manager); }

                // Deallocate.
                SAFE_DELETE(tmp_ptr_Threaded_Neural_Network_Manager);
            }

            if(global_ptr_vector_ptr_Threaded_Neural_Network_Manager->empty()) { SAFE_DELETE(global_ptr_vector_ptr_Threaded_Neural_Network_Manager); }

            return(true);
        }
        
        DLL_API bool API__Neural_Network__Allocate(void)
        {
            if(global_ptr_vector_ptr_Threaded_Neural_Network_Manager == nullptr)
            {
                global_ptr_vector_ptr_Threaded_Neural_Network_Manager = new std::vector<class Threaded_Neural_Network_Manager *>();

                return(true);
            }
            else { return(false); }
        }
    }
}