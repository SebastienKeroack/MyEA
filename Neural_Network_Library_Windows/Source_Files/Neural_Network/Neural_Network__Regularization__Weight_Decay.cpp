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

#include <Neural_Network/Neural_Network.hpp>

bool Neural_Network::Set__Regularization__Weight_Decay(T_ const regularization__weight_decay_received)
{
    if(regularization__weight_decay_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Weight decay (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(regularization__weight_decay_received),
                                 __LINE__);

        return(false);
    }
    else if(regularization__weight_decay_received > 1_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Weight decay (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(regularization__weight_decay_received),
                                 __LINE__);

        return(false);
    }

    if(this->regularization__weight_decay != regularization__weight_decay_received)
    {
        bool const tmp_use_regularization(this->Use__Regularization_Parameter());

        this->regularization__weight_decay = regularization__weight_decay_received;

        if(tmp_use_regularization == false && regularization__weight_decay_received != 0_T)
        {
            if(this->Allocate__Parameter__Regularization() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Parameter__Regularization()\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
            
            if(this->pre_training_level != 0_zu) { this->Indexing_Regularization_Parameters__Pre_training(); }
            else { this->Indexing_Regularization_Parameters(); }
        }

        if(this->Use__Regularization_Parameter() == false)
        { this->Deallocate__Parameter__Regularization(); }

    #if defined(COMPILE_CUDA)
        if(this->is_device_initialized)
        { this->ptr_device_Neural_Network->Set__Regularization__Weight_Decay(regularization__weight_decay_received); }
    #endif
    }

    return(true);
}
