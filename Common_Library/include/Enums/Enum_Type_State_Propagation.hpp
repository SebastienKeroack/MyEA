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

#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_STATE_PROPAGATION : unsigned int
        {
            TYPE_STATE_PROPAGATION_INFERENCE = 0u,
            TYPE_STATE_PROPAGATION_TRAINING = 1u,
            TYPE_STATE_PROPAGATION_UPDATE_BATCH_NORMALIZATION = 2u,
            TYPE_STATE_PROPAGATION_LENGTH = 3u
        };
        
        static std::map<enum ENUM_TYPE_STATE_PROPAGATION, std::string> ENUM_TYPE_STATE_PROPAGATION_NAMES = {{TYPE_STATE_PROPAGATION_INFERENCE, "Inference"},
                                                                                                                                                                                              {TYPE_STATE_PROPAGATION_TRAINING, "Training"},
                                                                                                                                                                                              {TYPE_STATE_PROPAGATION_UPDATE_BATCH_NORMALIZATION, "Update batch-normalization"},
                                                                                                                                                                                              {TYPE_STATE_PROPAGATION_LENGTH, "LENGTH"}};
    }
}
