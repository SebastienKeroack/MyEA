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
        enum ENUM_TYPE_NEURAL_NETWORK_USE : unsigned int
        {
            TYPE_NEURAL_NETWORK_ALL = 0u,
            TYPE_NEURAL_NETWORK_TRAINER = 1u,
            TYPE_NEURAL_NETWORK_TRAINED = 2u,
            TYPE_NEURAL_NETWORK_COMPETITOR = 3u,
            TYPE_NEURAL_NETWORK_LENGTH = 4u
        };

        static std::map<enum ENUM_TYPE_NEURAL_NETWORK_USE, std::string> ENUM_TYPE_NEURAL_NETWORK_USE_NAMES = {{TYPE_NEURAL_NETWORK_ALL, "ALL"},
                                                                                                                                                                                                        {TYPE_NEURAL_NETWORK_TRAINER, "Trainer"},
                                                                                                                                                                                                        {TYPE_NEURAL_NETWORK_TRAINED, "Trained"},
                                                                                                                                                                                                        {TYPE_NEURAL_NETWORK_COMPETITOR, "Competitor"},
                                                                                                                                                                                                        {TYPE_NEURAL_NETWORK_LENGTH, "LENGTH"}};
    }
}