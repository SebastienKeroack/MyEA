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
        enum ENUM_TYPE_DATASET_PROCESS : unsigned int
        {
            TYPE_DATASET_PROCESS_NONE = 0u,
            TYPE_DATASET_PROCESS_BATCH = 1u,
            TYPE_DATASET_PROCESS_MINI_BATCH = 2u,
            TYPE_DATASET_PROCESS_CROSS_VALIDATION = 3u,
            TYPE_DATASET_PROCESS_CROSS_VALIDATION_RANDOM_SEARCH = 4u,
            TYPE_DATASET_PROCESS_LENGTH = 5u
        };

        static std::map<enum ENUM_TYPE_DATASET_PROCESS, std::string> ENUM_TYPE_DATASET_PROCESS_NAMES = {{TYPE_DATASET_PROCESS_NONE, "NONE"},
                                                                                                                                                                                         {TYPE_DATASET_PROCESS_BATCH, "Batch"},
                                                                                                                                                                                         {TYPE_DATASET_PROCESS_MINI_BATCH, "Mini batch"},
                                                                                                                                                                                         {TYPE_DATASET_PROCESS_CROSS_VALIDATION, "Cross validation"},
                                                                                                                                                                                         {TYPE_DATASET_PROCESS_CROSS_VALIDATION_RANDOM_SEARCH, "Cross validation, random search"},
                                                                                                                                                                                         {TYPE_DATASET_PROCESS_LENGTH, "LENGTH"}};

    }
}