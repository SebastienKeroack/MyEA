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

namespace MyEA::Common
{
    enum ENUM_TYPE_DATASET : unsigned int
    {
        TYPE_DATASET_NONE       = 0u,
        TYPE_DATASET_TESTING    = 1u,
        TYPE_DATASET_TRAINING   = 2u,
        TYPE_DATASET_VALIDATION = 3u,
        TYPE_DATASET_LENGTH     = 4u
    };

    static std::map<enum ENUM_TYPE_DATASET, std::string> ENUM_TYPE_DATASET_NAMES = {
        {TYPE_DATASET_NONE,       "None"},
        {TYPE_DATASET_TESTING,    "Testing"},
        {TYPE_DATASET_TRAINING,   "Training"},
        {TYPE_DATASET_VALIDATION, "Validating"},
        {TYPE_DATASET_LENGTH,     "LENGTH"}
    };

}