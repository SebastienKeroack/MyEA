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
        enum ENUM_TYPE_DATASET_FILE : unsigned int
        {
            TYPE_DATASET_FILE_DATASET = 0u,
            TYPE_DATASET_FILE_DATASET_SPLIT = 1u,
            TYPE_DATASET_FILE_MNIST = 2u, // National Institute of Standards and Technology (NIST)
            TYPE_DATASET_FILE_LENGTH = 3u
        };

        static std::map<enum ENUM_TYPE_DATASET_FILE, std::string> ENUM_TYPE_DATASET_FILE_NAMES = {{TYPE_DATASET_FILE_DATASET, "Dataset"},
                                                                                                                                                                       {TYPE_DATASET_FILE_DATASET_SPLIT, "Dataset split"},
                                                                                                                                                                       {TYPE_DATASET_FILE_MNIST, "MNIST"},
                                                                                                                                                                       {TYPE_DATASET_FILE_LENGTH, "LENGTH"}};
    }
}
