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
        enum ENUM_TYPE_LAYER_ACTIVATION : unsigned int
        {
            TYPE_ACTIVATION_NONE = 0u,
            TYPE_ACTIVATION_ASYMMETRIC = 1u,
            TYPE_ACTIVATION_RECTIFIER = 2u,
            TYPE_ACTIVATION_SELF_NORMALIZATION = 3u,
            TYPE_ACTIVATION_SOFTMAX = 4u,
            TYPE_ACTIVATION_SYMMETRIC = 5u,
            TYPE_ACTIVATION_LENGTH = 6u
        };

        static std::map<enum ENUM_TYPE_LAYER_ACTIVATION, std::string> ENUM_TYPE_LAYER_ACTIVATION_NAME = {{TYPE_ACTIVATION_NONE, "NONE"},
                                                                                                                                                                                     {TYPE_ACTIVATION_ASYMMETRIC, "Asymmetric"},
                                                                                                                                                                                     {TYPE_ACTIVATION_RECTIFIER, "Rectifier"},
                                                                                                                                                                                     {TYPE_ACTIVATION_SELF_NORMALIZATION, "Self-normalization"},
                                                                                                                                                                                     {TYPE_ACTIVATION_SOFTMAX, "Softmax"},
                                                                                                                                                                                     {TYPE_ACTIVATION_SYMMETRIC, "Symmetric"},
                                                                                                                                                                                     {TYPE_ACTIVATION_LENGTH, "LENGTH"}};
    }
}