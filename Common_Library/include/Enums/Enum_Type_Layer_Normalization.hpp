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
        enum ENUM_TYPE_LAYER_NORMALIZATION : unsigned int
        {
            TYPE_LAYER_NORMALIZATION_NONE = 0u,
            TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION = 1u,
            TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION = 2u,
            TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION = 3u,
            TYPE_LAYER_NORMALIZATION_STREAMING_NORMALIZATION = 4u,
            TYPE_LAYER_NORMALIZATION_LENGTH = 5u
        };

        static std::map<enum ENUM_TYPE_LAYER_NORMALIZATION, std::string> ENUM_TYPE_LAYER_NORMALIZATION_NAMES = {{TYPE_LAYER_NORMALIZATION_NONE, "NONE"},
                                                                                                                                                                                                    {TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION, "Batch normalization"},
                                                                                                                                                                                                    {TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION, "Batch renormalization"},
                                                                                                                                                                                                    {TYPE_LAYER_NORMALIZATION_GHOST_BATCH_NORMALIZATION, "[x] Ghost batch normalization"},
                                                                                                                                                                                                    {TYPE_LAYER_NORMALIZATION_STREAMING_NORMALIZATION, "[x] Streaming normalization"},
                                                                                                                                                                                                    {TYPE_LAYER_NORMALIZATION_LENGTH, "LENGTH"}};

    }
}