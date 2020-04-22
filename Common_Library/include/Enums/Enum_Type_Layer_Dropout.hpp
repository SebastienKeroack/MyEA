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
        enum ENUM_TYPE_LAYER_DROPOUT : unsigned int
        {
            TYPE_LAYER_DROPOUT_NONE = 0u,
            TYPE_LAYER_DROPOUT_ALPHA = 1u, // https://arxiv.org/pdf/1706.02515v5.pdf
            TYPE_LAYER_DROPOUT_BERNOULLI = 2u,
            TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED = 3u,
            TYPE_LAYER_DROPOUT_GAUSSIAN = 4u,
            TYPE_LAYER_DROPOUT_SHAKEDROP = 5u, // https://arxiv.org/pdf/1802.02375.pdf
            TYPE_LAYER_DROPOUT_UOUT = 6u,
            TYPE_LAYER_DROPOUT_ZONEOUT = 7u,
            TYPE_LAYER_DROPOUT_LENGTH = 8u
        };

        static std::map<enum ENUM_TYPE_LAYER_DROPOUT, std::string> ENUM_TYPE_LAYER_DROPOUT_NAMES = {{TYPE_LAYER_DROPOUT_NONE, "NONE"},
                                                                                                                                                                                 {TYPE_LAYER_DROPOUT_ALPHA, "[x] Alpha"},
                                                                                                                                                                                 {TYPE_LAYER_DROPOUT_BERNOULLI, "Bernoulli"},
                                                                                                                                                                                 {TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED, "Bernoulli inverted"},
                                                                                                                                                                                 {TYPE_LAYER_DROPOUT_GAUSSIAN, "Gaussian"},
                                                                                                                                                                                 {TYPE_LAYER_DROPOUT_SHAKEDROP, "ShakeDrop"},
                                                                                                                                                                                 {TYPE_LAYER_DROPOUT_UOUT, "Uout"},
                                                                                                                                                                                 {TYPE_LAYER_DROPOUT_ZONEOUT, "Zoneout"},
                                                                                                                                                                                 {TYPE_LAYER_DROPOUT_LENGTH, "LENGTH"}};
    }
}