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
        enum ENUM_TYPE_WEIGHTS_INITIALIZERS : unsigned int
        {
            TYPE_WEIGHTS_INITIALIZER_NONE = 0u,
            TYPE_WEIGHTS_INITIALIZER_GLOROT_GAUSSIAN = 1u, // http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf "Understanding the difficulty of training deep feedforward neural networks".
            TYPE_WEIGHTS_INITIALIZER_GLOROT_UNIFORM = 2u, // http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf "Understanding the difficulty of training deep feedforward neural networks".
            TYPE_WEIGHTS_INITIALIZER_IDENTITY = 3u,
            TYPE_WEIGHTS_INITIALIZER_LSUV = 4u, // https://arxiv.org/abs/1511.06422 "All you need is a good init".
            TYPE_WEIGHTS_INITIALIZER_ORTHOGONAL = 5u, // https://arxiv.org/abs/1312.6120 "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks".
            TYPE_WEIGHTS_INITIALIZER_UNIFORM = 6u,
            TYPE_WEIGHTS_INITIALIZER_LENGTH = 7u
        };

        static std::map<enum ENUM_TYPE_WEIGHTS_INITIALIZERS, std::string> ENUM_TYPE_WEIGHTS_INITIALIZERS_NAMES = {{TYPE_WEIGHTS_INITIALIZER_NONE, "NONE"},
                                                                                                                                                                                                  {TYPE_WEIGHTS_INITIALIZER_GLOROT_GAUSSIAN, "Glorot gaussian"},
                                                                                                                                                                                                  {TYPE_WEIGHTS_INITIALIZER_GLOROT_UNIFORM, "Glorot uniform"},
                                                                                                                                                                                                  {TYPE_WEIGHTS_INITIALIZER_IDENTITY, "Identity"},
                                                                                                                                                                                                  {TYPE_WEIGHTS_INITIALIZER_LSUV, "[x] LSUV"},
                                                                                                                                                                                                  {TYPE_WEIGHTS_INITIALIZER_ORTHOGONAL, "Orthogonal"},
                                                                                                                                                                                                  {TYPE_WEIGHTS_INITIALIZER_UNIFORM, "Uniform"},
                                                                                                                                                                                                  {TYPE_WEIGHTS_INITIALIZER_LENGTH, "LENGTH"}};

    }
}
