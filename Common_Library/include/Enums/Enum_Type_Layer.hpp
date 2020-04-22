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
        enum ENUM_TYPE_LAYER : unsigned int
        {
            TYPE_LAYER_NONE = 0u,
            TYPE_LAYER_AVERAGE_POOLING = 1u,
            TYPE_LAYER_CONVOLUTION = 2u,
            TYPE_LAYER_DENSE = 3u,
            TYPE_LAYER_FULLY_CONNECTED = 4u,
            TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT = 5u,
            TYPE_LAYER_FULLY_CONNECTED_RECURRENT = 6u,
            TYPE_LAYER_GRU = 7u,
            TYPE_LAYER_LSTM = 8u,
            TYPE_LAYER_MAX_POOLING = 9u,
            TYPE_LAYER_RESIDUAL = 10u,
            TYPE_LAYER_SHORTCUT = 11u,
            TYPE_LAYER_LENGTH = 12u
        };

        static std::map<enum ENUM_TYPE_LAYER, std::string> ENUM_TYPE_LAYER_NAME = {{TYPE_LAYER_NONE, "NONE"},
                                                                                                                                             {TYPE_LAYER_AVERAGE_POOLING, "Average pooling"},
                                                                                                                                             {TYPE_LAYER_CONVOLUTION, "[x] Convolution"},
                                                                                                                                             {TYPE_LAYER_DENSE, "[x] Dense"},
                                                                                                                                             {TYPE_LAYER_FULLY_CONNECTED, "Fully connected"},
                                                                                                                                             {TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT, "Fully connected, independently recurrent"},
                                                                                                                                             {TYPE_LAYER_FULLY_CONNECTED_RECURRENT, "[x] Fully connected, recurrent"},
                                                                                                                                             {TYPE_LAYER_GRU, "[x] Gated recurrent unit"},
                                                                                                                                             {TYPE_LAYER_LSTM, "Long short-term memory"},
                                                                                                                                             {TYPE_LAYER_MAX_POOLING, "Max pooling"},
                                                                                                                                             {TYPE_LAYER_RESIDUAL, "Residual"},
                                                                                                                                             {TYPE_LAYER_SHORTCUT, "[x] Shorcut"},
                                                                                                                                             {TYPE_LAYER_LENGTH, "LENGTH"}};
        
        static std::map<enum ENUM_TYPE_LAYER, std::string> ENUM_TYPE_LAYER_CONNECTION_NAME = {{TYPE_LAYER_NONE, "NONE"},
                                                                                                                                                                   {TYPE_LAYER_AVERAGE_POOLING, "connected_to_basic"},
                                                                                                                                                                   {TYPE_LAYER_CONVOLUTION, "connected_to_convolution"},
                                                                                                                                                                   {TYPE_LAYER_DENSE, "connected_to_basic"},
                                                                                                                                                                   {TYPE_LAYER_FULLY_CONNECTED, "connected_to_neuron"},
                                                                                                                                                                   {TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT, "connected_to_AF_Ind_R"},
                                                                                                                                                                   {TYPE_LAYER_FULLY_CONNECTED_RECURRENT, "connected_to_neuron"},
                                                                                                                                                                   {TYPE_LAYER_GRU, "connected_to_cell"},
                                                                                                                                                                   {TYPE_LAYER_LSTM, "connected_to_cell"},
                                                                                                                                                                   {TYPE_LAYER_MAX_POOLING, "connected_to_basic_indice"},
                                                                                                                                                                   {TYPE_LAYER_RESIDUAL, "connected_to_basic"},
                                                                                                                                                                   {TYPE_LAYER_SHORTCUT, "connected_to_neuron"},
                                                                                                                                                                   {TYPE_LAYER_LENGTH, "LENGTH"}};

        enum ENUM_TYPE_GROUP : unsigned int
        {
            TYPE_GROUP_NONE = 0u,
            TYPE_GROUP_RESIDUAL = 1u,
            TYPE_GROUP_LENGTH = 2u
        };
        
        static std::map<enum ENUM_TYPE_GROUP, std::string> ENUM_TYPE_GROUP_NAME = {{TYPE_GROUP_NONE, "NONE"},
                                                                                                                                               {TYPE_GROUP_RESIDUAL, "Residual"},
                                                                                                                                               {TYPE_GROUP_LENGTH, "LENGTH"}};
        
    }
}