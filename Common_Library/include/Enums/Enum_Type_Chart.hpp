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
        enum ENUM_TYPE_CHART
        {
            TYPE_CHART_NONE = 0,
            TYPE_CHART_LOSS = 1,
            TYPE_CHART_ACCURACY = 2,
            TYPE_CHART_OUTPUT = 3,
            TYPE_CHART_GRID_SEARCH = 4,
            TYPE_CHART_LENGTH = 5
        };

        static std::map<enum ENUM_TYPE_CHART, std::string> ENUM_TYPE_CHART_NAMES = {
            {TYPE_CHART_NONE, "None"},
            {TYPE_CHART_LOSS, "Loss"},
            {TYPE_CHART_ACCURACY, "Accuracy"},
            {TYPE_CHART_OUTPUT, "Output"},
            {TYPE_CHART_GRID_SEARCH, "Grid Search"},
            {TYPE_CHART_LENGTH, "Length"}
        };
    }
}