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