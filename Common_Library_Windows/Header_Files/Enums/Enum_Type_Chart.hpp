#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_CHART : unsigned int
        {
            TYPE_CHART_LOSS = 0u,
            TYPE_CHART_ACCURACY = 1u,
            TYPE_CHART_OUTPUT = 2u,
            TYPE_CHART_GRID_SEARCH = 3u,
            TYPE_CHART_LENGTH = 4u
        };

        static std::map<enum ENUM_TYPE_CHART, std::string> ENUM_TYPE_CHART_NAMES = {{TYPE_CHART_LOSS, "TYPE_CHART_LOSS"},
                                                                                                                                              {TYPE_CHART_ACCURACY, "TYPE_CHART_ACCURACY"},
                                                                                                                                              {TYPE_CHART_OUTPUT, "TYPE_CHART_OUTPUT"},
                                                                                                                                              {TYPE_CHART_GRID_SEARCH, "TYPE_CHART_GRID_SEARCH"},
                                                                                                                                              {TYPE_CHART_LENGTH, "LENGTH"}};

    }
}