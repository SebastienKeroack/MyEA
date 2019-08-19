#pragma once

#include <string>
#include <map>

namespace MyEA::Common
{
    enum ENUM_TYPE_DATASET : unsigned int
    {
        TYPE_DATASET_TESTING    = 0u,
        TYPE_DATASET_TRAINING   = 1u,
        TYPE_DATASET_VALIDATION = 2u,
        TYPE_DATASET_LENGTH     = 3u
    };

    static std::map<enum ENUM_TYPE_DATASET, std::string> ENUM_TYPE_DATASET_NAMES = {
        {TYPE_DATASET_TESTING,    "Testing"},
        {TYPE_DATASET_TRAINING,   "Training"},
        {TYPE_DATASET_VALIDATION, "Validating"},
        {TYPE_DATASET_LENGTH,     "LENGTH"}
                                                                                    };

}