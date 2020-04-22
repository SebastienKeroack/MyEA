#pragma once

#include <string>
#include <map>

namespace MyEA::Common
{
    enum ENUM_TYPE_DATASET : unsigned int
    {
        TYPE_DATASET_NONE       = 0u,
        TYPE_DATASET_TESTING    = 1u,
        TYPE_DATASET_TRAINING   = 2u,
        TYPE_DATASET_VALIDATION = 3u,
        TYPE_DATASET_LENGTH     = 4u
    };

    static std::map<enum ENUM_TYPE_DATASET, std::string> ENUM_TYPE_DATASET_NAMES = {
        {TYPE_DATASET_NONE,       "None"},
        {TYPE_DATASET_TESTING,    "Testing"},
        {TYPE_DATASET_TRAINING,   "Training"},
        {TYPE_DATASET_VALIDATION, "Validating"},
        {TYPE_DATASET_LENGTH,     "LENGTH"}
    };

}