#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_DATASET_PROCESS : unsigned int
        {
            TYPE_DATASET_PROCESS_NONE = 0u,
            TYPE_DATASET_PROCESS_BATCH = 1u,
            TYPE_DATASET_PROCESS_MINI_BATCH = 2u,
            TYPE_DATASET_PROCESS_CROSS_VALIDATION = 3u,
            TYPE_DATASET_PROCESS_CROSS_VALIDATION_RANDOM_SEARCH = 4u,
            TYPE_DATASET_PROCESS_LENGTH = 5u
        };

        static std::map<enum ENUM_TYPE_DATASET_PROCESS, std::string> ENUM_TYPE_DATASET_PROCESS_NAMES = {{TYPE_DATASET_PROCESS_NONE, "NONE"},
                                                                                                                                                                                         {TYPE_DATASET_PROCESS_BATCH, "Batch"},
                                                                                                                                                                                         {TYPE_DATASET_PROCESS_MINI_BATCH, "Mini batch"},
                                                                                                                                                                                         {TYPE_DATASET_PROCESS_CROSS_VALIDATION, "Cross validation"},
                                                                                                                                                                                         {TYPE_DATASET_PROCESS_CROSS_VALIDATION_RANDOM_SEARCH, "Cross validation, random search"},
                                                                                                                                                                                         {TYPE_DATASET_PROCESS_LENGTH, "LENGTH"}};

    }
}