#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_DATASET_MANAGER_STORAGE : unsigned int
        {
            TYPE_STORAGE_NONE = 0u,
            TYPE_STORAGE_TRAINING = 1u,
            TYPE_STORAGE_TRAINING_TESTING = 2u,
            TYPE_STORAGE_TRAINING_VALIDATION_TESTING = 3u,
            TYPE_STORAGE_LENGTH = 4u
        };

        static std::map<enum ENUM_TYPE_DATASET_MANAGER_STORAGE, std::string> ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES = {{TYPE_STORAGE_NONE, "NONE"},
                                                                                                                                                                                                                             {TYPE_STORAGE_TRAINING, "Training"},
                                                                                                                                                                                                                             {TYPE_STORAGE_TRAINING_TESTING, "Training - Testing"},
                                                                                                                                                                                                                             {TYPE_STORAGE_TRAINING_VALIDATION_TESTING, "Training - Validating - Testing"},
                                                                                                                                                                                                                             {TYPE_STORAGE_LENGTH, "LENGTH"}};

    }
}