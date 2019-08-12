#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_DATASET_FILE : unsigned int
        {
            TYPE_DATASET_FILE_DATASET = 0u,
            TYPE_DATASET_FILE_DATASET_SPLIT = 1u,
            TYPE_DATASET_FILE_MNIST = 2u, // National Institute of Standards and Technology (NIST)
            TYPE_DATASET_FILE_LENGTH = 3u
        };

        static std::map<enum ENUM_TYPE_DATASET_FILE, std::string> ENUM_TYPE_DATASET_FILE_NAMES = {{TYPE_DATASET_FILE_DATASET, "Dataset"},
                                                                                                                                                                       {TYPE_DATASET_FILE_DATASET_SPLIT, "Dataset split"},
                                                                                                                                                                       {TYPE_DATASET_FILE_MNIST, "MNIST"},
                                                                                                                                                                       {TYPE_DATASET_FILE_LENGTH, "LENGTH"}};
    }
}
