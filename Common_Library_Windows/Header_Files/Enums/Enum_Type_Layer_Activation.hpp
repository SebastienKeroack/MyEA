#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_LAYER_ACTIVATION : unsigned int
        {
            TYPE_ACTIVATION_NONE = 0u,
            TYPE_ACTIVATION_ASYMMETRIC = 1u,
            TYPE_ACTIVATION_RECTIFIER = 2u,
            TYPE_ACTIVATION_SELF_NORMALIZATION = 3u,
            TYPE_ACTIVATION_SOFTMAX = 4u,
            TYPE_ACTIVATION_SYMMETRIC = 5u,
            TYPE_ACTIVATION_LENGTH = 6u
        };

        static std::map<enum ENUM_TYPE_LAYER_ACTIVATION, std::string> ENUM_TYPE_LAYER_ACTIVATION_NAME = {{TYPE_ACTIVATION_NONE, "NONE"},
                                                                                                                                                                                     {TYPE_ACTIVATION_ASYMMETRIC, "Asymmetric"},
                                                                                                                                                                                     {TYPE_ACTIVATION_RECTIFIER, "Rectifier"},
                                                                                                                                                                                     {TYPE_ACTIVATION_SELF_NORMALIZATION, "Self-normalization"},
                                                                                                                                                                                     {TYPE_ACTIVATION_SOFTMAX, "Softmax"},
                                                                                                                                                                                     {TYPE_ACTIVATION_SYMMETRIC, "Symmetric"},
                                                                                                                                                                                     {TYPE_ACTIVATION_LENGTH, "LENGTH"}};
    }
}