#pragma once

#include <string>
#include <map>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_ACTIVATION_FUNCTION : unsigned int
        {
            TYPE_NN_A_F_NONE = 0u,
            TYPE_NN_A_F_COSINE = 1u,
            TYPE_NN_A_F_COSINE_SYMMETRIC = 2u,
            TYPE_NN_A_F_ELU = 3u,
            TYPE_NN_A_F_ELLIOT = 4u,
            TYPE_NN_A_F_ELLIOT_SYMMETRIC = 5u,
            TYPE_NN_A_F_GAUSSIAN = 6u,
            TYPE_NN_A_F_GAUSSIAN_STEPWISE = 7u,
            TYPE_NN_A_F_GAUSSIAN_SYMMETRIC = 8u,
            TYPE_NN_A_F_ISRU = 9u,
            TYPE_NN_A_F_ISRLU = 10u,
            TYPE_NN_A_F_LINEAR = 11u,
            TYPE_NN_A_F_LINEAR_PIECE = 12u,
            TYPE_NN_A_F_LINEAR_PIECE_SYMMETRIC = 13u,
            TYPE_NN_A_F_LEAKY_RELU = 14u,
            TYPE_NN_A_F_PARAMETRIC_RELU = 15u,
            TYPE_NN_A_F_RELU = 16u,
            TYPE_NN_A_F_SELU = 17u,
            TYPE_NN_A_F_SIGMOID = 18u,
            TYPE_NN_A_F_SINE = 19u,
            TYPE_NN_A_F_SIGMOID_STEPWISE = 20u,
            TYPE_NN_A_F_SINE_SYMMETRIC = 21u,
            TYPE_NN_A_F_SOFTMAX = 22u,
            TYPE_NN_A_F_TANH = 23u,
            TYPE_NN_A_F_TANH_STEPWISE = 24u,
            TYPE_NN_A_F_THRESHOLD = 25u,
            TYPE_NN_A_F_THRESHOLD_SYMMETRIC = 26u,
            TYPE_NN_A_F_LENGTH = 27u
        };

        static std::map<enum ENUM_TYPE_ACTIVATION_FUNCTION, std::string> ENUM_TYPE_ACTIVATION_FUNCTION_NAME = {{TYPE_NN_A_F_NONE, "NONE"},
                                                                                                                                                                                                {TYPE_NN_A_F_COSINE, "Cosine"},
                                                                                                                                                                                                {TYPE_NN_A_F_COSINE_SYMMETRIC, "Cosine symmetric"},
                                                                                                                                                                                                {TYPE_NN_A_F_ELU, "Exponential Linear Unit"},
                                                                                                                                                                                                {TYPE_NN_A_F_ELLIOT, "Elliot"},
                                                                                                                                                                                                {TYPE_NN_A_F_ELLIOT_SYMMETRIC, "Elliot symmetric"},
                                                                                                                                                                                                {TYPE_NN_A_F_GAUSSIAN, "Gaussian"},
                                                                                                                                                                                                {TYPE_NN_A_F_GAUSSIAN_STEPWISE, "Gaussian stepwise"},
                                                                                                                                                                                                {TYPE_NN_A_F_GAUSSIAN_SYMMETRIC, "Gaussian symmetric"},
                                                                                                                                                                                                {TYPE_NN_A_F_ISRU, "Inverse Square Root Unit"},
                                                                                                                                                                                                {TYPE_NN_A_F_ISRLU, "Inverse Square Root Linear Unit"},
                                                                                                                                                                                                {TYPE_NN_A_F_LINEAR, "Linear"},
                                                                                                                                                                                                {TYPE_NN_A_F_LINEAR_PIECE, "Linear piece"},
                                                                                                                                                                                                {TYPE_NN_A_F_LINEAR_PIECE_SYMMETRIC, "Linear piece symmetric"},
                                                                                                                                                                                                {TYPE_NN_A_F_LEAKY_RELU, "Leaky Rectified Linear Units"},
                                                                                                                                                                                                {TYPE_NN_A_F_PARAMETRIC_RELU, "[x] Parametric Rectified Linear Units"},
                                                                                                                                                                                                {TYPE_NN_A_F_RELU, "Rectified Linear Units"},
                                                                                                                                                                                                {TYPE_NN_A_F_SELU, "Scaled exponential Linear Unit"},
                                                                                                                                                                                                {TYPE_NN_A_F_SIGMOID, "Sigmoid"},
                                                                                                                                                                                                {TYPE_NN_A_F_SINE, "Sine"},
                                                                                                                                                                                                {TYPE_NN_A_F_SIGMOID_STEPWISE, "Sigmoid stepwise"},
                                                                                                                                                                                                {TYPE_NN_A_F_SINE_SYMMETRIC, "Sine symmetric"},
                                                                                                                                                                                                {TYPE_NN_A_F_SOFTMAX, "Softmax"},
                                                                                                                                                                                                {TYPE_NN_A_F_TANH, "Tanh"},
                                                                                                                                                                                                {TYPE_NN_A_F_TANH_STEPWISE, "Tanh stepwise"},
                                                                                                                                                                                                {TYPE_NN_A_F_THRESHOLD, "Threshold"},
                                                                                                                                                                                                {TYPE_NN_A_F_THRESHOLD_SYMMETRIC, "Threshold symmetric"},
                                                                                                                                                                                                {TYPE_NN_A_F_LENGTH, "LENGTH"}};
    }
}
