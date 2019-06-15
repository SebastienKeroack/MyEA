#pragma once

#include <math.h>
#include <stdio.h>
#include <cstdlib>

#include <Neural_Network/Data.hpp>
#include <Neural_Network/Train.hpp>

T_ Activation_Real(class Neural_Network *ptr_Neural_Network_received,
                            unsigned int const activation_function_received,
                            T_ const steepness_received,
                            T_ value_received);

int fann_trainer_outputs(class Neural_Network *ptr_Neural_Network_received,
                                   class Dataset<T_> *ptr_CTrain_Data_received,
                                   float desired_loss_received);

float fann_trainer_outputs_epoch(class Neural_Network *ptr_Neural_Network_received, class Dataset<T_> *ptr_CTrain_Data_received);

int fann_trainer_candidates(class Neural_Network *ptr_Neural_Network_received, class Dataset<T_> *ptr_CTrain_Data_received);

T_ fann_trainer_candidates_epoch(class Neural_Network *ptr_Neural_Network_received, class Dataset<T_> *ptr_CTrain_Data_received);

void fann_install_candidate(class Neural_Network *ptr_Neural_Network_received);

int fann_initialize_candidates(class Neural_Network *ptr_Neural_Network_received);

#define fann_exp2(x) exp(0.69314718055994530942 * (x))

#define MATH_RAND(min_value_received, max_value_received) ((static_cast<float>(min_value_received)) + ((static_cast<float>(max_value_received) - (static_cast<float>(min_value_received))) * static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) + 1.0f)))

#define fann_abs(value) (((value) >= 0) ? (value) : -(value))

#ifdef USE_FIXED
    #define fann_mult(x, y) ((x * y) >> decimal_point)
    #define fann_div(x, y) (((x) << decimal_point) / y)
    #define fann_random_weight() (T_)(MATH_RAND(0, multiplier / 10))
    #define fann_random_bias_weight() (T_)(MATH_RAND((0 - multiplier) / 10, multiplier / 10))
#else
    #define fann_mult(x, y) (x * y)
    #define fann_div(x, y) (x / y)
    #define fann_random_weight() (MATH_RAND(-0.1f, 0.1f))
    #define fann_random_bias_weight() (MATH_RAND(-0.1f, 0.1f))
#endif