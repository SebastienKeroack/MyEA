#pragma once

#include <Enums/Enum_Type_Activation_Functions.hpp>
#include <Configuration/Configuration.hpp>

/* stepwise linear functions used for some of the activation functions */
/* defines used for the stepwise linear functions */
#define AF_LINEAR_real(v1, r1, v2, r2, summation_received) (((((r2) - (r1)) * ((summation_received) - (v1))) / ((v2) - (v1))) + (r1))

#if defined(COMPILE_AUTODIFF)
    #define AF_STEPWISE_real(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6, min_received, max_received, summation_received) (summation_received)
#else
    #define AF_STEPWISE_real(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6, min_received, max_received, summation_received) \
                                             (summation_received < v5 ? (summation_received < v3 ? (summation_received < v2 ? (summation_received < v1 ? min_received : AF_LINEAR_real(v1, r1, v2, r2, summation_received)) : AF_LINEAR_real(v2, r2, v3, r3, summation_received)) : (summation_received < v4 ? AF_LINEAR_real(v3, r3, v4, r4, summation_received) : AF_LINEAR_real(v4, r4, v5, r5, summation_received))) \
                                                                                      : \
                                                                                      (summation_received < v6 ? AF_LINEAR_real(v5, r5, v6, r6, summation_received) : max_received))
#endif

// Linear.
//#define MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR(steepness_received, summation_received) fann_mult(steepness_received, summation_received)
#define AF_LINEAR_derive(steepness_received, value_received) (steepness_received)

// Sigmoid.
//#define MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID(steepness_received, summation_received) (1_T/(1_T + exp(-2_T * steepness_received * summation_received)))
#define AF_SIGMOID_real(summation_received) (1_T / (1_T + exp(-summation_received)))
#define AF_SIGMOID_derive(steepness_received, value_received) (steepness_received * value_received * (1_T - value_received))

// Sigmoid symmetric.
//#define MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_TANH(steepness_received, summation_received) (2_T/(1_T + exp(-2_T * steepness_received * summation_received)) - 1_T)
#define AF_TANH_real(summation_received) (2_T / (1_T + exp(-2_T * summation_received)) - 1_T)
#define AF_TANH_derive(steepness_received, value_received) (steepness_received * (1_T - value_received * value_received))

// Gaussian.
//#define MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_GAUSSIAN(steepness_received, summation_received) (exp(-summation_received * steepness_received * summation_received * steepness_received))
#define AF_GAUSSIAN_real(summation_received) (exp(-summation_received * summation_received))
#define AF_GAUSSIAN_derive(steepness_received, value_received, summation_received) (-2_T * summation_received * value_received * steepness_received * steepness_received)

// Gaussian symmetric.
//#define MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_GAUSSIAN_SYMMETRIC(steepness_received, summation_received) ((exp(-summation_received * steepness_received * summation_received * steepness_received)*2.0)-1.0)
#define AF_GAUSSIAN_SYMMETRIC_real(summation_received) ((exp(-summation_received * summation_received) * 2_T) - 1_T)
#define AF_GAUSSIAN_SYMMETRIC_derive(steepness_received, value_received, summation_received) (-2_T * summation_received * (value_received + 1_T) * steepness_received * steepness_received)

// Elliot.
//#define MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELLIOT(steepness_received, summation_received) (((summation_received * steepness_received) / 2_T) / (1_T + MyEA::Math::Absolute(summation_received * steepness_received)) + 0.5_T)
#define AF_ELLIOT_real(summation_received) (((summation_received) / 2_T) / (1_T + MyEA::Math::Absolute<T_>(summation_received)) + 0.5_T)
#define AF_ELLIOT_derive(steepness_received, summation_received) (steepness_received * 1_T / (2_T * (1_T + MyEA::Math::Absolute<T_>(summation_received)) * (1_T + MyEA::Math::Absolute<T_>(summation_received))))

// Elliot symmetric.
//#define MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELLIOT_SYMMETRIC(steepness_received, summation_received) ((summation_received * steepness_received) / (1_T + MyEA::Math::Absolute(summation_received * steepness_received)))
#define AF_ELLIOT_SYMMETRIC_real(summation_received) ((summation_received) / (1_T + MyEA::Math::Absolute<T_>(summation_received)))
#define AF_ELLIOT_SYMMETRIC_derive(steepness_received, summation_received) (steepness_received * 1_T / ((1_T + MyEA::Math::Absolute<T_>(summation_received)) * (1_T + MyEA::Math::Absolute<T_>(summation_received))))

// Sine.
#define AF_SIN_real(summation_received) (sin(summation_received) / 2_T + 0.5_T)
#define AF_SIN_derive(steepness_received, summation_received) (steepness_received * cos(steepness_received * summation_received) / 2_T)

// Sine symmetric.
#define AF_SIN_SYMMETRIC_real(summation_received) (sin(summation_received))
#define AF_SIN_SYMMETRIC_derive(steepness_received, summation_received) (steepness_received * cos(steepness_received * summation_received))

// Cosine.
#define AF_COS_real(summation_received) (cos(summation_received) / 2_T + 0.5_T)
#define AF_COS_derive(steepness_received, summation_received) (steepness_received * -sin(steepness_received * summation_received) / 2_T)

// Cosine symmetric.
#define AF_COS_SYMMETRIC_real(summation_received) (cos(summation_received))
#define AF_COS_SYMMETRIC_derive(steepness_received, summation_received) (steepness_received * -sin(steepness_received * summation_received))

//  Inverse square root unit.
#define AF_ISRU_real(summation_received, slope_received) (summation_received * (1_T / sqrt(1_T + slope_received * summation_received * summation_received)))
#define AF_ISRU_derive(steepness_received, summation_received, value_received, slope_received) (steepness_received * static_cast<T_>(pow(value_received / summation_received, 3u)))

//  Inverse square root linear unit.
#define AF_ISRLU_real(summation_received, slope_received) (summation_received < 0_T ? summation_received * (1_T / sqrt(1_T + slope_received * summation_received * summation_received)) : summation_received)
#define AF_ISRLU_derive(steepness_received, summation_received, value_received, slope_received) (summation_received < 0_T ? steepness_received * static_cast<T_>(pow(value_received / summation_received, 3u)) : steepness_received)

// Exponential linear unit.
#define AF_ELU_real(summation_received, slope_received) (summation_received < 0_T ? slope_received * (exp(summation_received) - 1_T) : summation_received)
#define AF_ELU_derive(steepness_received, summation_received, value_received, slope_received) (summation_received < 0_T ? steepness_received * slope_received * value_received : steepness_received)

// Scaled exponential linear unit.
#define SELU_Alpha 1.6732632423543772848170429916717_T
#define SELU_Scale 1.0507009873554804934193349852946_T
#define AF_SELU_real(summation_received) (SELU_Scale * (summation_received <= 0_T ? SELU_Alpha * exp(summation_received) - SELU_Alpha : summation_received))
#define AF_SELU_derive(steepness_received, summation_received, value_received) (SELU_Scale * (summation_received <= 0_T ? steepness_received * (value_received + SELU_Alpha) : steepness_received))

// Rectifier linear unit.
#define AF_RELU_real(summation_received) (summation_received < 0_T ? 0_T : summation_received)
#define AF_RELU_derive(steepness_received, summation_received) (summation_received < 0_T ? 0_T : steepness_received)

// Leaky rectifier linear unit.
#define AF_LRELU_real(summation_received, slope_received) (summation_received < 0_T ? slope_received * summation_received : summation_received)
#define AF_LRELU_derive(steepness_received, summation_received, slope_received) (summation_received < 0_T ? steepness_received * slope_received : steepness_received)
#define AF_LRELU_ALPHA 0.01_T

// Parametric rectifier linear unit.
#define AF_PRELU_real(summation_received, slope_received) (summation_received < 0_T ? slope_received * summation_received : summation_received)
#define AF_PRELU_derive(steepness_received, summation_received, slope_received) (summation_received < 0_T ? steepness_received * slope_received : steepness_received)
#define AF_PRELU_ALPHA 0.01_T

// Softmax.
#define AF_SOFTMAX_real(summation_received) (exp(summation_received))
// w.r.t Cross-entropy.
#define AF_SOFTMAX_CE_derive(steepness_received) (steepness_received)
// w.r.t Loss.
#define AF_SOFTMAX_ii_derive(steepness_received, value_received) (steepness_received * value_received * (1_T - value_received))
#define AF_SOFTMAX_ik_derive(steepness_received, value_received, value_k_received) (steepness_received * -value_received * value_k_received)

#define AF_FIRE(activation_function_received, \
                         summation_received, \
                         activation_received) \
    switch(activation_function_received) \
    { \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR: activation_received = summation_received; break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR_PIECE: activation_received = summation_received < 0_T ? 0_T : (summation_received > 1_T ? 1_T : summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LINEAR_PIECE_SYMMETRIC: activation_received = summation_received < -1_T ? -1_T : (summation_received > 1_T ? 1_T : summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID: activation_received = AF_SIGMOID_real(summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_TANH: activation_received = AF_TANH_real(summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID_STEPWISE: activation_received = AF_STEPWISE_real(-2.64665246009826660156_T, \
                                                                                                                                                                                                                              -1.47221946716308593750_T, \
                                                                                                                                                                                                                              -5.49306154251098632812e-01_T, \
                                                                                                                                                                                                                              5.49306154251098632812e-01_T, \
                                                                                                                                                                                                                              1.47221934795379638672_T, \
                                                                                                                                                                                                                              2.64665293693542480469_T, \
                                                                                                                                                                                                                              4.99999988824129104614e-03_T, \
                                                                                                                                                                                                                              5.00000007450580596924e-02_T, \
                                                                                                                                                                                                                              2.50000000000000000000e-01_T, \
                                                                                                                                                                                                                              7.50000000000000000000e-01_T, \
                                                                                                                                                                                                                              9.49999988079071044922e-01_T, \
                                                                                                                                                                                                                              9.95000004768371582031e-01_T, \
                                                                                                                                                                                                                              0_T, \
                                                                                                                                                                                                                              1_T, \
                                                                                                                                                                                                                              summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_TANH_STEPWISE: activation_received = AF_STEPWISE_real(-2.64665293693542480469_T, \
                                                                                                                                                                                                                         -1.47221934795379638672_T, \
                                                                                                                                                                                                                         -5.49306154251098632812e-01_T, \
                                                                                                                                                                                                                         5.49306154251098632812e-01_T, \
                                                                                                                                                                                                                         1.47221934795379638672_T, \
                                                                                                                                                                                                                         2.64665293693542480469_T, \
                                                                                                                                                                                                                         -9.90000009536743164062e-01_T, \
                                                                                                                                                                                                                         -8.99999976158142089844e-01_T, \
                                                                                                                                                                                                                         -5.00000000000000000000e-01_T, \
                                                                                                                                                                                                                         5.00000000000000000000e-01_T, \
                                                                                                                                                                                                                         8.99999976158142089844e-01_T, \
                                                                                                                                                                                                                         9.90000009536743164062e-01_T, \
                                                                                                                                                                                                                         -1_T, \
                                                                                                                                                                                                                         1_T, \
                                                                                                                                                                                                                         summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_THRESHOLD: activation_received = summation_received < 0_T ? 0_T : 1_T; break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_THRESHOLD_SYMMETRIC: activation_received = summation_received < 0_T ? -1_T : 1_T; break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_GAUSSIAN: activation_received = AF_GAUSSIAN_real(summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_GAUSSIAN_SYMMETRIC: activation_received = AF_GAUSSIAN_SYMMETRIC_real(summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_GAUSSIAN_STEPWISE: activation_received = 0; break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELLIOT: activation_received = AF_ELLIOT_real(summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELLIOT_SYMMETRIC: activation_received = AF_ELLIOT_SYMMETRIC_real(summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SINE: activation_received = AF_SIN_real(summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SINE_SYMMETRIC: activation_received = AF_SIN_SYMMETRIC_real(summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_COSINE: activation_received = AF_COS_real(summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_COSINE_SYMMETRIC: activation_received = AF_COS_SYMMETRIC_real(summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ISRU: activation_received = AF_ISRU_real(summation_received, 1_T); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ISRLU: activation_received = AF_ISRLU_real(summation_received, 1_T); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_ELU: activation_received = AF_ELU_real(summation_received, 1_T); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SELU: activation_received = AF_SELU_real(summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_RELU: activation_received = AF_RELU_real(summation_received); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_LEAKY_RELU: activation_received = AF_LRELU_real(summation_received, AF_LRELU_ALPHA); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_PARAMETRIC_RELU: activation_received = AF_PRELU_real(summation_received, AF_PRELU_ALPHA); break; \
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SOFTMAX: activation_received = AF_SOFTMAX_real(summation_received); break; \
        default: break; \
    }
