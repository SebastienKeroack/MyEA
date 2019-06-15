#pragma once

#include <Math/Mathematic.hpp>

// Linear
//#define Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_LINEAR(steepness_received, sum_received) fann_mult(steepness_received, sum_received)
template<typename T>
T _Activation_Function_LINEAR_extra_real_t_(T value_0_received,
                                                                            T r_0,
                                                                            T value_1_received,
                                                                            T r_1,
                                                                            T value_received) // ptr_array_value_received[0] = Sum
{ return((((r_1 - r_0) * (value_received - value_0_received)) / (value_1_received - value_0_received)) + r_0); }

template<typename T>
T _Activation_Function_LINEAR_real_t_(T *ptr_array_value_received) { return(ptr_array_value_received[0u]); } // ptr_array_value_received[0] = Steepness

template<typename T>
T _Activation_Function_LINEAR_derive_t_(T *ptr_array_value_received) { return(ptr_array_value_received[0u]); } // ptr_array_value_received[0] = Steepness

// Linear piece
template<typename T>
T _Activation_Function_LINEAR_PIECE_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum
{ return(ptr_array_value_received[0u] < T(0) ? T(0) : (ptr_array_value_received[0u] > T(1)) ? T(1) : ptr_array_value_received[0u]); }

template<typename T>
T _Activation_Function_LINEAR_PIECE_derive_t_(T *ptr_array_value_received) { return(ptr_array_value_received[0u]); } // ptr_array_value_received[0] = Steepness

// Linear piece symmetric
template<typename T>
T _Activation_Function_LINEAR_PIECE_SYMMETRIC_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum
{ return(ptr_array_value_received[0u] < T(-1) ? T(-1) : (ptr_array_value_received[0u] > T(1)) ? T(1) : ptr_array_value_received[0u]); }

template<typename T>
T _Activation_Function_LINEAR_PIECE_SYMMETRIC_derive_t_(T *ptr_array_value_received) { return(ptr_array_value_received[0u]); } // ptr_array_value_received[0] = Steepness

// Stepwise
template<typename T>
T _Activation_Function_STEPWISE_real_t_(T value_0_received,
                                                                        T value_1_received,
                                                                        T value_2_received,
                                                                        T value_3_received,
                                                                        T value_4_received,
                                                                        T value_5_received,
                                                                        T r_0_received,
                                                                        T r_1_received,
                                                                        T r_2_received,
                                                                        T r_3_received,
                                                                        T r_4_received,
                                                                        T r_5_received,
                                                                        T min_received,
                                                                        T max_received,
                                                                        T value_received) // ptr_array_value_received[0] = Sum
{ return(value_received < value_4_received ? (value_received < value_2_received ? (value_received < value_1_received ? (value_received < value_0_received ? min_received : _Activation_Function_LINEAR_extra_real_t_(value_0_received, r_0_received, value_1_received, r_1_received, value_received)) : _Activation_Function_LINEAR_extra_real_t_(value_1_received, r_1_received, value_2_received, r_2_received, value_received)) : (value_received < value_3_received ? _Activation_Function_LINEAR_extra_real_t_(value_2_received, r_2_received, value_3_received, r_3_received, value_received) : _Activation_Function_LINEAR_extra_real_t_(value_3_received, r_3_received, value_4_received, r_4_received, value_received))) : (value_received < value_5_received ? _Activation_Function_LINEAR_extra_real_t_(value_4_received, r_4_received, value_5_received, r_5_received, value_received) : max_received)); }

// Sigmoid
//#define Common::ENUM_TYPE_NN__Activation_FunctionS::TYPE_NN_A_F_SIGMOID(steepness_received, sum_received) (T(1)/(T(1) + exp(-T(2) * steepness_received * sum_received)))
template<typename T>
T _Activation_Function_SIGMOID_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum
{ return(T(1) / (T(1) + exp(T(-2) * ptr_array_value_received[0u]))); }

template<typename T>
T _Activation_Function_SIGMOID_derive_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Value
{
    T tmp_value_clip(MyEA::Math::Clip<T>(ptr_array_value_received[1u], T(0.01), T(0.99)));
    return(T(2) * ptr_array_value_received[0u] * tmp_value_clip * (T(1) - tmp_value_clip));
}

// Sigmoid symmetric
//#define Common::ENUM_TYPE_NN__Activation_FunctionS::TYPE_NN_A_F_TANH(steepness_received, sum_received) (T(2)/(T(1) + exp(-T(2) * steepness_received * sum_received)) - T(1))
template<typename T>
T _Activation_Function_TANH_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum
{ return(T(2) / (T(1) + exp(T(-2) * ptr_array_value_received[0u])) - T(1)); }

template<typename T>
T _Activation_Function_TANH_derive_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Value
{
    T tmp_value_clip(MyEA::Math::Clip<T>(ptr_array_value_received[1u], T(-0.98), T(0.98)));
    return(ptr_array_value_received[0u] * (T(1) - (tmp_value_clip * tmp_value_clip)));
}

// Sigmoid stepwise
template<typename T>
T _Activation_Function_SIGMOID_STEPWISE_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum
{ return(_Activation_Function_STEPWISE_real_t_(T(-2.64665246009826660156e+00),
                                                                                 T(-1.47221946716308593750e+00),
                                                                                 T(-5.49306154251098632812e-01),
                                                                                 T(5.49306154251098632812e-01),
                                                                                 T(1.47221934795379638672e+00),
                                                                                 T(2.64665293693542480469e+00),
                                                                                 T(4.99999988824129104614e-03),
                                                                                 T(5.00000007450580596924e-02),
                                                                                 T(2.50000000000000000000e-01),
                                                                                 T(7.50000000000000000000e-01),
                                                                                 T(9.49999988079071044922e-01),
                                                                                 T(9.95000004768371582031e-01),
                                                                                 T(0),
                                                                                 T(1),
                                                                                 ptr_array_value_received[0u])); }
template<typename T>
T _Activation_Function_SIGMOID_STEPWISE_derive_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Value
{
    T tmp_value_clip(MyEA::Math::Clip<T>(ptr_array_value_received[1u], 0.01f, 0.99f));
    return(T(2) * ptr_array_value_received[0u] * tmp_value_clip * (T(1) - tmp_value_clip));
}

// Sigmoid stepwise symmetric
template<typename T>
T _Activation_Function_TANH_STEPWISE_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum
{ return(_Activation_Function_STEPWISE_real_t_(T(-2.64665293693542480469e+00),
                                                                             T(-1.47221934795379638672e+00),
                                                                             T(-5.49306154251098632812e-01),
                                                                             T(5.49306154251098632812e-01),
                                                                             T(1.47221934795379638672e+00),
                                                                             T(2.64665293693542480469e+00),
                                                                             T(-9.90000009536743164062e-01),
                                                                             T(-8.99999976158142089844e-01),
                                                                             T(-5.00000000000000000000e-01),
                                                                             T(5.00000000000000000000e-01),
                                                                             T(8.99999976158142089844e-01),
                                                                             T(9.90000009536743164062e-01),
                                                                             T(-1),
                                                                             T(1),
                                                                             ptr_array_value_received[0u])); }
template<typename T>
T _Activation_Function_TANH_STEPWISE_derive_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Value
{
    T tmp_value_clip(MyEA::Math::Clip<T>(ptr_array_value_received[1u], T(-0.98), T(0.98)));
    return(ptr_array_value_received[0u] * (T(1) - (tmp_value_clip * tmp_value_clip)));
}

// Threshold
template<typename T>
T _Activation_Function_THRESHOLD_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum
{ return(ptr_array_value_received[0u] < T(0) ? T(0) : T(1)); }

// Threshold symmetric
template<typename T>
T _Activation_Function_THRESHOLD_SYMMETRIC_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum
{ return(ptr_array_value_received[0u] < T(0) ? T(-1) : T(1)); }

// Gaussian
template<typename T>
T _Activation_Function_GAUSSIAN_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum
{ return(exp(-ptr_array_value_received[0u] * ptr_array_value_received[0u])); }

template<typename T>
T _Activation_Function_GAUSSIAN_derive_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Value, ptr_array_value_received[2] = Sum
{ return(T(-2) * ptr_array_value_received[2u] * ptr_array_value_received[1u] * ptr_array_value_received[0u] * ptr_array_value_received[0u]); }

// Gaussian symmetric
template<typename T>
T _Activation_Function_GAUSSIAN_SYMMETRIC_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum
{ return((exp(-ptr_array_value_received[0u] * ptr_array_value_received[0u]) * T(2)) - T(1)); }

template<typename T>
T _Activation_Function_GAUSSIAN_SYMMETRIC_derive_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Value, ptr_array_value_received[2] = Sum
{ return(T(-2) * ptr_array_value_received[2u] * (ptr_array_value_received[1u] + T(1)) * ptr_array_value_received[0u] * ptr_array_value_received[0u]); }

// Elliot
template<typename T>
T _Activation_Function_ELLIOT_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum
{ return((ptr_array_value_received[0u] / T(2)) / (T(1) + MyEA::Math::Absolute(ptr_array_value_received[0u])) + T(0.5)); }

template<typename T>
T _Activation_Function_ELLIOT_derive_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Sum
{ return(ptr_array_value_received[0u] * T(1) / (T(2) * (T(1) + MyEA::Math::Absolute(ptr_array_value_received[1u])) * (T(1) + MyEA::Math::Absolute(ptr_array_value_received[1u])))); }

// Elliot symmetric
template<typename T>
T _Activation_Function_ELLIOT_SYMMETRIC_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum
{ return((ptr_array_value_received[0u]) / (T(1) + MyEA::Math::Absolute(ptr_array_value_received[0u]))); }

template<typename T>
T _Activation_Function_ELLIOT_SYMMETRIC_derive_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Sum
{ return(ptr_array_value_received[0u] * T(1) / ((T(1) + MyEA::Math::Absolute(ptr_array_value_received[1u])) * (T(1) + MyEA::Math::Absolute(ptr_array_value_received[0u])))); }

// Sin
template<typename T>
T _Activation_Function_SIN_real_t_(T *ptr_array_value_received)
{ return(sin(ptr_array_value_received[0u]) / T(2) + T(0.5)); } // ptr_array_value_received[0] = Sum

template<typename T>
T _Activation_Function_SIN_derive_t_(T *ptr_array_value_received)
{ return(ptr_array_value_received[0u] * cos(ptr_array_value_received[0u] * ptr_array_value_received[1u]) / T(2)); } // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Sum

// Sin symmetric
template<typename T>
T _Activation_Function_SIN_SYMMETRIC_real_t_(T *ptr_array_value_received)
{ return(sin(ptr_array_value_received[0u])); } // ptr_array_value_received[0] = Sum

template<typename T>
T _Activation_Function_SIN_SYMMETRIC_derive_t_(T *ptr_array_value_received)
{ return(ptr_array_value_received[0u] * cos(ptr_array_value_received[0u] * ptr_array_value_received[1u])); } // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Sum

// Cos
template<typename T>
T _Activation_Function_COS_real_t_(T *ptr_array_value_received)
{ return(cos(ptr_array_value_received[0u]) / T(2) + T(0.5)); } // ptr_array_value_received[0] = Sum

template<typename T>
T _Activation_Function_COS_derive_t_(T *ptr_array_value_received)
{ return(ptr_array_value_received[0u] * -sin(ptr_array_value_received[0u] * ptr_array_value_received[1u]) / T(2)); } // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Sum

// Cos symmetric
template<typename T>
T _Activation_Function_COS_SYMMETRIC_real_t_(T *ptr_array_value_received)
{ return(cos(ptr_array_value_received[0u])); } // ptr_array_value_received[0] = Sum

template<typename T>
T _Activation_Function_COS_SYMMETRIC_derive_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Sum
{ return(ptr_array_value_received[0u] * -sin(ptr_array_value_received[0u] * ptr_array_value_received[1u])); }

// Rectifier linear unit
template<typename T>
T _Activation_Function_RELU_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum, ptr_array_value_received[1] = Slope
{ return(ptr_array_value_received[0u] < T(0) ? T(0) : ptr_array_value_received[0u]); }

template<typename T>
T _Activation_Function_RELU_derive_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Sum
{ return(ptr_array_value_received[1u] < T(0) ? T(0) : ptr_array_value_received[0u]); }

// Leaky rectifier linear unit, slope = 0.01
template<typename T>
T _Activation_Function_LRELU_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum, ptr_array_value_received[1] = Slope
{ return(ptr_array_value_received[0u] < T(0) ? T(0.01) * ptr_array_value_received[0u] : ptr_array_value_received[0u]); }

template<typename T>
T _Activation_Function_LRELU_derive_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Sum, ptr_array_value_received[2] = Slope
{ return(ptr_array_value_received[1u] < T(0) ? T(0.01) * ptr_array_value_received[0u] : ptr_array_value_received[0u]); }

// Parametric rectifier linear unit
template<typename T>
T _Activation_Function_PRELU_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum, ptr_array_value_received[1] = Slope
{ return(ptr_array_value_received[0u] < T(0) ? T(0.01) * ptr_array_value_received[0u] : ptr_array_value_received[0u]); }

template<typename T>
T _Activation_Function_PRELU_derive_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Sum, ptr_array_value_received[2] = Slope
{ return(ptr_array_value_received[1u] < T(0) ? T(0.01) * ptr_array_value_received[0u] : ptr_array_value_received[0u]); }

// Softmax
template<typename T>
T _Activation_Function_SOFTMAX_real_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Sum
 { return(exp(ptr_array_value_received[0u])); }

// W.R.T Cross-entropy
template<typename T>
T _Activation_Function_SOFTMAX_CE_derive_t_(T *ptr_array_value_received) { return(ptr_array_value_received[0u]); } // ptr_array_value_received[0] = Steepness

// W.R.T MSE
template<typename T>
T _Activation_Function_SOFTMAX_MSE_derive_t_(T *ptr_array_value_received) // ptr_array_value_received[0] = Steepness, ptr_array_value_received[1] = Value
{ return(ptr_array_value_received[0u] * ptr_array_value_received[1u] * (T(1) - ptr_array_value_received[1u])); } // steepness * value * (1 - value)