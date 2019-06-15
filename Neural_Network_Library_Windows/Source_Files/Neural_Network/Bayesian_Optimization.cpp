#include "stdafx.hpp"

#if defined(COMPILE_CUDA)
    #include <CUDA/CUDA_Dataset_Manager.cuh>
#endif

#include <Neural_Network/Dataset_Manager.hpp>

#include <iostream>
#include <array>

template<typename T>
T constexpr sqrtNewtonRaphson(T x_received, T curr, T prev)
{
    return(curr == prev ?
                                curr
                                :
                                sqrtNewtonRaphson(x_received, T(0.5) * (curr + x_received / curr), curr));
}

template<typename T>
T constexpr constexpr_sqrt(T x_received)
{
    return(x_received >= T(0) && x_received < std::numeric_limits<T>::infinity() ?
                                                                                                                 sqrtNewtonRaphson(x_received, x_received, T(0))
                                                                                                                 :
                                                                                                                 std::numeric_limits<T>::quiet_NaN());
}

// Normal probability density function.
template<typename T>
T normal_pdf(T x_received)
{
    T constexpr inv_sqrt_2pi(T(1) / constexpr_sqrt(T(2) * MyEA::Math::PI<T>));

    return(inv_sqrt_2pi * std::exp(x_received * x_received * T(-0.5)));
}

// Normal cumulative distribution function.
template<typename T>
T normal_cdf(T const x_received) { return(erfc(-x_received / constexpr_sqrt(T(2))) * T(0.5)); }
