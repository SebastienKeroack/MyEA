/* Copyright 2020 Sébastien Kéroack. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "pch.hpp"

// This.
#include <Random/Gaussian.hpp>

namespace MyEA::Random
{
    template<typename T>
    Gaussian<T>::Gaussian(void) : Base()
    {
        this->_normal_distribution.param(typename std::normal_distribution<T>::param_type(this->_mean, this->_std));
    }

    template<typename T>
    Gaussian<T>::Gaussian(T const mean_received,
                          T const std_received,
                          unsigned int const seed_received) : Base(seed_received)
    {
        this->Range(mean_received, std_received);
    }

    template<typename T>
    class Gaussian<T>& Gaussian<T>::operator=(class Gaussian<T> const &ref_source_Class_Generator_Random_Gaussian_received)
    {
        if(&ref_source_Class_Generator_Random_Gaussian_received != this) { this->Copy(ref_source_Class_Generator_Random_Gaussian_received); }

        return(*this);
    }

    template<typename T>
    void Gaussian<T>::Copy(class Gaussian<T> const &ref_source_Class_Generator_Random_Gaussian_received)
    {
        Base::Copy(ref_source_Class_Generator_Random_Gaussian_received);

        this->_mean = ref_source_Class_Generator_Random_Gaussian_received._mean;
        this->_std  = ref_source_Class_Generator_Random_Gaussian_received._std;

        this->_normal_distribution = ref_source_Class_Generator_Random_Gaussian_received._normal_distribution;
    }

    template<typename T>
    void Gaussian<T>::Range(T const mean_received, T const std_received)
    {
        if(this->_mean == mean_received && this->_std == std_received) { return; }

        this->_mean = mean_received;
        this->_std  = std_received;

        this->_normal_distribution.param(typename std::normal_distribution<T>::param_type(mean_received, std_received));
    }

    template<typename T>
    void Gaussian<T>::Clear(void)
    {
        Base::Clear();

        this->Range(T(0), T(1));
    }

    template<typename T>
    void Gaussian<T>::Reset(void)
    {
        Base::Reset();

        this->_normal_distribution.reset();
    }

    template<typename T>
    T Gaussian<T>::operator()(void) { return(this->_normal_distribution(this->p_generator_mt19937)); }

    // |STR| Template initialization declaration. |STR|
    template class Gaussian<float      >;
    template class Gaussian<double     >;
    template class Gaussian<long double>;
    // |END| Template initialization declaration. |END|
}
