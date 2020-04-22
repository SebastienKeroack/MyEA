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
#include <Random/Floating.hpp>

namespace MyEA::Random
{
    template<typename T>
    Floating<T>::Floating(void) : Base()
    {
        this->_uniform_real_distribution.param(typename std::uniform_real_distribution<T>::param_type(this->_minimum_range, this->_maximum_range));
    }

    template<typename T>
    Floating<T>::Floating(T const minimum_range_received,
                          T const maximum_range_received,
                          unsigned int const seed_received) : Base(seed_received)
    {
        this->Range(minimum_range_received, maximum_range_received);
    }

    template<typename T>
    class Floating<T>& Floating<T>::operator=(class Floating<T> const &ref_source_Class_Generator_Random_Real_received)
    {
        if(&ref_source_Class_Generator_Random_Real_received != this) { this->Copy(ref_source_Class_Generator_Random_Real_received); }

        return(*this);
    }

    template<typename T>
    void Floating<T>::Copy(class Floating<T> const &ref_source_Class_Generator_Random_Real_received)
    {
        Base::Copy(ref_source_Class_Generator_Random_Real_received);

        this->_minimum_range = ref_source_Class_Generator_Random_Real_received._minimum_range;
        this->_maximum_range = ref_source_Class_Generator_Random_Real_received._maximum_range;

        this->_uniform_real_distribution = ref_source_Class_Generator_Random_Real_received._uniform_real_distribution;
    }

    template<typename T>
    void Floating<T>::Range(T const minimum_range_received, T const maximum_range_received)
    {
        BOOST_ASSERT_MSG(minimum_range_received < maximum_range_received, "`minimum_range_received` can not be less than `minimum_range_received`");

        if(this->_minimum_range == minimum_range_received && this->_maximum_range == maximum_range_received) { return; }

        this->_minimum_range = minimum_range_received;
        this->_maximum_range = maximum_range_received;

        this->_uniform_real_distribution.param(typename std::uniform_real_distribution<T>::param_type(minimum_range_received, maximum_range_received));
    }

    template<typename T>
    void Floating<T>::Clear(void)
    {
        Base::Clear();

        this->Range(T(0), T(1));
    }

    template<typename T>
    void Floating<T>::Reset(void)
    {
        Base::Reset();

        this->_uniform_real_distribution.reset();
    }

    template<typename T>
    T Floating<T>::operator()(void) { return(this->_uniform_real_distribution(this->p_generator_mt19937)); }

    // |STR| Template initialization declaration. |STR|
    template class Floating<float      >;
    template class Floating<double     >;
    template class Floating<long double>;
    // |END| Template initialization declaration. |END|
}
