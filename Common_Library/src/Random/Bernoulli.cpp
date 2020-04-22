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
#include <Random/Bernoulli.hpp>

namespace MyEA::Random
{
    template<typename T>
    Bernoulli<T>::Bernoulli(void) : Base() { }

    template<typename T>
    Bernoulli<T>::Bernoulli(T const probability_received, unsigned int const seed_received) : Base(seed_received)
    {
        this->Probability(probability_received);
    }

    template<typename T>
    class Bernoulli<T>& Bernoulli<T>::operator=(class Bernoulli<T> const &ref_source_Class_Generator_Random_Bernoulli_received)
    {
        if(&ref_source_Class_Generator_Random_Bernoulli_received != this) { this->Copy(ref_source_Class_Generator_Random_Bernoulli_received); }

        return(*this);
    }

    template<typename T>
    bool Bernoulli<T>::operator()(void) { return(this->_bernoulli_distribution(this->p_generator_mt19937)); }

    template<typename T>
    void Bernoulli<T>::Copy(class Bernoulli<T> const &ref_source_Class_Generator_Random_Bernoulli_received)
    {
        Base::Copy(ref_source_Class_Generator_Random_Bernoulli_received);

        this->_probability = ref_source_Class_Generator_Random_Bernoulli_received._probability;

        this->_bernoulli_distribution = ref_source_Class_Generator_Random_Bernoulli_received._bernoulli_distribution;
    }

    template<typename T>
    void Bernoulli<T>::Probability(T const probability_received)
    {
        BOOST_ASSERT_MSG(probability_received >= T(0) && probability_received < T(1), "`probability_received` need to be in the range of [0, 1).");

        if(this->_probability == probability_received) { return; }

        this->_probability = probability_received;

        std::bernoulli_distribution::param_type tmp_bernoulli_distribution_param_type(probability_received);
        this->_bernoulli_distribution.param(tmp_bernoulli_distribution_param_type);
    }

    template<typename T>
    void Bernoulli<T>::Clear(void)
    {
        Base::Clear();

        this->_probability = T(0);
    }

    template<typename T>
    void Bernoulli<T>::Reset(void)
    {
        Base::Reset();

        this->_bernoulli_distribution.reset();
    }

    // |STR| Template initialization declaration. |STR|
    template class Bernoulli<float      >;
    template class Bernoulli<double     >;
    template class Bernoulli<long double>;
    // |END| Template initialization declaration. |END|
}
