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

#pragma once

// This.
#include <Random/Base.hpp>

namespace MyEA::Random
{
    template<typename T>
    class Gaussian : public Base
    {
        public:
            Gaussian(void);

            Gaussian(T const mean_received,
                     T const std_received,
                     unsigned int const seed_received = _SEED());

            class Gaussian<T>& operator=(class Gaussian<T> const &ref_source_Class_Generator_Random_Gaussian_received);

            void Copy(class Gaussian<T> const &ref_source_Class_Generator_Random_Gaussian_received);

            void Range(T const mean_received, T const std_received);

            virtual void Clear(void);

            virtual void Reset(void);

            T operator()(void);

        private:
            T _mean = 0;
            T _std  = 1;

            std::normal_distribution<T> _normal_distribution;
    };
}
