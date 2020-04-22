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
    class Bernoulli : public Base
    {
        public:
            Bernoulli(void);

            Bernoulli(T const probability_received, unsigned int const seed_received = _SEED());

            class Bernoulli& operator=(class Bernoulli const &ref_source_Class_Generator_Random_Bernoulli_received);

            void Copy(class Bernoulli const &ref_source_Class_Generator_Random_Bernoulli_received);

            void Probability(T const probability_received);

            virtual void Clear(void);

            virtual void Reset(void);

            bool operator()(void);

        private:
            T _probability = 0;

            std::bernoulli_distribution _bernoulli_distribution;
    };
}