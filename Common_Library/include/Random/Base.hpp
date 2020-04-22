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

// Standard.
#include <random>

namespace MyEA::Random
{
    constexpr
    unsigned int _SEED(void) { return(5489u); }

    class Base
    {
        protected:
            unsigned int p_seed;

            std::mt19937 p_generator_mt19937; // https://fr.wikipedia.org/wiki/Mersenne_Twister

        public:
            Base(unsigned int const seed_received = _SEED());

            virtual ~Base(void);

            class Base& operator=(class Base const &ref_source_Class_Generator_Random_received);

            void Copy(class Base const &ref_source_Class_Generator_Random_received);

            void Seed(unsigned int const seed_received);

            virtual void Clear(void);

            virtual void Reset(void);
    };
}
