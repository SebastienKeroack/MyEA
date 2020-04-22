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
    class Integer : public Base
    {
        public:
            Integer(void);

            Integer(T const minimum_range_received,
                    T const maximum_range_received,
                    unsigned int const seed_received = _SEED());

            class Integer<T>& operator=(class Integer<T> const &ref_source_Class_Generator_Random_Int_received);

            void Copy(class Integer<T> const &ref_source_Class_Generator_Random_Int_received);

            void Range(T const minimum_range_received, T const maximum_range_received);

            virtual void Clear(void);

            virtual void Reset(void);

            T operator()(void);

        private:
            T _minimum_range = 0;
            T _maximum_range = 1;

            std::uniform_int_distribution<T> _uniform_int_distribution;
    };
}
