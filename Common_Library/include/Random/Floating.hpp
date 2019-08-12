#pragma once

// This.
#include <Random/Base.hpp>

namespace MyEA::Random
{
    template<typename T>
    class Floating : public Base
    {
        public:
            Floating(void);

            Floating(T const minimum_range_received,
                     T const maximum_range_received,
                     unsigned int const seed_received = _SEED());

            class Floating<T>& operator=(class Floating<T> const &ref_source_Class_Generator_Random_Real_received);

            void Copy(class Floating<T> const &ref_source_Class_Generator_Random_Real_received);

            void Range(T const minimum_range_received, T const maximum_range_received);

            virtual void Clear(void);

            virtual void Reset(void);

            T operator()(void);

        private:
            T _minimum_range = 0;
            T _maximum_range = 1;

            std::uniform_real_distribution<T> _uniform_real_distribution;
    };
}
