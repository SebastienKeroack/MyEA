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
