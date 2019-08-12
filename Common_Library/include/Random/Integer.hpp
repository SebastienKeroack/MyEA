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
