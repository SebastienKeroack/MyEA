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
