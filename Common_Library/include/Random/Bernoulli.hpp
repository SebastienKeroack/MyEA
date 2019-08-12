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