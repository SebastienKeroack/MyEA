#pragma once

#include <random>

#include <Math/Math.hpp>

namespace MyEA::Random
{
    class Random_Base
    {
        protected:
            unsigned int p_seed = 5489u;

            std::mt19937 p_generator_mt19937; // https://fr.wikipedia.org/wiki/Mersenne_Twister

        public:
            Random_Base(void);

            Random_Base(unsigned int const seed_received);

            virtual ~Random_Base(void);

            class Random_Base& operator=(class Random_Base const &ref_source_Class_Generator_Random_received);

            void Copy(class Random_Base const &ref_source_Class_Generator_Random_received);

            void Seed(unsigned int const seed_received);

            virtual void Clear(void);

            virtual void Reset(void);
    };

    template<typename T>
    class Random_Int : public Random_Base
    {
        public:
            Random_Int(void);

            Random_Int(unsigned int const seed_received,
                       T const minimum_range_received,
                       T const maximum_range_received);

            class Random_Int<T>& operator=(class Random_Int<T> const &ref_source_Class_Generator_Random_Int_received);

            void Copy(class Random_Int<T> const &ref_source_Class_Generator_Random_Int_received);

            void Range(T const minimum_range_received, T const maximum_range_received);

            virtual void Clear(void);

            virtual void Reset(void);

            T Generate_Integer(void);

        private:
            T _minimum_range = 0;
            T _maximum_range = 1;

            std::uniform_int_distribution<T> _uniform_int_distribution;
    };

    template<typename T>
    class Random_Real : public Random_Base
    {
        public:
            Random_Real(void);
            Random_Real(unsigned int const seed_received,
                        T const minimum_range_received,
                        T const maximum_range_received);

            class Random_Real<T>& operator=(class Random_Real<T> const &ref_source_Class_Generator_Random_Real_received);

            void Copy(class Random_Real<T> const &ref_source_Class_Generator_Random_Real_received);

            void Range(T const minimum_range_received, T const maximum_range_received);

            virtual void Clear(void);

            virtual void Reset(void);

            T Generate_Real(void);

        private:
            T _minimum_range = 0;
            T _maximum_range = 1;

            std::uniform_real_distribution<T> _uniform_real_distribution;
    };

    template<typename T>
    class Random_Gaussian : public Random_Base
    {
        public:
            Random_Gaussian(void);
            Random_Gaussian(unsigned int const seed_received,
                            T const mean_received,
                            T const std_received);

            class Random_Gaussian<T>& operator=(class Random_Gaussian<T> const &ref_source_Class_Generator_Random_Gaussian_received);

            void Copy(class Random_Gaussian<T> const &ref_source_Class_Generator_Random_Gaussian_received);

            void Range(T const mean_received, T const std_received);

            virtual void Clear(void);

            virtual void Reset(void);

            T Generate_Gaussian(void);

        private:
            T _mean = 0;
            T _std  = 1;

            std::normal_distribution<T> _normal_distribution;
    };

    template<typename T>
    class Random_Bernoulli : public Random_Base
    {
        public:
            Random_Bernoulli(void);

            Random_Bernoulli(unsigned int const seed_received);

            class Random_Bernoulli& operator=(class Random_Bernoulli const &ref_source_Class_Generator_Random_Bernoulli_received);

            void Copy(class Random_Bernoulli const &ref_source_Class_Generator_Random_Bernoulli_received);

            void Probability(T const probability_received);

            virtual void Clear(void);

            virtual void Reset(void);

            bool Generate_Bernoulli(void);

        private:
            T _probability = 0;

            std::bernoulli_distribution _bernoulli_distribution;
    };
}

#if defined(COMPILE_ADEPT)
    #if defined(COMPILE_FLOAT)
        #include <../Source_Files/Tools/Class_Generator_Random__Adept__Float.cpp>
    #elif defined(COMPILE_DOUBLE)
        #include <../Source_Files/Tools/Class_Generator_Random__Adept__Double.cpp>
    #endif
#endif
