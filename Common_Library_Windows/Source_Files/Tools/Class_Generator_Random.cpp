#include "stdafx.hpp"

#include <Tools/Class_Generator_Random.hpp>

#include <Math/Mathematic.hpp>

#include <Strings/String.hpp>

namespace MyEA
{
    namespace Common
    {
        Class_Generator_Random::Class_Generator_Random(void) { }
        
        Class_Generator_Random::Class_Generator_Random(unsigned int const seed_received) { this->Seed(seed_received); }
        
        class Class_Generator_Random& Class_Generator_Random::operator=(class Class_Generator_Random const &ref_source_Class_Generator_Random_received)
        {
            if(&ref_source_Class_Generator_Random_received != this) { this->Copy(ref_source_Class_Generator_Random_received); }

            return(*this);
        }

        Class_Generator_Random::~Class_Generator_Random(void) { }
        
        void Class_Generator_Random::Copy(class Class_Generator_Random const &ref_source_Class_Generator_Random_received)
        {
            this->p_generator_mt19937 = ref_source_Class_Generator_Random_received.p_generator_mt19937;
            
            this->p_seed = ref_source_Class_Generator_Random_received.p_seed;
        }
        
        void Class_Generator_Random::Seed(unsigned int const seed_received)
        {
            this->p_generator_mt19937.seed(seed_received);
        
            this->p_seed = seed_received;
        }

        void Class_Generator_Random::Clear(void) { this->p_seed = 5413u; }
        
        void Class_Generator_Random::Reset(void) { this->Seed(this->p_seed); }
        
        long unsigned int Class_Generator_Random::Generate_Integer(void) { return(this->p_generator_mt19937()); }
        
        template<typename T> Class_Generator_Random_Int<T>::Class_Generator_Random_Int(void) : Class_Generator_Random() { this->_uniform_int_distribution.param(typename std::uniform_int_distribution<T>::param_type(this->_minimum_range, this->_maximum_range)); }
        
        template<typename T> Class_Generator_Random_Int<T>::Class_Generator_Random_Int(unsigned int const seed_received,
                                                                                                       T const minimum_range_received,
                                                                                                       T const maximum_range_received) : Class_Generator_Random(seed_received)
        {
            T tmp_minimum_range(MyEA::Math::Minimum<T>(minimum_range_received, maximum_range_received)),
               tmp_maximum_range(MyEA::Math::Maximum<T>(minimum_range_received, maximum_range_received));

            this->_minimum_range = tmp_minimum_range;
            this->_maximum_range = tmp_maximum_range;
            
            this->_uniform_int_distribution.param(typename std::uniform_int_distribution<T>::param_type(this->_minimum_range, this->_maximum_range));
        }
        
        template<typename T> class Class_Generator_Random_Int<T>& Class_Generator_Random_Int<T>::operator=(class Class_Generator_Random_Int<T> const &ref_source_Class_Generator_Random_Int_received)
        {
            if(&ref_source_Class_Generator_Random_Int_received != this) { this->Copy(ref_source_Class_Generator_Random_Int_received); }

            return(*this);
        }

        template<typename T> void Class_Generator_Random_Int<T>::Copy(class Class_Generator_Random_Int<T> const &ref_source_Class_Generator_Random_Int_received)
        {
            Class_Generator_Random::Copy(ref_source_Class_Generator_Random_Int_received);

            this->_minimum_range = ref_source_Class_Generator_Random_Int_received._minimum_range;
            this->_maximum_range = ref_source_Class_Generator_Random_Int_received._maximum_range;
            
            this->_uniform_int_distribution = ref_source_Class_Generator_Random_Int_received._uniform_int_distribution;
        }
        
        template<typename T> void Class_Generator_Random_Int<T>::Range(T const minimum_range_received, T const maximum_range_received)
        {
            T tmp_minimum_range(MyEA::Math::Minimum<T>(minimum_range_received, maximum_range_received)),
               tmp_maximum_range(MyEA::Math::Maximum<T>(minimum_range_received, maximum_range_received));
            
            if(this->_minimum_range == tmp_minimum_range && this->_maximum_range == tmp_maximum_range) { return; }

            this->_minimum_range = tmp_minimum_range;
            this->_maximum_range = tmp_maximum_range;
            
            this->_uniform_int_distribution.param(typename std::uniform_int_distribution<T>::param_type(this->_minimum_range, this->_maximum_range));
        }
        
        template<typename T> void Class_Generator_Random_Int<T>::Clear(void)
        {
            this->Class_Generator_Random::Clear();

            this->Range(T(0), T(1));
        }
        
        template<typename T> void Class_Generator_Random_Int<T>::Reset(void)
        {
            this->Class_Generator_Random::Reset();

            this->_uniform_int_distribution.reset();
        }
        
        template<typename T> T Class_Generator_Random_Int<T>::Generate_Integer(void) { return(this->_uniform_int_distribution(this->p_generator_mt19937)); }

        template<typename T> Class_Generator_Random_Real<T>::Class_Generator_Random_Real(void) : Class_Generator_Random() { this->_uniform_real_distribution.param(typename std::uniform_real_distribution<T>::param_type(this->_minimum_range, this->_maximum_range)); }
        
        template<typename T> Class_Generator_Random_Real<T>::Class_Generator_Random_Real(unsigned int const seed_received,
                                                                                                             T const minimum_range_received,
                                                                                                             T const maximum_range_received) : Class_Generator_Random(seed_received)
        {
            T tmp_minimum_range(MyEA::Math::Minimum<T>(minimum_range_received, maximum_range_received)),
               tmp_maximum_range(MyEA::Math::Maximum<T>(minimum_range_received, maximum_range_received));

            this->_minimum_range = tmp_minimum_range;
            this->_maximum_range = tmp_maximum_range;
            
            this->_uniform_real_distribution.param(typename std::uniform_real_distribution<T>::param_type(this->_minimum_range, this->_maximum_range));
        }
        
        template<typename T> class Class_Generator_Random_Real<T>& Class_Generator_Random_Real<T>::operator=(class Class_Generator_Random_Real<T> const &ref_source_Class_Generator_Random_Real_received)
        {
            if(&ref_source_Class_Generator_Random_Real_received != this) { this->Copy(ref_source_Class_Generator_Random_Real_received); }

            return(*this);
        }

        template<typename T> void Class_Generator_Random_Real<T>::Copy(class Class_Generator_Random_Real<T> const &ref_source_Class_Generator_Random_Real_received)
        {
            Class_Generator_Random::Copy(ref_source_Class_Generator_Random_Real_received);

            this->_minimum_range = ref_source_Class_Generator_Random_Real_received._minimum_range;
            this->_maximum_range = ref_source_Class_Generator_Random_Real_received._maximum_range;
            
            this->_uniform_real_distribution = ref_source_Class_Generator_Random_Real_received._uniform_real_distribution;
        }
        
        template<typename T> void Class_Generator_Random_Real<T>::Range(T const minimum_range_received, T const maximum_range_received)
        {
            T tmp_minimum_range(MyEA::Math::Minimum<T>(minimum_range_received, maximum_range_received)),
               tmp_maximum_range(MyEA::Math::Maximum<T>(minimum_range_received, maximum_range_received));
            
            if(this->_minimum_range == tmp_minimum_range && this->_maximum_range == tmp_maximum_range) { return; }

            this->_minimum_range = tmp_minimum_range;
            this->_maximum_range = tmp_maximum_range;
            
            this->_uniform_real_distribution.param(typename std::uniform_real_distribution<T>::param_type(this->_minimum_range, this->_maximum_range));
        }
        
        template<typename T> void Class_Generator_Random_Real<T>::Clear(void)
        {
            this->Class_Generator_Random::Clear();

            this->Range(T(0), T(1));
        }
        
        template<typename T> void Class_Generator_Random_Real<T>::Reset(void)
        {
            this->Class_Generator_Random::Reset();

            this->_uniform_real_distribution.reset();
        }
        
        template<typename T> T Class_Generator_Random_Real<T>::Generate_Real(void) { return(this->_uniform_real_distribution(this->p_generator_mt19937)); }

        template<typename T> Class_Generator_Random_Gaussian<T>::Class_Generator_Random_Gaussian(void) : Class_Generator_Random() { this->_normal_distribution.param(typename std::normal_distribution<T>::param_type(this->_mean, this->_std)); }
        
        template<typename T> Class_Generator_Random_Gaussian<T>::Class_Generator_Random_Gaussian(unsigned int const seed_received,
                                                                                                                                                              T const mean_received,
                                                                                                                                                              T const std_received) : Class_Generator_Random(seed_received)
        {
            this->_mean = mean_received;
            this->_std = std_received;
            
            this->_normal_distribution.param(typename std::normal_distribution<T>::param_type(this->_mean, this->_std));
        }
        
        template<typename T> class Class_Generator_Random_Gaussian<T>& Class_Generator_Random_Gaussian<T>::operator=(class Class_Generator_Random_Gaussian<T> const &ref_source_Class_Generator_Random_Gaussian_received)
        {
            if(&ref_source_Class_Generator_Random_Gaussian_received != this) { this->Copy(ref_source_Class_Generator_Random_Gaussian_received); }

            return(*this);
        }

        template<typename T> void Class_Generator_Random_Gaussian<T>::Copy(class Class_Generator_Random_Gaussian<T> const &ref_source_Class_Generator_Random_Gaussian_received)
        {
            Class_Generator_Random::Copy(ref_source_Class_Generator_Random_Gaussian_received);

            this->_mean = ref_source_Class_Generator_Random_Gaussian_received._mean;
            this->_std = ref_source_Class_Generator_Random_Gaussian_received._std;
            
            this->_normal_distribution = ref_source_Class_Generator_Random_Gaussian_received._normal_distribution;
        }
        
        template<typename T> void Class_Generator_Random_Gaussian<T>::Range(T const mean_received, T const std_received)
        {
            if(this->_mean == mean_received && this->_std == std_received) { return; }

            this->_mean = mean_received;
            this->_std = std_received;
            
            this->_normal_distribution.param(typename std::normal_distribution<T>::param_type(this->_mean, this->_std));
        }
        
        template<typename T> void Class_Generator_Random_Gaussian<T>::Clear(void)
        {
            this->Class_Generator_Random::Clear();

            this->Range(T(0), T(1));
        }
        
        template<typename T> void Class_Generator_Random_Gaussian<T>::Reset(void)
        {
            this->Class_Generator_Random::Reset();

            this->_normal_distribution.reset();
        }
        
        template<typename T> T Class_Generator_Random_Gaussian<T>::Generate_Gaussian(void) { return(this->_normal_distribution(this->p_generator_mt19937)); }

        template<typename T> Class_Generator_Random_Bernoulli<T>::Class_Generator_Random_Bernoulli(void) : Class_Generator_Random() { }
        
        template<typename T> Class_Generator_Random_Bernoulli<T>::Class_Generator_Random_Bernoulli(unsigned int const seed_received) : Class_Generator_Random(seed_received) { }
        
        template<typename T> class Class_Generator_Random_Bernoulli<T>& Class_Generator_Random_Bernoulli<T>::operator=(class Class_Generator_Random_Bernoulli<T> const &ref_source_Class_Generator_Random_Bernoulli_received)
        {
            if(&ref_source_Class_Generator_Random_Bernoulli_received != this) { this->Copy(ref_source_Class_Generator_Random_Bernoulli_received); }

            return(*this);
        }

        template<typename T> bool Class_Generator_Random_Bernoulli<T>::Generate_Bernoulli(void) { return(this->_bernoulli_distribution(this->p_generator_mt19937)); }
        
        template<typename T> void Class_Generator_Random_Bernoulli<T>::Copy(class Class_Generator_Random_Bernoulli<T> const &ref_source_Class_Generator_Random_Bernoulli_received)
        {
            Class_Generator_Random::Copy(ref_source_Class_Generator_Random_Bernoulli_received);

            this->_probability = ref_source_Class_Generator_Random_Bernoulli_received._probability;
            
            this->_bernoulli_distribution = ref_source_Class_Generator_Random_Bernoulli_received._bernoulli_distribution;
        }
        
        template<typename T> void Class_Generator_Random_Bernoulli<T>::Probability(T const probability_received)
        {
            if(probability_received < T(0) || probability_received > T(1)) { return; }
            else if(this->_probability == probability_received) { return; }
            
            std::bernoulli_distribution::param_type tmp_bernoulli_distribution_param_type(probability_received);
            this->_bernoulli_distribution.param(tmp_bernoulli_distribution_param_type);
        }
        
        template<typename T> void Class_Generator_Random_Bernoulli<T>::Clear(void)
        {
            this->Class_Generator_Random::Clear();

            this->_probability = T(0);
        }
        
        template<typename T> void Class_Generator_Random_Bernoulli<T>::Reset(void)
        {
            this->Class_Generator_Random::Reset();

            this->_bernoulli_distribution.reset();
        }
        
        // template initialization declaration.
        template class Class_Generator_Random_Int<short>;
        template class Class_Generator_Random_Int<unsigned short>;
        template class Class_Generator_Random_Int<int>;
        template class Class_Generator_Random_Int<unsigned int>;
        template class Class_Generator_Random_Int<long>;
        template class Class_Generator_Random_Int<unsigned long>;
        template class Class_Generator_Random_Int<long long>;
        template class Class_Generator_Random_Int<unsigned long long>;
        template class Class_Generator_Random_Real<float>;
        template class Class_Generator_Random_Real<double>;
        template class Class_Generator_Random_Real<long double>;
        template class Class_Generator_Random_Gaussian<float>;
        template class Class_Generator_Random_Gaussian<double>;
        template class Class_Generator_Random_Gaussian<long double>;
        template class Class_Generator_Random_Bernoulli<float>;
        template class Class_Generator_Random_Bernoulli<double>;
        template class Class_Generator_Random_Bernoulli<long double>;
    }
}
