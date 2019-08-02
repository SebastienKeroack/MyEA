#include "stdafx.hpp"

#include <Random/Random.hpp>

#include <Math/Math.hpp>

#include <Strings/String.hpp>

namespace MyEA::Random
{
    Random_Base::Random_Base(void) { }

    Random_Base::Random_Base(unsigned int const seed_received)
    {
        this->Seed(seed_received);
    }

    class Random_Base& Random_Base::operator=(class Random_Base const &ref_source_Class_Generator_Random_received)
    {
        if(&ref_source_Class_Generator_Random_received != this) { this->Copy(ref_source_Class_Generator_Random_received); }

        return(*this);
    }

    Random_Base::~Random_Base(void) { }

    void Random_Base::Copy(class Random_Base const &ref_source_Class_Generator_Random_received)
    {
        this->p_generator_mt19937 = ref_source_Class_Generator_Random_received.p_generator_mt19937;

        this->p_seed = ref_source_Class_Generator_Random_received.p_seed;
    }

    void Random_Base::Seed(unsigned int const seed_received)
    {
        this->p_generator_mt19937.seed(seed_received);

        this->p_seed = seed_received;
    }

    void Random_Base::Clear(void) { this->p_seed = 5413u; }

    void Random_Base::Reset(void) { this->Seed(this->p_seed); }

    template<typename T>
    Random_Int<T>::Random_Int(void) : Random_Base()
    {
        this->_uniform_int_distribution.param(typename std::uniform_int_distribution<T>::param_type(this->_minimum_range, this->_maximum_range));
    }

    template<typename T>
    Random_Int<T>::Random_Int(unsigned int const seed_received,
                              T const minimum_range_received,
                              T const maximum_range_received) : Random_Base(seed_received)
    {
        T tmp_minimum_range(MyEA::Math::Minimum<T>(minimum_range_received, maximum_range_received)),
          tmp_maximum_range(MyEA::Math::Maximum<T>(minimum_range_received, maximum_range_received));

        this->_minimum_range = tmp_minimum_range;
        this->_maximum_range = tmp_maximum_range;

        this->_uniform_int_distribution.param(typename std::uniform_int_distribution<T>::param_type(this->_minimum_range, this->_maximum_range));
    }

    template<typename T>
    class Random_Int<T>& Random_Int<T>::operator=(class Random_Int<T> const &ref_source_Class_Generator_Random_Int_received)
    {
        if(&ref_source_Class_Generator_Random_Int_received != this) { this->Copy(ref_source_Class_Generator_Random_Int_received); }

        return(*this);
    }

    template<typename T>
    void Random_Int<T>::Copy(class Random_Int<T> const &ref_source_Class_Generator_Random_Int_received)
    {
        Random_Base::Copy(ref_source_Class_Generator_Random_Int_received);

        this->_minimum_range = ref_source_Class_Generator_Random_Int_received._minimum_range;
        this->_maximum_range = ref_source_Class_Generator_Random_Int_received._maximum_range;

        this->_uniform_int_distribution = ref_source_Class_Generator_Random_Int_received._uniform_int_distribution;
    }

    template<typename T>
    void Random_Int<T>::Range(T const minimum_range_received, T const maximum_range_received)
    {
        T tmp_minimum_range(MyEA::Math::Minimum<T>(minimum_range_received, maximum_range_received)),
           tmp_maximum_range(MyEA::Math::Maximum<T>(minimum_range_received, maximum_range_received));

        if(this->_minimum_range == tmp_minimum_range && this->_maximum_range == tmp_maximum_range) { return; }

        this->_minimum_range = tmp_minimum_range;
        this->_maximum_range = tmp_maximum_range;

        this->_uniform_int_distribution.param(typename std::uniform_int_distribution<T>::param_type(this->_minimum_range, this->_maximum_range));
    }

    template<typename T>
    void Random_Int<T>::Clear(void)
    {
        this->Random_Base::Clear();

        this->Range(T(0), T(1));
    }

    template<typename T>
    void Random_Int<T>::Reset(void)
    {
        this->Random_Base::Reset();

        this->_uniform_int_distribution.reset();
    }

    template<typename T>
    T Random_Int<T>::Generate_Integer(void)
    {
        return(this->_uniform_int_distribution(this->p_generator_mt19937));
    }

    template<typename T>
    Random_Real<T>::Random_Real(void) : Random_Base()
    {
        this->_uniform_real_distribution.param(typename std::uniform_real_distribution<T>::param_type(this->_minimum_range, this->_maximum_range));
    }

    template<typename T>
    Random_Real<T>::Random_Real(unsigned int const seed_received,
                                T const minimum_range_received,
                                T const maximum_range_received) : Random_Base(seed_received)
    {
        T tmp_minimum_range(MyEA::Math::Minimum<T>(minimum_range_received, maximum_range_received)),
          tmp_maximum_range(MyEA::Math::Maximum<T>(minimum_range_received, maximum_range_received));

        this->_minimum_range = tmp_minimum_range;
        this->_maximum_range = tmp_maximum_range;

        this->_uniform_real_distribution.param(typename std::uniform_real_distribution<T>::param_type(this->_minimum_range, this->_maximum_range));
    }

    template<typename T>
    class Random_Real<T>& Random_Real<T>::operator=(class Random_Real<T> const &ref_source_Class_Generator_Random_Real_received)
    {
        if(&ref_source_Class_Generator_Random_Real_received != this) { this->Copy(ref_source_Class_Generator_Random_Real_received); }

        return(*this);
    }

    template<typename T>
    void Random_Real<T>::Copy(class Random_Real<T> const &ref_source_Class_Generator_Random_Real_received)
    {
        Random_Base::Copy(ref_source_Class_Generator_Random_Real_received);

        this->_minimum_range = ref_source_Class_Generator_Random_Real_received._minimum_range;
        this->_maximum_range = ref_source_Class_Generator_Random_Real_received._maximum_range;

        this->_uniform_real_distribution = ref_source_Class_Generator_Random_Real_received._uniform_real_distribution;
    }

    template<typename T>
    void Random_Real<T>::Range(T const minimum_range_received, T const maximum_range_received)
    {
        T tmp_minimum_range(MyEA::Math::Minimum<T>(minimum_range_received, maximum_range_received)),
          tmp_maximum_range(MyEA::Math::Maximum<T>(minimum_range_received, maximum_range_received));

        if(this->_minimum_range == tmp_minimum_range && this->_maximum_range == tmp_maximum_range) { return; }

        this->_minimum_range = tmp_minimum_range;
        this->_maximum_range = tmp_maximum_range;

        this->_uniform_real_distribution.param(typename std::uniform_real_distribution<T>::param_type(this->_minimum_range, this->_maximum_range));
    }

    template<typename T>
    void Random_Real<T>::Clear(void)
    {
        this->Random_Base::Clear();

        this->Range(T(0), T(1));
    }

    template<typename T>
    void Random_Real<T>::Reset(void)
    {
        this->Random_Base::Reset();

        this->_uniform_real_distribution.reset();
    }

    template<typename T>
    T Random_Real<T>::Generate_Real(void)
    {
        return(this->_uniform_real_distribution(this->p_generator_mt19937));
    }

    template<typename T>
    Random_Gaussian<T>::Random_Gaussian(void) : Random_Base()
    {
        this->_normal_distribution.param(typename std::normal_distribution<T>::param_type(this->_mean, this->_std));
    }

    template<typename T>
    Random_Gaussian<T>::Random_Gaussian(unsigned int const seed_received,
                                        T const mean_received,
                                        T const std_received) : Random_Base(seed_received)
    {
        this->_mean = mean_received;
        this->_std = std_received;

        this->_normal_distribution.param(typename std::normal_distribution<T>::param_type(this->_mean, this->_std));
    }

    template<typename T>
    class Random_Gaussian<T>& Random_Gaussian<T>::operator=(class Random_Gaussian<T> const &ref_source_Class_Generator_Random_Gaussian_received)
    {
        if(&ref_source_Class_Generator_Random_Gaussian_received != this) { this->Copy(ref_source_Class_Generator_Random_Gaussian_received); }

        return(*this);
    }

    template<typename T>
    void Random_Gaussian<T>::Copy(class Random_Gaussian<T> const &ref_source_Class_Generator_Random_Gaussian_received)
    {
        Random_Base::Copy(ref_source_Class_Generator_Random_Gaussian_received);

        this->_mean = ref_source_Class_Generator_Random_Gaussian_received._mean;
        this->_std = ref_source_Class_Generator_Random_Gaussian_received._std;

        this->_normal_distribution = ref_source_Class_Generator_Random_Gaussian_received._normal_distribution;
    }

    template<typename T>
    void Random_Gaussian<T>::Range(T const mean_received, T const std_received)
    {
        if(this->_mean == mean_received && this->_std == std_received) { return; }

        this->_mean = mean_received;
        this->_std = std_received;

        this->_normal_distribution.param(typename std::normal_distribution<T>::param_type(this->_mean, this->_std));
    }

    template<typename T>
    void Random_Gaussian<T>::Clear(void)
    {
        this->Random_Base::Clear();

        this->Range(T(0), T(1));
    }

    template<typename T>
    void Random_Gaussian<T>::Reset(void)
    {
        this->Random_Base::Reset();

        this->_normal_distribution.reset();
    }

    template<typename T>
    T Random_Gaussian<T>::Generate_Gaussian(void)
    {
        return(this->_normal_distribution(this->p_generator_mt19937));
    }

    template<typename T>
    Random_Bernoulli<T>::Random_Bernoulli(void) : Random_Base() { }

    template<typename T>
    Random_Bernoulli<T>::Random_Bernoulli(unsigned int const seed_received) : Random_Base(seed_received) { }

    template<typename T>
    class Random_Bernoulli<T>& Random_Bernoulli<T>::operator=(class Random_Bernoulli<T> const &ref_source_Class_Generator_Random_Bernoulli_received)
    {
        if(&ref_source_Class_Generator_Random_Bernoulli_received != this) { this->Copy(ref_source_Class_Generator_Random_Bernoulli_received); }

        return(*this);
    }

    template<typename T>
    bool Random_Bernoulli<T>::Generate_Bernoulli(void) { return(this->_bernoulli_distribution(this->p_generator_mt19937)); }

    template<typename T>
    void Random_Bernoulli<T>::Copy(class Random_Bernoulli<T> const &ref_source_Class_Generator_Random_Bernoulli_received)
    {
        Random_Base::Copy(ref_source_Class_Generator_Random_Bernoulli_received);

        this->_probability = ref_source_Class_Generator_Random_Bernoulli_received._probability;

        this->_bernoulli_distribution = ref_source_Class_Generator_Random_Bernoulli_received._bernoulli_distribution;
    }

    template<typename T>
    void Random_Bernoulli<T>::Probability(T const probability_received)
    {
        if(probability_received < T(0) || probability_received > T(1)) { return; }
        else if(this->_probability == probability_received) { return; }

        std::bernoulli_distribution::param_type tmp_bernoulli_distribution_param_type(probability_received);
        this->_bernoulli_distribution.param(tmp_bernoulli_distribution_param_type);
    }

    template<typename T>
    void Random_Bernoulli<T>::Clear(void)
    {
        this->Random_Base::Clear();

        this->_probability = T(0);
    }

    template<typename T>
    void Random_Bernoulli<T>::Reset(void)
    {
        this->Random_Base::Reset();

        this->_bernoulli_distribution.reset();
    }

    // |STR| Template initialization declaration. |STR|
    template class Random_Int<short             >;
    template class Random_Int<unsigned short    >;
    template class Random_Int<int               >;
    template class Random_Int<unsigned int      >;
    template class Random_Int<long              >;
    template class Random_Int<unsigned long     >;
    template class Random_Int<long long         >;
    template class Random_Int<unsigned long long>;

    template class Random_Real<float      >;
    template class Random_Real<double     >;
    template class Random_Real<long double>;

    template class Random_Gaussian<float      >;
    template class Random_Gaussian<double     >;
    template class Random_Gaussian<long double>;

    template class Random_Bernoulli<float      >;
    template class Random_Bernoulli<double     >;
    template class Random_Bernoulli<long double>;
    // |END| Template initialization declaration. |END|
}
