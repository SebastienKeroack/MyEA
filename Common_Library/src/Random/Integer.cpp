#include "pch.hpp"

// This.
#include <Random/Integer.hpp>

namespace MyEA::Random
{
    template<typename T>
    Integer<T>::Integer(void) : Base()
    {
        this->_uniform_int_distribution.param(typename std::uniform_int_distribution<T>::param_type(this->_minimum_range, this->_maximum_range));
    }

    template<typename T>
    Integer<T>::Integer(T const minimum_range_received,
                        T const maximum_range_received,
                        unsigned int const seed_received) : Base(seed_received)
    {
        this->Range(minimum_range_received, maximum_range_received);
    }

    template<typename T>
    class Integer<T>& Integer<T>::operator=(class Integer<T> const &ref_source_Class_Generator_Random_Int_received)
    {
        if(&ref_source_Class_Generator_Random_Int_received != this) { this->Copy(ref_source_Class_Generator_Random_Int_received); }

        return(*this);
    }

    template<typename T>
    void Integer<T>::Copy(class Integer<T> const &ref_source_Class_Generator_Random_Int_received)
    {
        Base::Copy(ref_source_Class_Generator_Random_Int_received);

        this->_minimum_range = ref_source_Class_Generator_Random_Int_received._minimum_range;
        this->_maximum_range = ref_source_Class_Generator_Random_Int_received._maximum_range;

        this->_uniform_int_distribution = ref_source_Class_Generator_Random_Int_received._uniform_int_distribution;
    }

    template<typename T>
    void Integer<T>::Range(T const minimum_range_received, T const maximum_range_received)
    {
        BOOST_ASSERT_MSG(minimum_range_received < maximum_range_received, "`minimum_range_received` can not be less than `minimum_range_received`");

        if(this->_minimum_range == minimum_range_received && this->_maximum_range == maximum_range_received) { return; }

        this->_minimum_range = minimum_range_received;
        this->_maximum_range = maximum_range_received;

        this->_uniform_int_distribution.param(typename std::uniform_int_distribution<T>::param_type(minimum_range_received, maximum_range_received));
    }

    template<typename T>
    void Integer<T>::Clear(void)
    {
        Base::Clear();

        this->Range(T(0), T(1));
    }

    template<typename T>
    void Integer<T>::Reset(void)
    {
        Base::Reset();

        this->_uniform_int_distribution.reset();
    }

    template<typename T>
    T Integer<T>::operator()(void) { return(this->_uniform_int_distribution(this->p_generator_mt19937)); }

    // |STR| Template initialization declaration. |STR|
    template class Integer<short             >;
    template class Integer<unsigned short    >;
    template class Integer<int               >;
    template class Integer<unsigned int      >;
    template class Integer<long              >;
    template class Integer<unsigned long     >;
    template class Integer<long long         >;
    template class Integer<unsigned long long>;
    // |END| Template initialization declaration. |END|
}
