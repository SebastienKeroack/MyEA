#pragma once

#include <limits>

#if defined(COMPILE_FLOAT)
    typedef float ST_;
    
    #if defined(COMPILE_AUTODIFF)
        // TODO: Replace Adept by Autodiff (https://github.com/autodiff/autodiff)
        #include <adept.h>

        typedef adept::afloat T_;

        inline float Cast_T(T_ T_received) { return(T_received.value()); }
    #else
        typedef float T_;

        inline float Cast_T(T_ T_received) { return(T_received); }
    #endif

    typedef int I_;
    
    constexpr
    T_ T_EPSILON(void) { return(1.192092896e-07f); }
#elif defined(COMPILE_DOUBLE)
    typedef double ST_;
    
    #if defined(COMPILE_AUTODIFF)
        // TODO: Replace Adept by Autodiff (https://github.com/autodiff/autodiff)
        #include <adept.h>
        
        typedef adept::adouble T_;

        inline double Cast_T(T_ T_received) { return(T_received.value()); }
    #else
        typedef double T_;

        inline double Cast_T(T_ T_received) { return(T_received); }
    #endif

    typedef long long I_;
    
    constexpr
    T_ T_EPSILON(void) { return(2.2204460492503131e-016); }
#elif defined(COMPILE_LONG_DOUBLE)
    typedef long double T_;

    typedef long long I_;
    
    constexpr
    T_ T_EPSILON(void) { return(2.2204460492503131e-016L); }
#endif

constexpr
T_ T_EMPTY(void) { return((std::numeric_limits<T_>::max)()); }

constexpr
T_ operator ""_T(unsigned long long int variable_to_size_t_received)
{
    return(static_cast<ST_>(variable_to_size_t_received));
}

constexpr
T_ operator ""_T(long double variable_to_size_t_received)
{
    return(static_cast<ST_>(variable_to_size_t_received));
}