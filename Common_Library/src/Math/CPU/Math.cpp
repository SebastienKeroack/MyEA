#include "pch.hpp"

// Standard.
#include <cmath>
#include <stdexcept>

// This.
#include <Math/CPU/Math.hpp>

namespace MyEA::Math
{
    size_t Recursive_Fused_Multiply_Add(size_t const *const ptr_array_value_received,
                                        size_t const depth_received,
                                        size_t const depth_end_received)
    {
        if(depth_received == depth_end_received) { return(ptr_array_value_received[depth_received]); }

        return(ptr_array_value_received[depth_received] + ptr_array_value_received[depth_received] * Recursive_Fused_Multiply_Add(ptr_array_value_received,
                                                                                                                                  depth_received + 1u,
                                                                                                                                  depth_end_received));
    }

    template<typename T>
    T Reverse_Integer(T const integer_received)
    {
        if constexpr (std::is_same<T, int          >::value
                      ||
                      std::is_same<T, long         >::value
                      ||
                      std::is_same<T, unsigned int >::value
                      ||
                      std::is_same<T, unsigned long>::value)
        {
            T const c1( integer_received        & 255),
                    c2((integer_received >>  8) & 255),
                    c3((integer_received >> 16) & 255),
                    c4((integer_received >> 24) & 255);

            return((c1 << 24) +
                   (c2 << 16) +
                   (c3 <<  8) +
                    c4);
        }
        else if constexpr (std::is_same<T, long long         >::value
                           ||
                           std::is_same<T, unsigned long long>::value)
        {
            T const c1(integer_received         & 255),
                    c2((integer_received >>  8) & 255),
                    c3((integer_received >> 16) & 255),
                    c4((integer_received >> 24) & 255),
                    c5((integer_received >> 32) & 255),
                    c6((integer_received >> 40) & 255),
                    c7((integer_received >> 48) & 255),
                    c8((integer_received >> 56) & 255);

            return((c1 << 56) +
                   (c2 << 48) +
                   (c3 << 40) +
                   (c4 << 32) +
                   (c5 << 24) +
                   (c6 << 16) +
                   (c7 <<  8) +
                    c8);
        }
        else { throw(std::logic_error("NotImplementedException")); }
    }

    template int                Reverse_Integer<int               >(int                const);
    template long               Reverse_Integer<long              >(long               const);
    template long long          Reverse_Integer<long long         >(long long          const);
    template unsigned int       Reverse_Integer<unsigned int      >(unsigned int       const);
    template unsigned long      Reverse_Integer<unsigned long     >(unsigned long      const);
    template unsigned long long Reverse_Integer<unsigned long long>(unsigned long long const);

    template<typename T>
    bool Is_NaN(T const value_received)
    {
        return(std::isnan(value_received) || std::isinf(value_received));
    }

    template bool Is_NaN<float      >(float       const);
    template bool Is_NaN<double     >(double      const);
    template bool Is_NaN<long double>(long double const);
}