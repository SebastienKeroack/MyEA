#pragma once

#include <device_launch_parameters.h>

namespace MyEA::Math
{
    __host__ __device__ static
    size_t Recursive_Fused_Multiply_Add(size_t const *const ptr_array_value_received,
                                        size_t const depth_received,
                                        size_t const depth_end_received)
    {
        if(depth_received == depth_end_received) { return(ptr_array_value_received[depth_received]); }

        return(ptr_array_value_received[depth_received] + ptr_array_value_received[depth_received] * Recursive_Fused_Multiply_Add(ptr_array_value_received,
                                                                                                                                  depth_received + 1u,
                                                                                                                                  depth_end_received));
    }

    #pragma warning(push)
    #pragma warning(disable : 4293)
    template<typename T> __host__ __device__
    T Reverse_Integer(T const integer_received)
    {
        if(std::is_same<T, int>::value
           ||
           std::is_same<T, long>::value
           ||
           std::is_same<T, unsigned int>::value
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
        else if(std::is_same<T, long long>::value
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
    }
    #pragma warning(pop)

    template<typename T> __host__ __device__
    bool Is_NaN(T const value_received)
    {
        return((value_received == value_received) == false);
    }

    template<typename T> __host__ __device__
    T Sign(T const value_received)
    {
        return(static_cast<T>(T(0) < value_received) - static_cast<T>(value_received < T(0)));
    }

    template<typename T> __host__ __device__
    T Absolute(T const value_received)
    {
        return(value_received >= 0 ? value_received : -value_received);
    }

    template<typename T> __host__ __device__
    T Maximum(T const x_received, T const y_received)
    {
        return(x_received > y_received ? x_received : y_received);
    }

    template<typename T> __host__ __device__
    T Minimum(T const x_received, T const y_received)
    {
        return(x_received < y_received ? x_received : y_received);
    }

    template<typename T> __host__ __device__
    T Clip(T const value_received,
           T const minimum_received,
           T const maximum_received)
    {
        return((value_received < minimum_received) ? minimum_received
                                                     :
                                                     ((value_received > maximum_received) ? maximum_received
                                                                                            :
                                                                                            value_received));
    }

    template<typename T> __host__ __device__
    T Round_Up_At_Power_Of_Two(T const value_received)
    {
        unsigned int tmp_value(static_cast<unsigned int>(value_received));

        --tmp_value;

        tmp_value |= tmp_value >> 1;
        tmp_value |= tmp_value >> 2;
        tmp_value |= tmp_value >> 4;
        tmp_value |= tmp_value >> 8;
        tmp_value |= tmp_value >> 16;
        //tmp_value |= tmp_value >> 32; unsigned long long, 64bit

        ++tmp_value;

        return(static_cast<T>(tmp_value));
    }

    template<typename T> __host__ __device__
    T Round_Down_At_Power_Of_Two(T const value_received)
    {
        unsigned int tmp_value(static_cast<unsigned int>(value_received));

        tmp_value |= tmp_value >> 1;
        tmp_value |= tmp_value >> 2;
        tmp_value |= tmp_value >> 4;
        tmp_value |= tmp_value >> 8;
        tmp_value |= tmp_value >> 16;
        //tmp_value |= tmp_value >> 32; unsigned long long, 64bit

        tmp_value -= tmp_value >> 1;

        return(static_cast<T>(tmp_value));
    }

    // TODO: Optimize function with bit operation.
    template<typename T> __host__ __device__
    T Round_Up_At_32(T const value_received) // floor((n + k - 1) / k) * k
    {
        unsigned int tmp_value(static_cast<unsigned int>(value_received));

        tmp_value = static_cast<unsigned int>(floor(static_cast<double>(tmp_value + 31u)) / 32.0) * 32u;

        return(static_cast<T>(tmp_value));
    }

    // TODO: Optimize function with bit operation.
    template<typename T> __host__ __device__
    T Round_Down_At_32(T const value_received) // floor(n / k) * k
    {
        unsigned int tmp_value(static_cast<unsigned int>(value_received));

        tmp_value = static_cast<unsigned int>(floor(static_cast<double>(tmp_value) / 32.0)) * 32u;
        tmp_value = MyEA::Math::Maximum(1u, tmp_value);

        return(static_cast<T>(tmp_value));
    }

    template<typename T> __host__ __device__
    bool Is_A_Power_Of_Two(T const value_received)
    {
        unsigned int const tmp_value(static_cast<unsigned int>(value_received));

        return(tmp_value && !(tmp_value & (tmp_value - 1)));
    }
}
