#include "stdafx.hpp"

// This.
#include <Math/MODWT.hpp>
#include <Reallocate/Reallocate.hpp>
#include <Strings/String.hpp>

template<typename T>
void Circular_Convolve_Decomposition(size_t const j_level_received,
                                     size_t const size_inputs_received,
                                     size_t const size_filters_received,
                                     T const *const ptr_array_filters_pass_received,
                                     T const *const ptr_array_inputs_received,
                                     T *const ptr_array_outputs_received)
{
    size_t tmp_time_step,
           tmp_filter_index;

    T tmp_summation;

    for(tmp_time_step = 0_zu; tmp_time_step != size_inputs_received; ++tmp_time_step)
    {
        tmp_summation = T(0);

        for(tmp_filter_index = 0_zu; tmp_filter_index != size_filters_received; ++tmp_filter_index)
        {
            tmp_summation += ptr_array_inputs_received[(tmp_time_step + size_inputs_received - static_cast<size_t>(pow(T(2), static_cast<T>(j_level_received) - T(1))) * tmp_filter_index) % size_inputs_received] * ptr_array_filters_pass_received[tmp_filter_index];
        }

        ptr_array_outputs_received[tmp_time_step] = tmp_summation;
    }
}

template<typename T>
bool MODWT(size_t const size_array_received,
           size_t &ref_size_matrix_received,
           T const *const ptr_array_inputs_received,
           T *&ptr_array_outputs_received,
           size_t J_level_received)
{
    size_t const tmp_maximum_level(static_cast<size_t>(floor(log(static_cast<T>(size_array_received)) / log(T(2)))));

    // |STR| Safety. |STR|
    if(J_level_received > tmp_maximum_level)
    {
        MyEA::String::Error("J level (%zu) greater than maximum level (%zu).",
                            J_level_received,
                            tmp_maximum_level);

        return(false);
    }
    else if(J_level_received == 0_zu) { J_level_received = tmp_maximum_level; }
    // |END| Safety. |END|

    // |STR| Output. |STR|
    if(ptr_array_outputs_received != nullptr)
    {
        if(ref_size_matrix_received != size_array_received * (J_level_received + 1_zu))
        {
            ptr_array_outputs_received = MyEA::Memory::Cpp::Reallocate<T, false, false>(ptr_array_outputs_received,
                                                                                        size_array_received * (J_level_received + 1_zu),
                                                                                        ref_size_matrix_received);

            if(ptr_array_outputs_received == nullptr)
            {
                MyEA::String::Error("An error has been triggered from the `Reallocate<%zu, false, false>(ptr, %zu, %zu)` function.",
                                    sizeof(T),
                                    size_array_received * (J_level_received + 1_zu),
                                    ref_size_matrix_received);

                return(false);
            }
        }
    }
    else if((ptr_array_outputs_received = new T[size_array_received * (J_level_received + 1_zu)]) == nullptr)
    {
        MyEA::String::Error("Cannot allocate %zu bytes.", size_array_received * (J_level_received + 1_zu) * sizeof(T));

        return(false);
    }

    ref_size_matrix_received = size_array_received * (J_level_received + 1_zu);
    // |END| Output. |END|

    // |STR| Array previous approximations. |STR|
    T *tmp_ptr_array_previous_approximations;

    if((tmp_ptr_array_previous_approximations = new T[size_array_received]) == nullptr)
    {
        MyEA::String::Error("Cannot allocate %zu bytes.", size_array_received * sizeof(T));

        return(false);
    }

    memcpy(tmp_ptr_array_previous_approximations,
           ptr_array_inputs_received,
           size_array_received * sizeof(T));
    // |END| Array previous approximations. |END|
    
    // |STR| Filters. |STR|
    // High pass db1: {1 / sqrt(2) * 1, 1 / sqrt(2) * -1}
    // Low pass db1: {1 / sqrt(2), 1 / sqrt(2)}
    //T tmp_high_pass__db1[2] = {T(-7.071067811865475244008443621048490392848359376884740365883398e-01), T(7.071067811865475244008443621048490392848359376884740365883398e-01)};
    T tmp_high_pass__db1[2] = {T(7.071067811865475244008443621048490392848359376884740365883398e-01), T(-7.071067811865475244008443621048490392848359376884740365883398e-01)};
    T tmp_low_pass__db1[2]  = {T(7.071067811865475244008443621048490392848359376884740365883398e-01), T( 7.071067811865475244008443621048490392848359376884740365883398e-01)};

    tmp_high_pass__db1[0] /= sqrt(T(2));
    tmp_high_pass__db1[1] /= sqrt(T(2));
    tmp_low_pass__db1[0] /= sqrt(T(2));
    tmp_low_pass__db1[1] /= sqrt(T(2));
    // |END| Filters. |END|

    for(size_t tmp_j_level(0_zu); tmp_j_level != J_level_received; ++tmp_j_level)
    {
        Circular_Convolve_Decomposition(tmp_j_level + 1_zu,
                                        size_array_received,
                                        2_zu,
                                        tmp_high_pass__db1,
                                        tmp_ptr_array_previous_approximations,
                                        ptr_array_outputs_received + tmp_j_level * size_array_received);

        Circular_Convolve_Decomposition(tmp_j_level + 1_zu,
                                        size_array_received,
                                        2_zu,
                                        tmp_low_pass__db1,
                                        tmp_ptr_array_previous_approximations,
                                        ptr_array_outputs_received + (tmp_j_level + 1_zu) * size_array_received);
        
        if(tmp_j_level + 1_zu != J_level_received)
        {
            memcpy(tmp_ptr_array_previous_approximations,
                   ptr_array_outputs_received + (tmp_j_level + 1_zu) * size_array_received,
                   size_array_received * sizeof(T));
        }
    }
    
    delete[](tmp_ptr_array_previous_approximations);

    return(true);
}
template bool MODWT<T_>(size_t const,
                        size_t &,
                        T_ const *const,
                        T_ *&,
                        size_t);

template<typename T>
void Circular_Convolve_Reconstruction(size_t const j_level_received,
                                      size_t const size_inputs_received,
                                      size_t const size_filters_received,
                                      T const *const ptr_array_filters_high_pass_received,
                                      T const *const ptr_array_filters_low_pass_received,
                                      T const *const ptr_array_inputs_received,
                                      T const *const ptr_array_previous_inputs_received,
                                      T *const ptr_array_outputs_received)
{
    size_t tmp_time_step,
           tmp_filter_index;

    T tmp_summation;

    for(tmp_time_step = 0_zu; tmp_time_step != size_inputs_received; ++tmp_time_step)
    {
        tmp_summation = T(0);

        for(tmp_filter_index = 0_zu; tmp_filter_index != size_filters_received; ++tmp_filter_index)
        {
            tmp_summation += ptr_array_inputs_received[(tmp_time_step + static_cast<size_t>(pow(T(2), static_cast<T>(j_level_received) - T(1))) * tmp_filter_index) % size_inputs_received] * ptr_array_filters_high_pass_received[tmp_filter_index];

            tmp_summation += ptr_array_previous_inputs_received[(tmp_time_step + static_cast<size_t>(pow(T(2), static_cast<T>(j_level_received) - T(1))) * tmp_filter_index) % size_inputs_received] * ptr_array_filters_low_pass_received[tmp_filter_index];
        }

        ptr_array_outputs_received[tmp_time_step] = tmp_summation;
    }
}

template<typename T>
bool MODWT_Inverse(size_t const size_matrix_received,
                   size_t const size_array_received,
                   T const *const ptr_array_inputs_received,
                   T *&ptr_array_outputs_received,
                   size_t J_level_received)
{
    size_t const tmp_output_level(size_matrix_received / size_array_received - 1_zu);
    size_t tmp_reverse_j_level,
           tmp_j_level;

    // |STR| Safety. |STR|
    if(size_matrix_received == 0_zu)
    {
        MyEA::String::Error("`size_matrix_received` can not be equal to zero.");

        return(false);
    }
    else if(size_array_received == 0_zu)
    {
        MyEA::String::Error("`size_array_received` can not be equal to zero.");

        return(false);
    }
    else if(J_level_received > tmp_output_level)
    {
        MyEA::String::Error("J level (%zu) greater than allowable level (%zu).",
                            J_level_received,
                            tmp_output_level);

        return(false);
    }

    if(J_level_received == 0_zu) { J_level_received = tmp_output_level                   ; }
    else                         { J_level_received = tmp_output_level - J_level_received; }
    // |END| Safety. |END|

    // |STR| Output. |STR|
    if(ptr_array_outputs_received != nullptr)
    {
        ptr_array_outputs_received = MyEA::Memory::Cpp::Reallocate<T, false, false>(ptr_array_outputs_received,
                                                                                    size_array_received,
                                                                                    0_zu);

        if(ptr_array_outputs_received == nullptr)
        {
            MyEA::String::Error("An error has been triggered from the `Reallocate<%zu, false, false>(ptr, %zu, %zu)` function.",
                                sizeof(T),
                                size_array_received,
                                0_zu);

            return(false);
        }
    }
    else if((ptr_array_outputs_received = new T[size_array_received]) == nullptr)
    {
        MyEA::String::Error("Cannot allocate %zu bytes.", size_array_received * sizeof(T));

        return(false);
    }

    memset(ptr_array_outputs_received,
           0,
           size_array_received * sizeof(T));
    // |END| Output. |END|

    // |STR| Array next approximations. |STR|
    T *tmp_ptr_array_next_approximations;

    if((tmp_ptr_array_next_approximations = new T[size_array_received]) == nullptr)
    {
        MyEA::String::Error("Cannot allocate %zu bytes.", size_array_received * sizeof(T));

        return(false);
    }

    memcpy(tmp_ptr_array_next_approximations,
           ptr_array_inputs_received + tmp_output_level * size_array_received,
           size_array_received * sizeof(T));
    // |END| Array next approximations. |END|
    
    // |STR| Filters. |STR|
    // High pass db1/haar: {1 / sqrt(2) * 1, 1 / sqrt(2) * -1}
    // Low pass db1/haar: {1 / sqrt(2), 1 / sqrt(2)}
    //T tmp_high_pass__db1[2] = {T(-7.071067811865475244008443621048490392848359376884740365883398e-01), T(7.071067811865475244008443621048490392848359376884740365883398e-01)};
    T tmp_high_pass__db1[2] = {T(7.071067811865475244008443621048490392848359376884740365883398e-01), T(-7.071067811865475244008443621048490392848359376884740365883398e-01)};
    T tmp_low_pass__db1[2]  = {T(7.071067811865475244008443621048490392848359376884740365883398e-01), T( 7.071067811865475244008443621048490392848359376884740365883398e-01)};

    tmp_high_pass__db1[0] /= sqrt(T(2));
    tmp_high_pass__db1[1] /= sqrt(T(2));
    tmp_low_pass__db1[0] /= sqrt(T(2));
    tmp_low_pass__db1[1] /= sqrt(T(2));
    // |END| Filters. |END|

    for(tmp_j_level = 0_zu; tmp_j_level != J_level_received; ++tmp_j_level)
    {
        tmp_reverse_j_level = tmp_output_level - tmp_j_level - 1_zu;

        Circular_Convolve_Reconstruction(tmp_reverse_j_level + 1_zu,
                                         size_array_received,
                                         2_zu,
                                         tmp_high_pass__db1,
                                         tmp_low_pass__db1,
                                         ptr_array_inputs_received + tmp_reverse_j_level * size_array_received,
                                         tmp_ptr_array_next_approximations,
                                         ptr_array_outputs_received);

        if(tmp_j_level + 1_zu != J_level_received)
        {
            memcpy(tmp_ptr_array_next_approximations,
                   ptr_array_outputs_received,
                   size_array_received * sizeof(T));
        }
    }

    delete[](tmp_ptr_array_next_approximations);

    return(true);
}
template bool MODWT_Inverse<T_>(size_t const,
                                size_t const,
                                T_ const *const,
                                T_ *&,
                                size_t);