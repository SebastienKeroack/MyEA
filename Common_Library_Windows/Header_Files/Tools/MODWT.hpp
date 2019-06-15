#pragma once

#include <Tools/Configuration.hpp>

template<typename T>
void Circular_Convolve_Decomposition(size_t const j_level_received,
                                                        size_t const size_inputs_received,
                                                        size_t const size_filters_received,
                                                        T const *const ptr_array_filters_pass_received,
                                                        T const *const ptr_array_inputs_received,
                                                        T *const ptr_array_outputs_received);

template<typename T>
bool MODWT(size_t const size_array_received,
                     size_t &ref_size_matrix_received,
                     T const *const ptr_array_inputs_received,
                     T *&ptr_array_outputs_received,
                     size_t J_level_received = 0_zu);

template<typename T>
void Circular_Convolve_Reconstruction(size_t const j_level_received,
                                                        size_t const size_inputs_received,
                                                        size_t const size_filters_received,
                                                        T const *const ptr_array_filters_high_pass_received,
                                                        T const *const ptr_array_filters_low_pass_received,
                                                        T const *const ptr_array_inputs_received,
                                                        T const *const ptr_array_previous_inputs_received,
                                                        T *const ptr_array_outputs_received);

template<typename T>
bool MODWT_Inverse(size_t const size_matrix_received,
                                size_t const size_array_received,
                                T const *const ptr_array_inputs_received,
                                T *&ptr_array_outputs_received,
                                size_t J_level_received = 0_zu);