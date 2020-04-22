/* Copyright 2020 Sébastien Kéroack. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

// This.
#include <Configuration/Configuration.hpp>

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