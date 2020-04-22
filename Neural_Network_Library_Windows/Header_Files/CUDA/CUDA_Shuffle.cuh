#pragma once

#include <Configuration/Configuration.hpp>

#include <device_launch_parameters.h>

struct curandStateMtgp32;

namespace Shuffle
{
    template<typename T>
    __device__ void Tree_Shift_Shuffle(size_t const size_received,
                                                      size_t const minimum_threads_occupancy_received,
                                                      T *const ptr_array_shuffle_received,
                                                      struct curandStateMtgp32 *const ptr_cuRAND_State_MTGP32_received,
                                                      struct dim3 const *const ptr_dimension_grid_received,
                                                      struct dim3 const *const ptr_dimension_block_received);

    template<typename T>
    __device__ void Tree_Shuffle(size_t const size_received,
                                              size_t const size_block_received,
                                              size_t const size_array_received,
                                              T *const ptr_array_shuffle_received,
                                              struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                              struct dim3 const *const ptr_dimension_grid_received,
                                              struct dim3 const *const ptr_dimension_block_received);
}