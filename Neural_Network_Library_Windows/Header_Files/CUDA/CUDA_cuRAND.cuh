#pragma once

#include <Tools/Configuration.hpp>

#include <device_launch_parameters.h>

__global__ void kernel__cuRAND__Memcpy_cuRAND_State_MTGP32(struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_destination_received,
                                                                                                       struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_source_received,
                                                                                                       struct mtgp32_kernel_params *const ptr_array_mtgp32_kernel_params_t_source_received);

__global__ void kernel__cuRAND__Memcpy_cuRAND_State_MTGP32(size_t const size_received,
                                                                                                       struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_destination_received,
                                                                                                       struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_source_received,
                                                                                                       struct mtgp32_kernel_params *const ptr_array_mtgp32_kernel_params_t_source_received);

__global__ void kernel_while__cuRAND__Memcpy_cuRAND_State_MTGP32(size_t const size_received,
                                                                                                               struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_destination_received,
                                                                                                               struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_source_received,
                                                                                                               struct mtgp32_kernel_params *const ptr_array_mtgp32_kernel_params_t_source_received);

__device__ void cuRAND__Memcpy_cuRAND_State_MTGP32(size_t const size_received,
                                                                                           struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_destination_received,
                                                                                           struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_source_received,
                                                                                           struct mtgp32_kernel_params *const ptr_array_mtgp32_kernel_params_t_source_received,
                                                                                           struct dim3 const *const ptr_dimension_grid_received,
                                                                                           struct dim3 const *const ptr_dimension_block_received);

__host__ bool Allocate_cuRAND_MTGP32(int const number_states_MTGP32_received,
                                                              size_t seed_received,
                                                              struct mtgp32_kernel_params *&ptr_mtgp32_kernel_params_received,
                                                              struct curandStateMtgp32 *&ptr_curandStateMtgp32_t_received);

__host__ void Cleanup_cuRAND_MTGP32(struct mtgp32_kernel_params *&ptr_mtgp32_kernel_params_received, struct curandStateMtgp32 *&ptr_curandStateMtgp32_t_received);