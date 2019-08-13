#pragma once

// This.
#include <Configuration/Configuration.hpp>

// Forward declaration.
namespace MyEA::Memory::Cpp
{
    template<class T> __host__ __device__
    void Fill_Nullptr(T *ptr_array_received, T const *const ptr_array_last_received);

    template<class T, bool CPY = true, bool SET = true> __host__ __device__
    T* Reallocate(T *ptr_array_received,
                  size_t const new_size_received,
                  size_t const old_size_received);

    // TODO: WARNING dimension set/cpy are inverted.
    template<class T, bool CPY = true, bool SET = true> __device__
    T* Reallocate(T *ptr_array_received,
                  size_t const new_size_received,
                  size_t const old_size_received,
                  struct dim3 const &ref_dimension_grid_set_received,
                  struct dim3 const &ref_dimension_block_set_received,
                  struct dim3 const &ref_dimension_grid_cpy_received,
                  struct dim3 const &ref_dimension_block_cpy_received);

    template<class T, bool CPY = true> __host__ __device__
    T* Reallocate_Objects(T *ptr_array_received,
                          size_t const new_size_received,
                          size_t const old_size_received);

    // TODO: WARNING dimension set/cpy are inverted.
    template<class T, bool CPY = true> __device__
    T* Reallocate_Objects(T *ptr_array_received,
                          size_t const new_size_received,
                          size_t const old_size_received,
                          struct dim3 const &ref_dimension_grid_set_received,
                          struct dim3 const &ref_dimension_block_set_received,
                          struct dim3 const &ref_dimension_grid_cpy_received,
                          struct dim3 const &ref_dimension_block_cpy_received);

    template<class T, bool CPY = true, bool SET = true> __host__ __device__
    T* Reallocate_PtOfPt(T *ptr_array_received,
                         size_t const new_size_received,
                         size_t const old_size_received);

    // TODO: WARNING dimension set/cpy are inverted.
    template<class T, bool CPY = true, bool SET = true> __device__
    T* Reallocate_PtOfPt(T *ptr_array_received,
                         size_t const new_size_received,
                         size_t const old_size_received,
                         struct dim3 const &ref_dimension_grid_set_received,
                         struct dim3 const &ref_dimension_block_set_received,
                         struct dim3 const &ref_dimension_grid_cpy_received,
                         struct dim3 const &ref_dimension_block_cpy_received);
}

// This.
#include <../src/Reallocate/CUDA/Reallocate_Cpp.cu>
