// Standard.
#include <iterator>

// This.
#include <Configuration/CUDA/Configuration.cuh>
#include <Tools/CUDA_Fill_1D.cuh>
#include <Tools/CUDA_Fill_Pointers_1D.cuh>
#include <Tools/CUDA_Memory_Copy_1D.cuh>
#include <Tools/CUDA_Zero_1D.cuh>

namespace MyEA::Memory::Cpp
{
    template<class T> __host__ __device__
    void Fill_Nullptr(T *ptr_array_received, T const *const ptr_array_last_received)
    {
        while(ptr_array_received != ptr_array_last_received)
        {
            *ptr_array_received++ = nullptr;
        }
    }

    template<class T, bool CPY, bool SET> __host__ __device__
    T* Reallocate(T *ptr_array_received,
                  size_t const new_size_received,
                  size_t const old_size_received)
    {
        if(ptr_array_received == nullptr) { return(nullptr); }
        else if(new_size_received == old_size_received) { return(ptr_array_received); }

        T *tmp_ptr_array_T;

        if(CPY && old_size_received != 0_zu)
        {
            tmp_ptr_array_T = new T[new_size_received];

            if(old_size_received < new_size_received)
            {
                if(SET)
                {
                    MyEA::Memory::Fill<T, defined(__CUDA_ARCH__)>(tmp_ptr_array_T + old_size_received,
                                          tmp_ptr_array_T + new_size_received,
                                          T(0));
                }

                MyEA::Memory::Copy<T, defined(__CUDA_ARCH__)>(ptr_array_received,
                                      ptr_array_received + old_size_received,
                                      tmp_ptr_array_T);
            }
            else
            {
                MyEA::Memory::Copy<T, defined(__CUDA_ARCH__)>(ptr_array_received,
                                      ptr_array_received + new_size_received,
                                      tmp_ptr_array_T);
            }

            delete[](ptr_array_received);
            ptr_array_received = nullptr;
        }
        else
        {
            delete[](ptr_array_received);
            ptr_array_received = nullptr;

            tmp_ptr_array_T = new T[new_size_received];

            if(SET)
            {
                MyEA::Memory::Fill<T, defined(__CUDA_ARCH__)>(tmp_ptr_array_T,
                                      tmp_ptr_array_T + new_size_received,
                                      T(0));
            }
        }

        return(tmp_ptr_array_T);
    }

    template<class T, bool CPY, bool SET> __device__
    T* Reallocate(T *ptr_array_received,
                  size_t const new_size_received,
                  size_t const old_size_received,
                  struct dim3 const &ref_dimension_grid_set_received,
                  struct dim3 const &ref_dimension_block_set_received,
                  struct dim3 const &ref_dimension_grid_cpy_received,
                  struct dim3 const &ref_dimension_block_cpy_received)
    {
        if(ptr_array_received == nullptr) { return(nullptr); }
        else if(new_size_received == old_size_received) { return(ptr_array_received); }

        T *tmp_ptr_array_T;

        if(CPY && old_size_received != 0_zu)
        {
            tmp_ptr_array_T = new T[new_size_received];

            if(old_size_received < new_size_received)
            {
                if(SET)
                {
                    Zero_1D<T>(new_size_received - old_size_received,
                               tmp_ptr_array_T + old_size_received,
                               ptr_dimension_grid_zero_received,
                               ptr_dimension_block_zero_received);
                }

                MyEA::Memory::Memory_Copy_1D<T>(old_size_received,
                                                tmp_ptr_array_T,
                                                ptr_array_received,
                                                ptr_dimension_grid_copy_received,
                                                ptr_dimension_block_copy_received);

                // Do we need to synchronise? Based on "Memory_Copy_1D" Function.
                // => Synchronisation before deleting the old array.
                if(old_size_received >= warpSize * warpSize) { cudaDeviceSynchronize(); }
            }
            else
            {
                MyEA::Memory::Memory_Copy_1D<T>(new_size_received,
                                                tmp_ptr_array_T,
                                                ptr_array_received,
                                                ptr_dimension_grid_copy_received,
                                                ptr_dimension_block_copy_received);

                // Do we need to synchronise? Based on "Memory_Copy_1D" Function.
                // => Synchronisation before deleting the old array.
                if(new_size_received >= warpSize * warpSize) { cudaDeviceSynchronize(); }
            }

            delete[](ptr_array_received);
            ptr_array_received = nullptr;
        }
        else
        {
            delete[](ptr_array_received);
            ptr_array_received = nullptr;

            tmp_ptr_array_T = new T[new_size_received];

            if(SET)
            {
                Zero_1D<T>(new_size_received,
                           tmp_ptr_array_T,
                           ptr_dimension_grid_zero_received,
                           ptr_dimension_block_zero_received);

                // Do we need to synchronise? Based on "Zero_1D" Function.
                // => Synchronisation before deleting the old array.
                if(new_size_received >= warpSize * warpSize) { cudaDeviceSynchronize(); }
            }
        }

        return(tmp_ptr_array_T);
    }

    template<class T, bool CPY> __host__ __device__
    T* Reallocate_Objects(T *ptr_array_received,
                          size_t const new_size_received,
                          size_t const old_size_received)
    {
        if(ptr_array_received == nullptr) { return(nullptr); }
        else if(new_size_received == old_size_received) { return(ptr_array_received); }

        T *tmp_ptr_array_T;

        if(CPY && old_size_received != 0_zu)
        {
            tmp_ptr_array_T = new T[new_size_received];

            if(old_size_received < new_size_received)
            {
                MyEA::Memory::Copy<T, defined(__CUDA_ARCH__)>(ptr_array_received,
                                      ptr_array_received + old_size_received,
                                      tmp_ptr_array_T);
            }
            else
            {
                MyEA::Memory::Copy<T, defined(__CUDA_ARCH__)>(ptr_array_received,
                                      ptr_array_received + new_size_received,
                                      tmp_ptr_array_T);
            }

            delete[](ptr_array_received);
            ptr_array_received = nullptr;
        }
        else
        {
            delete[](ptr_array_received);
            ptr_array_received = nullptr;

            tmp_ptr_array_T = new T[new_size_received];
        }

        return(tmp_ptr_array_T);
    }

    template<class T, bool CPY> __device__
    T* Reallocate_Objects(T *ptr_array_received,
                          size_t const new_size_received,
                          size_t const old_size_received,
                          struct dim3 const &ref_dimension_grid_set_received,
                          struct dim3 const &ref_dimension_block_set_received,
                          struct dim3 const &ref_dimension_grid_cpy_received,
                          struct dim3 const &ref_dimension_block_cpy_received)
    {
        if(ptr_array_received == nullptr) { return(nullptr); }
        else if(new_size_received == old_size_received) { return(ptr_array_received); }

        T *tmp_ptr_array_T;

        if(CPY && old_size_received != 0_zu)
        {
            tmp_ptr_array_T = new T[new_size_received];

            if(old_size_received < new_size_received)
            {
                MyEA::Memory::Memory_Copy_1D<T>(old_size_received,
                                                tmp_ptr_array_T,
                                                ptr_array_received,
                                                ptr_dimension_grid_copy_received,
                                                ptr_dimension_block_copy_received);

                // Do we need to synchronise? Based on "Memory_Copy_1D" Function.
                // => Synchronisation before deleting the old array.
                if(old_size_received >= warpSize * warpSize) { cudaDeviceSynchronize(); }
            }
            else
            {
                MyEA::Memory::Memory_Copy_1D<T>(new_size_received,
                                                tmp_ptr_array_T,
                                                ptr_array_received,
                                                ptr_dimension_grid_copy_received,
                                                ptr_dimension_block_copy_received);

                // Do we need to synchronise? Based on "Memory_Copy_1D" Function.
                // => Synchronisation before deleting the old array.
                if(new_size_received >= warpSize * warpSize) { cudaDeviceSynchronize(); }
            }

            delete[](ptr_array_received);
            ptr_array_received = nullptr;
        }
        else
        {
            delete[](ptr_array_received);
            ptr_array_received = nullptr;

            tmp_ptr_array_T = new T[new_size_received];
        }

        return(tmp_ptr_array_T);
    }

    template<class T, bool CPY, bool SET> __host__ __device__
    T* Reallocate_PtOfPt(T *ptr_array_received,
                         size_t const new_size_received,
                         size_t const old_size_received)
    {
        if(ptr_array_received == nullptr) { return(nullptr); }
        else if(new_size_received == old_size_received) { return(ptr_array_received); }

        T *tmp_ptr_array_T;

        if(CPY && old_size_received != 0_zu)
        {
            tmp_ptr_array_T = new T[new_size_received];

            if(old_size_received < new_size_received)
            {
                if(SET) { Fill_Nullptr<T>(tmp_ptr_array_T + old_size_received, tmp_ptr_array_T + new_size_received); }

                // TODO: Check if std is compatible.
                MyEA::Memory::Copy<T, false>(ptr_array_received,
                                             ptr_array_received + old_size_received,
                                             tmp_ptr_array_T);
            }
            else
            {
                // TODO: Check if std is compatible.
                MyEA::Memory::Copy<T, false>(ptr_array_received,
                                             ptr_array_received + new_size_received,
                                             tmp_ptr_array_T);
            }

            delete[](ptr_array_received);
            ptr_array_received = nullptr;
        }
        else
        {
            delete[](ptr_array_received);
            ptr_array_received = nullptr;

            tmp_ptr_array_T = new T[new_size_received];

            if(SET) { Fill_Nullptr<T>(tmp_ptr_array_T, tmp_ptr_array_T + new_size_received); }
        }

        return(tmp_ptr_array_T);
    }

    template<class T, bool CPY, bool SET> __device__
    T* Reallocate_PtOfPt(T *ptr_array_received,
                         size_t const new_size_received,
                         size_t const old_size_received,
                         struct dim3 const &ref_dimension_grid_set_received,
                         struct dim3 const &ref_dimension_block_set_received,
                         struct dim3 const &ref_dimension_grid_cpy_received,
                         struct dim3 const &ref_dimension_block_cpy_received)
    {
        if(ptr_array_received == nullptr) { return(nullptr); }
        else if(new_size_received == old_size_received) { return(ptr_array_received); }

        T *tmp_ptr_array_T;

        if(CPY && old_size_received != 0_zu)
        {
            tmp_ptr_array_T = new T[new_size_received];

            if(old_size_received < new_size_received)
            {
                if(SET)
                {
                    Fill_Pointers_1D<T>(new_size_received - old_size_received,
                                        tmp_ptr_array_T + old_size_received,
                                        ptr_dimension_grid_fill_received,
                                        ptr_dimension_block_fill_received);
                }

                MyEA::Memory::Memory_Copy_1D<T>(old_size_received,
                                                tmp_ptr_array_T,
                                                ptr_array_received,
                                                ptr_dimension_grid_copy_received,
                                                ptr_dimension_block_copy_received);

                // Do we need to synchronise? Based on "Memory_Copy_1D" Function.
                // => Synchronisation before deleting the old array.
                if(old_size_received >= warpSize * warpSize) { cudaDeviceSynchronize(); }
            }
            else
            {
                MyEA::Memory::Memory_Copy_1D<T>(new_size_received,
                                                tmp_ptr_array_T,
                                                ptr_array_received,
                                                ptr_dimension_grid_copy_received,
                                                ptr_dimension_block_copy_received);

                // Do we need to synchronise? Based on "Memory_Copy_1D" Function.
                // => Synchronisation before deleting the old array.
                if(new_size_received >= warpSize * warpSize) { cudaDeviceSynchronize(); }
            }

            delete[](ptr_array_received);
            ptr_array_received = nullptr;
        }
        else
        {
            delete[](ptr_array_received);
            ptr_array_received = nullptr;

            tmp_ptr_array_T = new T[new_size_received];

            if(SET)
            {
                Fill_Pointers_1D<T>(new_size_received,
                                    tmp_ptr_array_T,
                                    ptr_dimension_grid_fill_received,
                                    ptr_dimension_block_fill_received);

                // Do we need to synchronise? Based on "Zero_1D" Function.
                // => Synchronisation before deleting the old array.
                if(new_size_received >= warpSize * warpSize) { cudaDeviceSynchronize(); }
            }
        }

        return(tmp_ptr_array_T);
    }
}
