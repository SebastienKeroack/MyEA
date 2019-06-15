#include <Tools/CUDA_Configuration.cuh>
#include <Math/CUDA_Mathematic.cuh>

#include <CUDA/CUDA_Shuffle.cuh>

#include <curand_kernel.h>

namespace Shuffle
{
    template<typename T>
    __global__ void kernel__Tree_Shift_Shuffle(size_t const half_size_floor_received,
                                                                  size_t const half_size_ceil_received,
                                                                  size_t const index_randomized_received,
                                                                  T *const ptr_array_shuffle_received)
    {
        size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

        T const tmp_swap(ptr_array_shuffle_received[tmp_thread_global_index]);

        ptr_array_shuffle_received[tmp_thread_global_index] = ptr_array_shuffle_received[half_size_floor_received + ((index_randomized_received + tmp_thread_global_index) % half_size_ceil_received)];

        ptr_array_shuffle_received[half_size_floor_received + ((index_randomized_received + tmp_thread_global_index) % half_size_ceil_received)] = tmp_swap;
    }
    
    template<typename T>
    __global__ void kernel__Tree_Shift_Shuffle(size_t const size_received,
                                                                  size_t const half_size_floor_received,
                                                                  size_t const half_size_ceil_received,
                                                                  size_t const index_randomized_received,
                                                                  T *const ptr_array_shuffle_received)
    {
        size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

        if(tmp_thread_global_index < size_received)
        {
            T const tmp_swap(ptr_array_shuffle_received[tmp_thread_global_index]);

            ptr_array_shuffle_received[tmp_thread_global_index] = ptr_array_shuffle_received[half_size_floor_received + ((index_randomized_received + tmp_thread_global_index) % half_size_ceil_received)];

            ptr_array_shuffle_received[half_size_floor_received + ((index_randomized_received + tmp_thread_global_index) % half_size_ceil_received)] = tmp_swap;
        }
    }
    
    template<typename T>
    __global__ void kernel_while__Tree_Shift_Shuffle(size_t const size_received,
                                                                           size_t const half_size_floor_received,
                                                                           size_t const half_size_ceil_received,
                                                                           size_t const index_randomized_received,
                                                                           T *const ptr_array_shuffle_received)
    {
        size_t const tmp_grid_stride(gridDim.x * blockDim.x);
        size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

        T tmp_swap;

        do
        {
            tmp_swap = ptr_array_shuffle_received[tmp_thread_global_index];

            ptr_array_shuffle_received[tmp_thread_global_index] = ptr_array_shuffle_received[half_size_floor_received + ((index_randomized_received + tmp_thread_global_index) % half_size_ceil_received)];

            ptr_array_shuffle_received[half_size_floor_received + ((index_randomized_received + tmp_thread_global_index) % half_size_ceil_received)] = tmp_swap;
            
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    
    template<typename T>
    __global__ void kernel_while__Tree_Shift_Shuffle_Power2(size_t const size_received,
                                                                                        size_t const size_block_received,
                                                                                        size_t const half_size_block_received,
                                                                                        size_t const index_randomized_received,
                                                                                        T *const ptr_array_shuffle_received)
    {
        size_t const tmp_grid_stride(gridDim.x * blockDim.x);
        size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
                          tmp_tree_index(static_cast<size_t>(tmp_thread_global_index / half_size_block_received)),
                          tmp_tree_thread_index(tmp_tree_index * half_size_block_received + tmp_thread_global_index);

        T tmp_swap;

        while(tmp_tree_thread_index < size_received)
        {
            tmp_swap = ptr_array_shuffle_received[tmp_tree_thread_index];

            ptr_array_shuffle_received[tmp_tree_thread_index] = ptr_array_shuffle_received[tmp_tree_index * size_block_received + half_size_block_received + ((index_randomized_received + tmp_tree_thread_index) % half_size_block_received)];

            ptr_array_shuffle_received[tmp_tree_index * size_block_received + half_size_block_received + ((index_randomized_received + tmp_tree_thread_index) % half_size_block_received)] = tmp_swap;
            
            tmp_thread_global_index += tmp_grid_stride;

            tmp_tree_index = static_cast<size_t>(tmp_thread_global_index / half_size_block_received);
            tmp_tree_thread_index = tmp_tree_index * half_size_block_received + tmp_thread_global_index;
        }
    }
    
    template<typename T>
    __device__ inline void Shuffle_Loop(size_t const size_received,
                                                        T *const ptr_array_shuffle_received,
                                                        struct curandStateMtgp32 *const ptr_cuRAND_State_MTGP32_received)
    {
        size_t tmp_randomize_index,
                          i;

        T tmp_swap;

        for(i = size_received; i--;)
        {
            tmp_randomize_index = static_cast<size_t>(curand(ptr_cuRAND_State_MTGP32_received) % (i + 1u));

            // Store the index to swap from the remaining index at "tmp_randomize_index"
            tmp_swap = ptr_array_shuffle_received[tmp_randomize_index];

            // Get remaining index starting at index "i"
            // And store it to the remaining index at "tmp_randomize_index"
            ptr_array_shuffle_received[tmp_randomize_index] = ptr_array_shuffle_received[i];

            // Store the swapped index at the index "i"
            ptr_array_shuffle_received[i] = tmp_swap;
        }
    }
    
    // TODO: Implement a more robust shuffle. This one is poor.
    template<typename T>
    __device__ void Tree_Shift_Shuffle(size_t const size_received,
                                                       size_t const minimum_threads_occupancy_received,
                                                       T *const ptr_array_shuffle_received,
                                                       struct curandStateMtgp32 *const ptr_cuRAND_State_MTGP32_received,
                                                       struct dim3 const *const ptr_dimension_grid_received,
                                                       struct dim3 const *const ptr_dimension_block_received)
    {
        if(size_received > 1u)
        {
            if(USE_PARALLEL && size_received > minimum_threads_occupancy_received)
            {
                size_t tmp_shuffle_block_size(static_cast<size_t>(size_received / 2));

                /*
                PRINT_FORMAT("size_received: %u" NEW_LINE, size_received);
                PRINT_FORMAT("size_half_floor: %u" NEW_LINE, tmp_shuffle_block_size);
                PRINT_FORMAT("size_half_ceil: %u" NEW_LINE, static_cast<size_t>(ceil(static_cast<double>(size_received) / 2.0)));
                PRINT_FORMAT("grid(%u, %u, %u)" NEW_LINE, ptr_dimension_grid_received->x, ptr_dimension_grid_received->y, ptr_dimension_grid_received->z);
                PRINT_FORMAT("block(%u, %u, %u)" NEW_LINE, ptr_dimension_block_received->x, ptr_dimension_block_received->y, ptr_dimension_block_received->z);
                
                PRINT_FORMAT("Before shuffle." NEW_LINE);
                for(size_t i = 0u; i != size_received; ++i)
                { PRINT_FORMAT("Index[%3u](%3u)" NEW_LINE, i, ptr_array_shuffle_received[i]); }
                */

                LAUNCH_KERNEL_POINTER_1D(Tree_Shift_Shuffle<T>,
                                                                  ptr_dimension_grid_received,
                                                                  ptr_dimension_block_received,
                                                                  0_zu,
                                                                  tmp_shuffle_block_size,
                                                                  tmp_shuffle_block_size,
                                                                  static_cast<size_t>(ceil(static_cast<double>(size_received) / 2.0)),
                                                                  curand(ptr_cuRAND_State_MTGP32_received),
                                                                  ptr_array_shuffle_received);

                /*
                PRINT_FORMAT("After shuffle." NEW_LINE);
                for(size_t i = 0u; i != size_received; ++i)
                { PRINT_FORMAT("Index[%3u](%3u)" NEW_LINE, i, ptr_array_shuffle_received[i]); }
                
                CUDA__Check_Error();
                PRINT_FORMAT("Check same..." NEW_LINE);
                for(size_t i = 0u, j; i != size_received; ++i)
                {
                    for(j = i + 1u; j != size_received; ++j)
                    {
                        if(ptr_array_shuffle_received[i] == ptr_array_shuffle_received[j])
                        { PRINT_FORMAT("i[%3u](%3u) == j[%3u](%3u)" NEW_LINE, i, ptr_array_shuffle_received[i], j, ptr_array_shuffle_received[j]); }
                    }
                }

                PRINT_FORMAT("Check present..." NEW_LINE);
                for(size_t i = 0u, j; i != size_received; ++i)
                {
                    for(j = 0u; j != size_received; ++j)
                    {
                        if(i == ptr_array_shuffle_received[j])
                        { break; }

                        if(j + 1u == size_received)
                        { PRINT_FORMAT("i[%3u] Not present!" NEW_LINE, i); }
                    }
                }
                */

                if((tmp_shuffle_block_size = MyEA::Math::Round_Down_At_Power_Of_Two<size_t>(tmp_shuffle_block_size - 1u)) >= minimum_threads_occupancy_received)
                {
                    size_t const tmp_shuffle_block_limit_size(tmp_shuffle_block_size * 2u);
                    
                    do
                    {
                        /*
                        PRINT_FORMAT("tmp_shuffle_block_limit_size: %u" NEW_LINE, tmp_shuffle_block_limit_size);
                        PRINT_FORMAT("tmp_shuffle_block_size * 2u: %u" NEW_LINE, tmp_shuffle_block_size * 2u);
                        PRINT_FORMAT("tmp_shuffle_block_size: %u" NEW_LINE, tmp_shuffle_block_size);
                        PRINT_FORMAT("grid(%u, %u, %u)" NEW_LINE, ptr_dimension_grid_received[1u].x, ptr_dimension_grid_received[1u].y, ptr_dimension_grid_received[1u].z);
                        PRINT_FORMAT("block(%u, %u, %u)" NEW_LINE, ptr_dimension_block_received[1u].x, ptr_dimension_block_received[1u].y, ptr_dimension_block_received[1u].z);
                        
                        PRINT_FORMAT("Before shuffle." NEW_LINE);
                        for(size_t i = 0u; i != tmp_shuffle_block_limit_size; ++i)
                        { PRINT_FORMAT("Index[%3u](%3u)" NEW_LINE, i, ptr_array_shuffle_received[i]); }
                        */

                        kernel_while__Tree_Shift_Shuffle_Power2<T> <<< ptr_dimension_grid_received[1u], ptr_dimension_block_received[1u] >>> (tmp_shuffle_block_limit_size,
                                                                                                                                                                                                            tmp_shuffle_block_size * 2u,
                                                                                                                                                                                                            tmp_shuffle_block_size,
                                                                                                                                                                                                            curand(ptr_cuRAND_State_MTGP32_received),
                                                                                                                                                                                                            ptr_array_shuffle_received);

                        /*
                        PRINT_FORMAT("After shuffle." NEW_LINE);
                        for(size_t i = 0u; i != tmp_shuffle_block_limit_size; ++i)
                        { PRINT_FORMAT("Index[%3u](%3u)" NEW_LINE, i, ptr_array_shuffle_received[i]); }
                        
                        CUDA__Check_Error();
                        PRINT_FORMAT("Check same..." NEW_LINE);
                        for(size_t i = 0u, j; i != tmp_shuffle_block_limit_size; ++i)
                        {
                            for(j = i + 1u; j != tmp_shuffle_block_limit_size; ++j)
                            {
                                if(ptr_array_shuffle_received[i] == ptr_array_shuffle_received[j])
                                { PRINT_FORMAT("i[%3u](%3u) == j[%3u](%3u)" NEW_LINE, i, ptr_array_shuffle_received[i], j, ptr_array_shuffle_received[j]); }
                            }
                        }
                        
                        PRINT_FORMAT("Check present..." NEW_LINE);
                        for(size_t i = 0u, j; i != tmp_shuffle_block_limit_size; ++i)
                        {
                            for(j = 0u; j != tmp_shuffle_block_limit_size; ++j)
                            {
                                if(i == ptr_array_shuffle_received[j])
                                { break; }

                                if(j + 1u == tmp_shuffle_block_limit_size)
                                { PRINT_FORMAT("i[%3u] Not present!" NEW_LINE, i); }
                            }
                        }
                        */
                    }
                    while((tmp_shuffle_block_size = MyEA::Math::Round_Down_At_Power_Of_Two<size_t>(tmp_shuffle_block_size - 1u)) >= minimum_threads_occupancy_received);
                }
            }
            else
            {
                Shuffle_Loop<T>(size_received,
                                         ptr_array_shuffle_received,
                                         ptr_cuRAND_State_MTGP32_received);
            }
        }
        else if(size_received == 0u) { PRINT_FORMAT("%s: ERROR: No array to shuffle!" NEW_LINE, __FUNCTION__); }
    }
    template __device__ void Tree_Shift_Shuffle(size_t const,
                                                                    size_t const,
                                                                    size_t *const,
                                                                    struct curandStateMtgp32 *const,
                                                                    struct dim3 const *const,
                                                                    struct dim3 const *const);
    
    template<typename T>
    __global__ void kernel__Tree_Shuffle(size_t const size_block_received,
                                                          size_t const size_array_received,
                                                          T *const ptr_array_shuffle_received,
                                                          struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
    {
        size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

        T *const tmp_ptr_array_shuffle(ptr_array_shuffle_received + tmp_thread_global_index * size_block_received),
           tmp_swap;
        
        for(size_t tmp_randomize_index,
                               i(size_block_received); i--;)
        {
            tmp_randomize_index = static_cast<size_t>(curand(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x) % (i + 1u));
            
            if(tmp_thread_global_index * size_block_received + i < size_array_received)
            {
                // Store the index to swap from the remaining index at "tmp_randomize_index"
                tmp_swap = tmp_ptr_array_shuffle[tmp_randomize_index];

                // Get remaining index starting at index "i"
                // And store it to the remaining index at "tmp_randomize_index"
                tmp_ptr_array_shuffle[tmp_randomize_index] = tmp_ptr_array_shuffle[i];

                // Store the swapped index at the index "i"
                tmp_ptr_array_shuffle[i] = tmp_swap;
            }
        }
    }
    
    template<typename T>
    __global__ void kernel__Tree_Shuffle(size_t const size_received,
                                                          size_t const size_block_received,
                                                          size_t const size_array_received,
                                                          T *const ptr_array_shuffle_received,
                                                          struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
    {
        size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
        
        T *const tmp_ptr_array_shuffle(ptr_array_shuffle_received + tmp_thread_global_index * size_block_received),
           tmp_swap;
        
        for(size_t tmp_randomize_index,
                               i(size_block_received); i--;)
        {
            tmp_randomize_index = static_cast<size_t>(curand(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x) % (i + 1u));
            
            if(tmp_thread_global_index * size_block_received + i < size_array_received)
            {
                // Store the index to swap from the remaining index at "tmp_randomize_index"
                tmp_swap = tmp_ptr_array_shuffle[tmp_randomize_index];

                // Get remaining index starting at index "i"
                // And store it to the remaining index at "tmp_randomize_index"
                tmp_ptr_array_shuffle[tmp_randomize_index] = tmp_ptr_array_shuffle[i];

                // Store the swapped index at the index "i"
                tmp_ptr_array_shuffle[i] = tmp_swap;
            }
        }
    }
    
    template<typename T>
    __global__ void kernel_while__Tree_Shuffle(size_t const size_received,
                                                                   size_t const size_block_received,
                                                                   size_t const size_array_received,
                                                                   T *const ptr_array_shuffle_received,
                                                                   struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
    {
        size_t const tmp_grid_stride(gridDim.x * blockDim.x);
        size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
                          tmp_thread_block_index(blockIdx.x * blockDim.x);
        
        T *tmp_ptr_array_shuffle,
           tmp_swap;

        do
        {
            tmp_ptr_array_shuffle = ptr_array_shuffle_received + tmp_thread_global_index * size_block_received;
            
            for(size_t tmp_randomize_index,
                                   i(size_block_received); i--;)
            {
                tmp_randomize_index = static_cast<size_t>(curand(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x) % (i + 1u));
                
                if(tmp_thread_global_index * size_block_received + i < size_array_received)
                {
                    // Store the index to swap from the remaining index at "tmp_randomize_index"
                    tmp_swap = tmp_ptr_array_shuffle[tmp_randomize_index];

                    // Get remaining index starting at index "i"
                    // And store it to the remaining index at "tmp_randomize_index"
                    tmp_ptr_array_shuffle[tmp_randomize_index] = tmp_ptr_array_shuffle[i];

                    // Store the swapped index at the index "i"
                    tmp_ptr_array_shuffle[i] = tmp_swap;
                }
            }

            tmp_thread_global_index += tmp_grid_stride;
            tmp_thread_block_index += tmp_grid_stride;
        } while(tmp_thread_block_index < size_received);
    }
    
    template<typename T>
    __device__ void Tree_Shuffle(size_t const size_received,
                                              size_t const size_block_received,
                                              size_t const size_array_received,
                                              T *const ptr_array_shuffle_received,
                                              struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                              struct dim3 const *const ptr_dimension_grid_received,
                                              struct dim3 const *const ptr_dimension_block_received)
    {
        if(size_array_received > 1u)
        {
            if(USE_PARALLEL && size_array_received >= size_block_received)
            {
                LAUNCH_KERNEL_POINTER_1D(Tree_Shuffle<T>,
                                                                  ptr_dimension_grid_received,
                                                                  ptr_dimension_block_received,
                                                                  0_zu,
                                                                  size_received,
                                                                  size_block_received,
                                                                  size_array_received,
                                                                  ptr_array_shuffle_received,
                                                                  ptr_array_cuRAND_State_MTGP32_received);
            }
            else
            {
                Shuffle_Loop<T>(size_array_received,
                                         ptr_array_shuffle_received,
                                         ptr_array_cuRAND_State_MTGP32_received);
            }
        }
        else if(size_received == 0u) { PRINT_FORMAT("%s: ERROR: No array to shuffle!" NEW_LINE, __FUNCTION__); }
    }
    template __device__ void Tree_Shuffle(size_t const,
                                                            size_t const,
                                                            size_t const,
                                                            size_t *const,
                                                            struct curandStateMtgp32 *const,
                                                            struct dim3 const *const,
                                                            struct dim3 const *const);
}
