#include <Tools/CUDA_Configuration.cuh>
#include <Tools/CUDA_Fill_1D.cuh>
#include <Tools/CUDA_Reallocate.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

__device__ bool CUDA_Neural_Network::Reallocate__Thread(size_t const number_threads_received)
{
    if(this->Reallocate__Thread__Cost(number_threads_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Reallocate__Thread__Cost\"" NEW_LINE,
                                __FUNCTION__);

        return(false);
    }
    else if(this->Reallocate_Reduce_Threads(number_threads_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Reallocate_Reduce_Threads\"" NEW_LINE,
                                __FUNCTION__);

        return(false);
    }
    else if(this->Reallocate__Thread__Parameter(number_threads_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Reallocate__Thread__Parameter\"" NEW_LINE,
                                __FUNCTION__);

        return(false);
    }

    this->Prepare__Threads__Grids_Blocks_Dimensions(number_threads_received);

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate__Batch(size_t const batch_size_received)
{
    if(this->Reallocate__Batch__Neuron_Unit(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Reallocate__Batch__Neuron_Unit\"" NEW_LINE,
                                __FUNCTION__);

        return(false);
    }
    else if(this->Reallocate__Batch__Neuron_Reduce_Summation(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Reallocate__Batch__Neuron_Reduce_Summation\"" NEW_LINE,
                                __FUNCTION__);

        return(false);
    }
    else if(this->Reallocate__Batch__Neuron_Reduce_Error(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Reallocate__Batch__Neuron_Reduce_Error\"" NEW_LINE,
                                __FUNCTION__);

        return(false);
    }
    else if(this->Reallocate__Normalized_Unit__Batch_Normalization(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Reallocate__Normalized_Unit__Batch_Normalization\"" NEW_LINE,
                                __FUNCTION__);

        return(false);
    }
    else if(this->Reallocate__Batch__Neuron_Batch_Normalization_Transpose(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Reallocate__Normalized_Unit__Batch_Normalization\"" NEW_LINE,
                                __FUNCTION__);

        return(false);
    }
    else if(this->Reallocate__Batch__Neuron_Batch_Normalization_Reduce(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Reallocate__Batch__Neuron_Batch_Normalization_Reduce\"" NEW_LINE,
                                __FUNCTION__);

        return(false);
    }

    this->Prepare__Batch__Grids_Blocks_Dimensions(batch_size_received);

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate__Thread__Cost(size_t const number_threads_received)
{
    // Reallocate loss values.
    T_ *tmp_ptr_array_loss_values(Memory::reallocate_cpp<T_>(this->ptr_array_loss_values,
                                                                                             number_threads_received,
                                                                                             this->number_threads,
                                                                                             false));
    if(tmp_ptr_array_loss_values == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 sizeof(T_),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_loss_values = tmp_ptr_array_loss_values;
    // |END| Reallocate loss values. |END|
    
    // Reallocate number loss.
    size_t *tmp_ptr_array_number_loss(Memory::reallocate_cpp<size_t>(this->ptr_array_number_loss,
                                                                                                         number_threads_received,
                                                                                                         this->number_threads,
                                                                                                         false));
    if(tmp_ptr_array_number_loss == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 sizeof(size_t),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_number_loss = tmp_ptr_array_number_loss;
    // |END| Reallocate number loss. |END|
    
    // Reallocate number loss.
    size_t *tmp_ptr_array_bit_fail_values(Memory::reallocate_cpp<size_t>(this->ptr_array_number_bit_fail,
                                                                                                          number_threads_received,
                                                                                                          this->number_threads,
                                                                                                          false));
    if(tmp_ptr_array_bit_fail_values == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 sizeof(size_t),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_number_bit_fail = tmp_ptr_array_bit_fail_values;
    // |END| Reallocate number loss. |END|
    
    // Reallocate number accuracy value.
    T_ *tmp_ptr_array_number_accuracy_value(Memory::reallocate_cpp<T_>(this->ptr_array_accuracy_values[0u],
                                                                                                               number_threads_received,
                                                                                                               this->number_threads,
                                                                                                               false));
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 sizeof(T_),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_accuracy_values[0u] = tmp_ptr_array_number_accuracy_value;

    tmp_ptr_array_number_accuracy_value = Memory::reallocate_cpp<T_>(this->ptr_array_accuracy_values[1u],
                                                                                                            number_threads_received,
                                                                                                            this->number_threads,
                                                                                                            false);
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 sizeof(T_),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_accuracy_values[1u] = tmp_ptr_array_number_accuracy_value;

    tmp_ptr_array_number_accuracy_value = Memory::reallocate_cpp<T_>(this->ptr_array_accuracy_values[2u],
                                                                                                            number_threads_received,
                                                                                                            this->number_threads,
                                                                                                            false);
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 sizeof(T_),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_accuracy_values[2u] = tmp_ptr_array_number_accuracy_value;

    tmp_ptr_array_number_accuracy_value = Memory::reallocate_cpp<T_>(this->ptr_array_accuracy_values[3u],
                                                                                                            number_threads_received,
                                                                                                            this->number_threads,
                                                                                                            false);
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 sizeof(T_),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_accuracy_values[3u] = tmp_ptr_array_number_accuracy_value;

    tmp_ptr_array_number_accuracy_value = Memory::reallocate_cpp<T_>(this->ptr_array_accuracy_values[4u],
                                                                                                            number_threads_received,
                                                                                                            this->number_threads,
                                                                                                            false);
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 sizeof(T_),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return(false);
    }
    this->ptr_array_accuracy_values[4u] = tmp_ptr_array_number_accuracy_value;
    // |END| Reallocate number accuracy value. |END|
    
    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate_Reduce_Threads(size_t const number_threads_received)
{
    if(this->total_reduce_batch_size != 0u)
    {
        size_t tmp_total_elements_to_reduce;

        class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;

        // Compute dimension reduce data batch.
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = number_threads_received;
        
        // Dimension required to reduce the number of elements.
        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                              0u,
                                                                                              tmp_dim3_grid,
                                                                                              tmp_dim3_block);
        
        // Get remaining elements to reduce.
        tmp_total_elements_to_reduce = tmp_dim3_grid.x;

        if(tmp_total_elements_to_reduce == 0u)
        {
            PRINT_FORMAT("%s: ERROR: No elements to reduce." NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        // |END| Compute dimension reduce data batch. |END|

        if(this->Reallocate_Reduce_Cost(tmp_total_elements_to_reduce) == false)
        {
            PRINT_FORMAT("%s: ERROR: From \"Reallocate_Reduce_Cost\"" NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        else if(this->Reallocate_Reduce_Threads_Dim(number_threads_received) == false)
        {
            PRINT_FORMAT("%s: ERROR: From \"Reallocate_Reduce_Threads_Dim\"" NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }

        this->total_reduce_batch_size = tmp_total_elements_to_reduce;
        
        // Compute dimension reduce data batch dynamic parallelisme.
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = number_threads_received;
        
        // Dimension required to reduce the number of elements.
        tmp_ptr_CUDA_Device->Grid_Block_Reduce_Dynamic_Parallelisme(tmp_total_elements_to_reduce,
                                                                                                            0u,
                                                                                                            tmp_dim3_grid,
                                                                                                            tmp_dim3_block);
        
        // Get remaining elements to reduce.
        tmp_total_elements_to_reduce = tmp_dim3_grid.x;

        if(tmp_total_elements_to_reduce == 0u)
        {
            PRINT_FORMAT("%s: ERROR: No elements to reduce." NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        // |END| Compute dimension reduce data batch dynamic parallelisme. |END|

        if(this->Reallocate_Reduce_Threads_Dim_DP(number_threads_received) == false)
        {
            PRINT_FORMAT("%s: ERROR: From \"Reallocate_Reduce_Threads_Dim_DP\"" NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }

        this->total_reduce_batch_DP_size = tmp_total_elements_to_reduce;
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate_Reduce_Threads_Dim(size_t const number_threads_received)
{
    size_t tmp_total_elements_to_reduce,
                      tmp_index_dim3(0u);
    
    if(this->ptr_array_dim3_grid_reduce_threads != nullptr && number_threads_received != 0u)
    {
        class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;

        // Compute dimension reduce data batch.
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = number_threads_received;
        
        // Dimension required to reduce the number of elements.
        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                0u,
                                                                                                tmp_dim3_grid,
                                                                                                tmp_dim3_block);
        
        // Get remaining elements to reduce.
        tmp_total_elements_to_reduce = tmp_dim3_grid.x;

        if(tmp_total_elements_to_reduce == 0u)
        {
            PRINT_FORMAT("%s: ERROR: No elements to reduce." NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        // |END| Compute dimension reduce data batch. |END|
        
        // Allocating neurons reduce summation dim3 grid.
        struct dim3 *tmp_ptr_array_dim3_grid_reduce_threads(Memory::reallocate<struct dim3>(this->ptr_array_dim3_grid_reduce_threads,
                                                                                                                                      tmp_total_elements_to_reduce * sizeof(struct dim3),
                                                                                                                                      this->total_reduce_batch_size * sizeof(struct dim3),
                                                                                                                                      false));
        if(tmp_ptr_array_dim3_grid_reduce_threads == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        this->ptr_array_dim3_grid_reduce_threads = tmp_ptr_array_dim3_grid_reduce_threads;
        // |END| Allocating neurons reduce summation dim3 grid. |END|
            
        // Allocating neurons reduce summation dim3 block.
        struct dim3 *tmp_ptr_array_dim3_block_reduce_threads(Memory::reallocate<struct dim3>(this->ptr_array_dim3_block_reduce_threads,
                                                                                                                                        tmp_total_elements_to_reduce * sizeof(struct dim3),
                                                                                                                                        this->total_reduce_batch_size * sizeof(struct dim3),
                                                                                                                                        false));
        if(tmp_ptr_array_dim3_block_reduce_threads == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        this->ptr_array_dim3_block_reduce_threads = tmp_ptr_array_dim3_block_reduce_threads;
        // |END| Allocating neurons reduce summation dim3 block. |END|
            
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = number_threads_received;

        // Loop to reduce "number of elements" to one at the end.
        do
        {
            // Compute remaining results to reduce.
            tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                    0u,
                                                                                                    tmp_ptr_array_dim3_grid_reduce_threads[tmp_index_dim3],
                                                                                                    tmp_ptr_array_dim3_block_reduce_threads[tmp_index_dim3]);

            // Get the remaining results to reduce.
            tmp_total_elements_to_reduce = tmp_ptr_array_dim3_grid_reduce_threads[tmp_index_dim3].x;

            // Increment index to dim3.
            ++tmp_index_dim3;
        } while(tmp_total_elements_to_reduce != 1u);
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate_Reduce_Threads_Dim_DP(size_t const number_threads_received)
{
    size_t tmp_total_elements_to_reduce,
                      tmp_index_dim3(0u);
    
    if(this->ptr_array_dim3_grid_reduce_threads_DP != nullptr && number_threads_received != 0u)
    {
        class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;

        // Compute dimension reduce data batch.
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = number_threads_received;
        
        // Dimension required to reduce the number of elements.
        tmp_ptr_CUDA_Device->Grid_Block_Reduce_Dynamic_Parallelisme(tmp_total_elements_to_reduce,
                                                                                                            tmp_ptr_CUDA_Device->Get__Maximum_Blocks_Per_Multiprocessor(),
                                                                                                            tmp_dim3_grid,
                                                                                                            tmp_dim3_block);
        
        // Get remaining elements to reduce.
        tmp_total_elements_to_reduce = tmp_dim3_grid.x;

        if(tmp_total_elements_to_reduce == 0u)
        {
            PRINT_FORMAT("%s: ERROR: No elements to reduce." NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        // |END| Compute dimension reduce data batch. |END|
        
        // Allocating neurons reduce summation dim3 grid.
        struct dim3 *tmp_ptr_array_dim3_grid_threads_DP(Memory::reallocate<struct dim3>(this->ptr_array_dim3_grid_reduce_threads_DP,
                                                                                                                                   tmp_total_elements_to_reduce * sizeof(struct dim3),
                                                                                                                                   this->total_reduce_batch_DP_size * sizeof(struct dim3),
                                                                                                                                   false));
        if(tmp_ptr_array_dim3_grid_threads_DP == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        this->ptr_array_dim3_grid_reduce_threads_DP = tmp_ptr_array_dim3_grid_threads_DP;
        // |END| Allocating neurons reduce summation dim3 grid. |END|
            
        // Allocating neurons reduce summation dim3 block.
        struct dim3 *tmp_ptr_array_dim3_block_threads_DP(Memory::reallocate<struct dim3>(this->ptr_array_dim3_block_reduce_threads_DP,
                                                                                                                                      tmp_total_elements_to_reduce * sizeof(struct dim3),
                                                                                                                                      this->total_reduce_batch_DP_size * sizeof(struct dim3),
                                                                                                                                      false));
        if(tmp_ptr_array_dim3_block_threads_DP == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        this->ptr_array_dim3_block_reduce_threads_DP = tmp_ptr_array_dim3_block_threads_DP;
        // |END| Allocating neurons reduce summation dim3 block. |END|
            
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = number_threads_received;

        // Loop to reduce "number of elements" to one at the end.
        do
        {
            // Compute remaining results to reduce.
            tmp_ptr_CUDA_Device->Grid_Block_Reduce_Dynamic_Parallelisme(tmp_total_elements_to_reduce,
                                                                                                                tmp_ptr_CUDA_Device->Get__Maximum_Blocks_Per_Multiprocessor(),
                                                                                                                tmp_ptr_array_dim3_grid_threads_DP[tmp_index_dim3],
                                                                                                                tmp_ptr_array_dim3_block_threads_DP[tmp_index_dim3]);

            // Get the remaining results to reduce.
            tmp_total_elements_to_reduce = tmp_ptr_array_dim3_grid_threads_DP[tmp_index_dim3].x;

            // Increment index to dim3.
            ++tmp_index_dim3;
        } while(tmp_total_elements_to_reduce != 1u);
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate_Reduce_Cost(size_t const total_reduce_batch_size_received)
{
    if(this->ptr_array_reduce_number_loss != nullptr && total_reduce_batch_size_received != 0u)
    {
        // Allocating reduce number loss.
        size_t *tmp_ptr_array_reduce_number_loss(Memory::reallocate_cpp<size_t>(this->ptr_array_reduce_number_loss,
                                                                                                                        total_reduce_batch_size_received,
                                                                                                                        this->total_reduce_batch_size,
                                                                                                                        false));
        if(tmp_ptr_array_reduce_number_loss == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     sizeof(size_t),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_reduce_number_loss = tmp_ptr_array_reduce_number_loss;
        // |END| Allocating reduce number loss. |END|
        
        // Allocating reduce bit fail values.
        size_t *tmp_ptr_array_reduce_bit_fail_values(Memory::reallocate_cpp<size_t>(this->ptr_array_reduce_bit_fail_values,
                                                                                                                          total_reduce_batch_size_received,
                                                                                                                          this->total_reduce_batch_size,
                                                                                                                          false));
        if(tmp_ptr_array_reduce_bit_fail_values == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     sizeof(size_t),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_reduce_bit_fail_values = tmp_ptr_array_reduce_bit_fail_values;
        // |END| Allocating reduce bit fail values. |END|
        
        // Allocating reduce accuracy values.
        T_ *tmp_ptr_array_reduce_accuracy_values(Memory::reallocate_cpp<T_>(this->ptr_array_reduce_accuracy_values[0u],
                                                                                                                   total_reduce_batch_size_received,
                                                                                                                   this->total_reduce_batch_size,
                                                                                                                   false));
        if(tmp_ptr_array_reduce_accuracy_values == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     sizeof(T_),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_reduce_accuracy_values[0u] = tmp_ptr_array_reduce_accuracy_values;

        tmp_ptr_array_reduce_accuracy_values = Memory::reallocate_cpp<T_>(this->ptr_array_reduce_accuracy_values[1u],
                                                                                                                 total_reduce_batch_size_received,
                                                                                                                 this->total_reduce_batch_size,
                                                                                                                 false);
        if(tmp_ptr_array_reduce_accuracy_values == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     sizeof(T_),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_reduce_accuracy_values[1u] = tmp_ptr_array_reduce_accuracy_values;

        tmp_ptr_array_reduce_accuracy_values = Memory::reallocate_cpp<T_>(this->ptr_array_reduce_accuracy_values[2u],
                                                                                                                 total_reduce_batch_size_received,
                                                                                                                 this->total_reduce_batch_size,
                                                                                                                 false);
        if(tmp_ptr_array_reduce_accuracy_values == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     sizeof(T_),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_reduce_accuracy_values[2u] = tmp_ptr_array_reduce_accuracy_values;

        tmp_ptr_array_reduce_accuracy_values = Memory::reallocate_cpp<T_>(this->ptr_array_reduce_accuracy_values[3u],
                                                                                                                 total_reduce_batch_size_received,
                                                                                                                 this->total_reduce_batch_size,
                                                                                                                 false);
        if(tmp_ptr_array_reduce_accuracy_values == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     sizeof(T_),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_reduce_accuracy_values[3u] = tmp_ptr_array_reduce_accuracy_values;

        tmp_ptr_array_reduce_accuracy_values = Memory::reallocate_cpp<T_>(this->ptr_array_reduce_accuracy_values[4u],
                                                                                                                 total_reduce_batch_size_received,
                                                                                                                 this->total_reduce_batch_size,
                                                                                                                 false);
        if(tmp_ptr_array_reduce_accuracy_values == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     sizeof(T_),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_reduce_accuracy_values[4u] = tmp_ptr_array_reduce_accuracy_values;
        // |END| Allocating reduce accuracy values.. |END|
        
        // Allocating reduce loss values.
        T_ *tmp_ptr_array_reduce_loss_values(Memory::reallocate_cpp<T_>(this->ptr_array_reduce_loss_values,
                                                                                                            total_reduce_batch_size_received,
                                                                                                            this->total_reduce_batch_size,
                                                                                                            false));
        if(tmp_ptr_array_reduce_loss_values == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     sizeof(T_),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return(false);
        }
        this->ptr_array_reduce_loss_values = tmp_ptr_array_reduce_loss_values;
        // |END| Allocating reduce loss values.. |END|
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate__Batch__Neuron_Unit(size_t const batch_size_received)
{
    if(this->total_neuron_units_allocated != 0u)
    {
        size_t tmp_number_neuron_units;

        struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct CUDA_Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct CUDA_Neuron const *tmp_ptr_last_neuron_unit;
        struct CUDA_Neuron *tmp_ptr_neuron_unit_it;

        // Allocating neuron unit(s) summation(s).
        T_ *tmp_ptr_array_neuron_units_summations(Memory::reallocate_cpp<T_>(this->ptr_array_neuron_units_summations,
                                                                                                            batch_size_received * this->total_neuron_units_allocated,
                                                                                                            this->batch_size * this->total_neuron_units_allocated,
                                                                                                            false));
        if(tmp_ptr_array_neuron_units_summations == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        this->ptr_array_neuron_units_summations = tmp_ptr_array_neuron_units_summations;
        // |END| Allocating neuron unit(s) summation(s). |END|
        
        // Allocating neuron unit(s) value(s).
        T_ *tmp_ptr_array_neuron_units_values(Memory::reallocate_cpp<T_>(this->ptr_array_neuron_units_values,
                                                                                                    batch_size_received * this->total_neuron_units_allocated,
                                                                                                    this->batch_size * this->total_neuron_units_allocated,
                                                                                                    false));
        if(tmp_ptr_array_neuron_units_values == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        this->ptr_array_neuron_units_values = tmp_ptr_array_neuron_units_values;
        // |END| Allocating neuron unit(s) value(s). |END|
        
        // Allocating neuron unit(s) error(s).
        T_ *tmp_ptr_array_neuron_units_errors(Memory::reallocate_cpp<T_>(this->ptr_array_neuron_units_errors,
                                                                                                    batch_size_received * this->total_neuron_units_allocated,
                                                                                                    this->batch_size * this->total_neuron_units_allocated,
                                                                                                    false));
        if(tmp_ptr_array_neuron_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        this->ptr_array_neuron_units_errors = tmp_ptr_array_neuron_units_errors;
        // |END| Allocating neuron unit(s) error(s). |END|
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            if((tmp_number_neuron_units = *tmp_ptr_layer_it->ptr_number_neurons) != 0u)
            {
                for(tmp_ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_last_neuron_unit,
                    tmp_ptr_neuron_unit_it = tmp_ptr_layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
                {
                    tmp_ptr_neuron_unit_it->ptr_array_summations = tmp_ptr_array_neuron_units_summations++;
                    tmp_ptr_neuron_unit_it->ptr_array_values = tmp_ptr_array_neuron_units_values++;
                    tmp_ptr_neuron_unit_it->ptr_array_errors = tmp_ptr_array_neuron_units_errors++;
                }

                tmp_ptr_array_neuron_units_summations += (batch_size_received - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_values += (batch_size_received - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_errors += (batch_size_received - 1u) * tmp_number_neuron_units;
            }
        }
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate__Batch__Neuron_Reduce_Summation(size_t const batch_size_received)
{
    if(this->total_neuron_units_allocated != 0u && this->ptr_array_2D_neurons_reduce_summation != nullptr)
    {
        struct CUDA_Neuron *tmp_ptr_neuron_unit_it(this->ptr_array_layers->ptr_array_neuron_units);
        struct CUDA_Neuron const *const tmp_ptr_last_neuron_unit(tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated);
        
        struct dim3 tmp_dim3_grid_zero(1u,1u, 1u),
                         tmp_dim3_block_zero(1u,1u, 1u),
                         tmp_dim3_grid_copy(1u,1u, 1u),
                         tmp_dim3_block_copy(1u,1u, 1u);

        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(batch_size_received * this->neurons_total_reduce_summation_size,
                                                                                                  this->batch_size * this->neurons_total_reduce_summation_size,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        // Allocating neuron unit(s) value(s).
        T_ **tmp_ptr_array_2D_neurons_position_reduce_summation_array(this->ptr_array_2D_neurons_reduce_summation);
        
        T_ *tmp_ptr_array_neuron_units_reduce_summation_results(Memory::reallocate_cpp<T_>(*this->ptr_array_2D_neurons_reduce_summation,
                                                                                                                                    batch_size_received * this->neurons_total_reduce_summation_size,
                                                                                                                                    this->batch_size * this->neurons_total_reduce_summation_size,
                                                                                                                                    &tmp_dim3_grid_zero,
                                                                                                                                    &tmp_dim3_block_zero,
                                                                                                                                    &tmp_dim3_grid_copy,
                                                                                                                                    &tmp_dim3_block_copy,
                                                                                                                                    false));
        if(tmp_ptr_array_neuron_units_reduce_summation_results == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        // |END| Allocating neuron unit(s) value(s). |END|
        
        // Loop through each neurons in the network.
        for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                             ++tmp_ptr_array_2D_neurons_position_reduce_summation_array)
        {
            // Assign the position index of the begining results array from that array.
            *tmp_ptr_array_2D_neurons_position_reduce_summation_array = tmp_ptr_array_neuron_units_reduce_summation_results;

            // Assign the begining results array to that pointer.
            tmp_ptr_neuron_unit_it->ptr_array_reduce_summation = tmp_ptr_array_2D_neurons_position_reduce_summation_array;
            
            // If is not the bias. (The bias have no elements to reduce.)
            if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections != 0u)
            {
                // Increment the begining results by the reduce summation size of that neuron.
                tmp_ptr_array_neuron_units_reduce_summation_results += *tmp_ptr_neuron_unit_it->ptr_reduce_summation_size;
            }
        }
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate__Batch__Neuron_Reduce_Error(size_t const batch_size_received)
{
    if(this->total_neuron_units_allocated != 0u && this->ptr_array_2D_neurons_reduce_error != nullptr)
    {
        struct CUDA_Neuron *tmp_ptr_neuron_unit_it(this->ptr_array_layers->ptr_array_neuron_units);
        struct CUDA_Neuron const *const tmp_ptr_last_neuron_unit(tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated);
        
        struct dim3 tmp_dim3_grid_zero(1u,1u, 1u),
                         tmp_dim3_block_zero(1u,1u, 1u),
                         tmp_dim3_grid_copy(1u,1u, 1u),
                         tmp_dim3_block_copy(1u,1u, 1u);

        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(batch_size_received * this->neurons_total_reduce_error_size,
                                                                                                  this->batch_size * this->neurons_total_reduce_error_size,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        // Allocating neuron unit(s) value(s).
        T_ **tmp_ptr_array_2D_neurons_position_reduce_error_array(this->ptr_array_2D_neurons_reduce_error);
        
        T_ *tmp_ptr_array_neuron_units_reduce_error_results(Memory::reallocate_cpp<T_>(*this->ptr_array_2D_neurons_reduce_error,
                                                                                                                          batch_size_received * this->neurons_total_reduce_error_size,
                                                                                                                          this->batch_size * this->neurons_total_reduce_error_size,
                                                                                                                          &tmp_dim3_grid_zero,
                                                                                                                          &tmp_dim3_block_zero,
                                                                                                                          &tmp_dim3_grid_copy,
                                                                                                                          &tmp_dim3_block_copy,
                                                                                                                          false));
        if(tmp_ptr_array_neuron_units_reduce_error_results == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        // |END| Allocating neuron unit(s) value(s). |END|
        
        // Loop through each neurons in the network.
        for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                             ++tmp_ptr_array_2D_neurons_position_reduce_error_array)
        {
            // Assign the position index of the begining results array from that array.
            *tmp_ptr_array_2D_neurons_position_reduce_error_array = tmp_ptr_array_neuron_units_reduce_error_results;

            // Assign the begining results array to that pointer.
            tmp_ptr_neuron_unit_it->ptr_array_reduce_error = tmp_ptr_array_2D_neurons_position_reduce_error_array;
            
            // Increment the begining results by the reduce error size of that neuron.
            tmp_ptr_array_neuron_units_reduce_error_results += *tmp_ptr_neuron_unit_it->ptr_reduce_error_size;
        }
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate__Normalized_Unit__Batch_Normalization(size_t const batch_size_received)
{
    if(this->use_Batch_Renormalization && this->total_neuron_units_allocated != 0u)
    {
        size_t tmp_number_neuron_units;

        struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct CUDA_Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct CUDA_Neuron const *tmp_ptr_last_neuron_unit;
        struct CUDA_Neuron *tmp_ptr_neuron_unit_it;
        
        struct dim3 tmp_dim3_grid_zero(1u,1u, 1u),
                         tmp_dim3_block_zero(1u,1u, 1u),
                         tmp_dim3_grid_copy(1u,1u, 1u),
                         tmp_dim3_block_copy(1u,1u, 1u);

        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(batch_size_received * this->total_neuron_units_allocated,
                                                                                                  this->batch_size * this->total_neuron_units_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        // Allocating neuron unit(s) value(s) hat.
        T_ *tmp_ptr_array_neuron_units_values_hat(Memory::reallocate_cpp<T_>(this->ptr_array_normalized_batch_units_values_hats,
                                                                                                            batch_size_received * this->total_neuron_units_allocated,
                                                                                                            this->batch_size * this->total_neuron_units_allocated,
                                                                                                            &tmp_dim3_grid_zero,
                                                                                                            &tmp_dim3_block_zero,
                                                                                                            &tmp_dim3_grid_copy,
                                                                                                            &tmp_dim3_block_copy,
                                                                                                            false));
        if(tmp_ptr_array_neuron_units_values_hat == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        // |END| Allocating neuron unit(s) value(s) hat. |END|
            
        // Allocating neuron unit(s) value(s) normalize.
        T_ *tmp_ptr_array_neuron_units_values_normalize(Memory::reallocate_cpp<T_>(this->ptr_array_normalized_batch_units_values_normalizes,
                                                                                                                      batch_size_received * this->total_neuron_units_allocated,
                                                                                                                      this->batch_size * this->total_neuron_units_allocated,
                                                                                                                      &tmp_dim3_grid_zero,
                                                                                                                      &tmp_dim3_block_zero,
                                                                                                                      &tmp_dim3_grid_copy,
                                                                                                                      &tmp_dim3_block_copy,
                                                                                                                      false));
        if(tmp_ptr_array_neuron_units_values_normalize == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        // |END| Allocating neuron unit(s) value(s) normalize. |END|
        
        // Allocating neurons mean.
        T_ *tmp_ptr_array_neuron_units_mean_it(Memory::reallocate_cpp<T_>(this->ptr_array_normalized_batch_units_means,
                                                                                                        batch_size_received * this->total_neuron_units_allocated,
                                                                                                        this->batch_size * this->total_neuron_units_allocated,
                                                                                                        &tmp_dim3_grid_zero,
                                                                                                        &tmp_dim3_block_zero,
                                                                                                        &tmp_dim3_grid_copy,
                                                                                                        &tmp_dim3_block_copy,
                                                                                                        false));
        if(tmp_ptr_array_neuron_units_mean_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        // |END| Allocating neurons mean. |END|
        
        // Allocating neurons variance.
        T_ *tmp_ptr_array_neuron_units_variance_it(Memory::reallocate_cpp<T_>(this->ptr_array_normalized_batch_units_variances,
                                                                                                            batch_size_received * this->total_neuron_units_allocated,
                                                                                                            this->batch_size * this->total_neuron_units_allocated,
                                                                                                            &tmp_dim3_grid_zero,
                                                                                                            &tmp_dim3_block_zero,
                                                                                                            &tmp_dim3_grid_copy,
                                                                                                            &tmp_dim3_block_copy,
                                                                                                            false));
        if(tmp_ptr_array_neuron_units_variance_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        // |END| Allocating neurons variance. |END|
        
        // Allocating neurons derivative mean.
        T_ *tmp_ptr_array_neuron_units_derivative_mean_it(Memory::reallocate_cpp<T_>(this->ptr_array_normalized_batch_units_derivatives_means,
                                                                                                                        batch_size_received * this->total_neuron_units_allocated,
                                                                                                                        this->batch_size * this->total_neuron_units_allocated,
                                                                                                                        &tmp_dim3_grid_zero,
                                                                                                                        &tmp_dim3_block_zero,
                                                                                                                        &tmp_dim3_grid_copy,
                                                                                                                        &tmp_dim3_block_copy,
                                                                                                                        false));
        if(tmp_ptr_array_neuron_units_derivative_mean_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        // |END| Allocating neurons derivative mean. |END|
        
        // Allocating neurons derivative variance.
        T_ *tmp_ptr_array_neuron_units_derivative_variance_it(Memory::reallocate_cpp<T_>(this->ptr_array_normalized_batch_units_derivatives_variances,
                                                                                                                           batch_size_received * this->total_neuron_units_allocated,
                                                                                                                           this->batch_size * this->total_neuron_units_allocated,
                                                                                                                           &tmp_dim3_grid_zero,
                                                                                                                           &tmp_dim3_block_zero,
                                                                                                                           &tmp_dim3_grid_copy,
                                                                                                                           &tmp_dim3_block_copy,
                                                                                                                           false));
        if(tmp_ptr_array_neuron_units_derivative_variance_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        // |END| Allocating neurons derivative variance. |END|
        
        this->ptr_array_normalized_batch_units_values_hats = tmp_ptr_array_neuron_units_values_hat;
        this->ptr_array_normalized_batch_units_values_normalizes = tmp_ptr_array_neuron_units_values_normalize;
        this->ptr_array_normalized_batch_units_means = tmp_ptr_array_neuron_units_mean_it;
        this->ptr_array_normalized_batch_units_variances = tmp_ptr_array_neuron_units_variance_it;
        this->ptr_array_normalized_batch_units_derivatives_means = tmp_ptr_array_neuron_units_derivative_mean_it;
        this->ptr_array_normalized_batch_units_derivatives_variances = tmp_ptr_array_neuron_units_derivative_variance_it;
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            if((tmp_number_neuron_units = *tmp_ptr_layer_it->ptr_number_neurons) != 0u)
            {
                for(tmp_ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_last_neuron_unit,
                    tmp_ptr_neuron_unit_it = tmp_ptr_layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_values_hat,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_values_normalize,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_mean_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_variance_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_derivative_mean_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_derivative_variance_it)
                {
                    tmp_ptr_neuron_unit_it->ptr_array_values_hats = tmp_ptr_array_neuron_units_values_hat;
                    tmp_ptr_neuron_unit_it->ptr_array_values_normalizes = tmp_ptr_array_neuron_units_values_normalize;
                    tmp_ptr_neuron_unit_it->ptr_array_means = tmp_ptr_array_neuron_units_mean_it;
                    tmp_ptr_neuron_unit_it->ptr_array_variances = tmp_ptr_array_neuron_units_variance_it;
                    tmp_ptr_neuron_unit_it->ptr_array_derivatives_means = tmp_ptr_array_neuron_units_derivative_mean_it;
                    tmp_ptr_neuron_unit_it->ptr_array_derivatives_variances = tmp_ptr_array_neuron_units_derivative_variance_it;
                }

                tmp_ptr_array_neuron_units_values_hat += (batch_size_received - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_values_normalize += (batch_size_received - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_mean_it += (batch_size_received - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_variance_it += (batch_size_received - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_derivative_mean_it += (batch_size_received - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_derivative_variance_it += (batch_size_received - 1u) * tmp_number_neuron_units;
            }
        }
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate__Batch__Neuron_Batch_Normalization_Transpose(size_t const batch_size_received)
{
    if(this->use_Batch_Renormalization && this->ptr_array_neuron_units_transposed_mean != nullptr)
    {
        struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct CUDA_Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct CUDA_Neuron const *tmp_ptr_last_neuron_unit;
        struct CUDA_Neuron *tmp_ptr_neuron_unit_it;
        
        struct dim3 tmp_dim3_grid_zero(1u,1u, 1u),
                         tmp_dim3_block_zero(1u,1u, 1u),
                         tmp_dim3_grid_copy(1u,1u, 1u),
                         tmp_dim3_block_copy(1u,1u, 1u);

        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(batch_size_received * this->total_neuron_units_allocated,
                                                                                                  this->batch_size * this->total_neuron_units_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        // Allocating neurons mean.
        T_ *tmp_ptr_array_neuron_units_transposed_mean_it(Memory::reallocate_cpp<T_>(this->ptr_array_neuron_units_transposed_mean,
                                                                                                                          batch_size_received * this->total_neuron_units_allocated,
                                                                                                                          this->batch_size * this->total_neuron_units_allocated,
                                                                                                                          &tmp_dim3_grid_zero,
                                                                                                                          &tmp_dim3_block_zero,
                                                                                                                          &tmp_dim3_grid_copy,
                                                                                                                          &tmp_dim3_block_copy,
                                                                                                                          false));
        if(tmp_ptr_array_neuron_units_transposed_mean_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        // |END| Allocating neurons mean. |END|
        
        // Allocating neurons variance.
        T_ *tmp_ptr_array_neuron_units_transposed_variance_it(Memory::reallocate_cpp<T_>(this->ptr_array_neuron_units_transposed_variance,
                                                                                                                              batch_size_received * this->total_neuron_units_allocated,
                                                                                                                              this->batch_size * this->total_neuron_units_allocated,
                                                                                                                              &tmp_dim3_grid_zero,
                                                                                                                              &tmp_dim3_block_zero,
                                                                                                                              &tmp_dim3_grid_copy,
                                                                                                                              &tmp_dim3_block_copy,
                                                                                                                              false));
        if(tmp_ptr_array_neuron_units_transposed_variance_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        // |END| Allocating neurons variance. |END|
        
        this->ptr_array_neuron_units_transposed_mean = tmp_ptr_array_neuron_units_transposed_mean_it;
        this->ptr_array_neuron_units_transposed_variance = tmp_ptr_array_neuron_units_transposed_variance_it;
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            for(tmp_ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_last_neuron_unit,
                tmp_ptr_neuron_unit_it = tmp_ptr_layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
            {
                tmp_ptr_neuron_unit_it->ptr_array_transposed_mean = tmp_ptr_array_neuron_units_transposed_mean_it;
                tmp_ptr_neuron_unit_it->ptr_array_transposed_variance = tmp_ptr_array_neuron_units_transposed_variance_it;

                tmp_ptr_array_neuron_units_transposed_mean_it += batch_size_received;
                tmp_ptr_array_neuron_units_transposed_variance_it += batch_size_received;
            }
        }
    }

    return(true);
}

// TODO: Make "Reallocate__Batch__Neuron_Batch_Normalization_Reduce" and "Reallocate__Batch__Neuron_Reduce_Batch"
__device__ bool CUDA_Neural_Network::Reallocate__Batch__Neuron_Batch_Normalization_Reduce(size_t const batch_size_received)
{
    size_t tmp_neurons_reduce_batch_size_so_far,
                      tmp_total_elements_to_reduce,
                      tmp_layer_reduce_batch_size,
                      tmp_number_neurons_in_layer,
                      tmp_index_dim3;
    
    if(this->use_Batch_Renormalization
      &&
      this->total_neuron_units_allocated != 0u
      &&
      this->ptr_array_neuron_units_reduce_batch_size != nullptr)
    {
        size_t *tmp_ptr_array_neuron_units_reduce_batch_size(this->ptr_array_neuron_units_reduce_batch_size);

        // ONLY FOR DENSE LAYER.
        // TODO: Make shortcut layer compatible.
        struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct CUDA_Layer *tmp_ptr_layer_it;
        
        struct CUDA_Neuron const *tmp_ptr_last_neuron_unit;
        struct CUDA_Neuron *tmp_ptr_neuron_unit_it;
        
        class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block,
                         tmp_dim3_grid_zero(1u,1u, 1u),
                         tmp_dim3_block_zero(1u,1u, 1u),
                         tmp_dim3_grid_copy(1u,1u, 1u),
                         tmp_dim3_block_copy(1u,1u, 1u);
        
        // COMPUTE REDUCE BATCH SIZE.
        for(tmp_neurons_reduce_batch_size_so_far = 0u,
            tmp_ptr_neuron_unit_it = this->ptr_array_layers->ptr_array_neuron_units,
            tmp_ptr_last_neuron_unit = tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                                      ++tmp_ptr_array_neuron_units_reduce_batch_size)
        {
            // Number elements to reduce equal the size of batch.
            tmp_total_elements_to_reduce = batch_size_received;

            // If the neuron is a bias. Number of elements to reduce equal zero.
            if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections == 0u)
            { tmp_total_elements_to_reduce = 0u; }
            
            // If is not the bias. (The bias have no elements to reduce.)
            if(tmp_total_elements_to_reduce != 0u)
            {
                // Dimension required to reduce the number of elements.
                tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                      0u,
                                                                                                      tmp_dim3_grid,
                                                                                                      tmp_dim3_block);
            
                // Get remaining elements to reduce.
                tmp_total_elements_to_reduce = tmp_dim3_grid.x;
            }

            // Maximum remaining elements to reduce.
            *tmp_ptr_array_neuron_units_reduce_batch_size = tmp_total_elements_to_reduce;

            // Summation of the total maximum number of batch result.
            tmp_neurons_reduce_batch_size_so_far += tmp_total_elements_to_reduce;
        }

        if(tmp_neurons_reduce_batch_size_so_far == 0u)
        {
            PRINT_FORMAT("%s: ERROR: No elements to reduce." NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        // |END| Compute dimension reduce batch. |END|
        // |END| COMPUTE REDUCE BATCH SIZE. |END|
        
        // COMPUTE DIMENSION REDUCE BATCH.
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(tmp_neurons_reduce_batch_size_so_far,
                                                                                                  this->neurons_total_reduce_batch_size,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  tmp_ptr_CUDA_Device,
                                                                                                  false);
        
        // Allocating neurons reduce batch mean.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        T_ **tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array(this->ptr_array_2D_neurons_reduce_batch_mean);
        
        T_ *tmp_ptr_array_neuron_units_reduce_batch_mean_results(Memory::reallocate_cpp<T_>(*this->ptr_array_2D_neurons_reduce_batch_mean,
                                                                                                                                      tmp_neurons_reduce_batch_size_so_far,
                                                                                                                                      this->neurons_total_reduce_batch_size,
                                                                                                                                      &tmp_dim3_grid_zero,
                                                                                                                                      &tmp_dim3_block_zero,
                                                                                                                                      &tmp_dim3_grid_copy,
                                                                                                                                      &tmp_dim3_block_copy,
                                                                                                                                      false));
        if(tmp_ptr_array_neuron_units_reduce_batch_mean_results == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        // |END| Allocating neurons reduce batch mean. |END|
        
        // Allocating neurons reduce batch variance.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        T_ **tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array(this->ptr_array_2D_neurons_reduce_batch_variance);
        
        T_ *tmp_ptr_array_neuron_units_reduce_batch_variance_results(Memory::reallocate_cpp<T_>(*this->ptr_array_2D_neurons_reduce_batch_variance,
                                                                                                                                         tmp_neurons_reduce_batch_size_so_far,
                                                                                                                                         this->neurons_total_reduce_batch_size,
                                                                                                                                         &tmp_dim3_grid_zero,
                                                                                                                                         &tmp_dim3_block_zero,
                                                                                                                                         &tmp_dim3_grid_copy,
                                                                                                                                         &tmp_dim3_block_copy,
                                                                                                                                         false));
        if(tmp_ptr_array_neuron_units_reduce_batch_variance_results == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        // |END| Allocating neurons reduce batch variance. |END|
        
        // Allocating neurons reduce batch dim3 grid.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_grid_batch(Memory::reallocate<struct dim3>(this->ptr_array_neuron_units_dim3_grid_reduce_batch,
                                                                                                                                       tmp_neurons_reduce_batch_size_so_far * sizeof(struct dim3),
                                                                                                                                       this->neurons_total_reduce_batch_size * sizeof(struct dim3),
                                                                                                                                       false));
        if(tmp_ptr_array_neuron_units_dim3_grid_batch == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        this->ptr_array_neuron_units_dim3_grid_reduce_batch = tmp_ptr_array_neuron_units_dim3_grid_batch;
        // |END| Allocating neurons reduce batch dim3 grid. |END|
            
        // Allocating neurons reduce batch dim3 block.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_block_batch(Memory::reallocate<struct dim3>(this->ptr_array_neuron_units_dim3_block_reduce_batch,
                                                                                                                                         tmp_neurons_reduce_batch_size_so_far * sizeof(struct dim3),
                                                                                                                                         this->neurons_total_reduce_batch_size * sizeof(struct dim3),
                                                                                                                                         false));
        if(tmp_ptr_array_neuron_units_dim3_block_batch == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        this->ptr_array_neuron_units_dim3_block_reduce_batch = tmp_ptr_array_neuron_units_dim3_block_batch;
        // |END| Allocating neurons reduce batch dim3 block. |END|
        
        // Loop through each layers.
        for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            // Get neurons array from that layer.
            tmp_ptr_neuron_unit_it = tmp_ptr_layer_it->ptr_array_neuron_units;

            // Get the reduce batch size of each neurons in that layer.
            tmp_layer_reduce_batch_size = *tmp_ptr_neuron_unit_it->ptr_reduce_batch_size;
            
            // Get the number of neurons in layer.
            tmp_number_neurons_in_layer = *tmp_ptr_layer_it->ptr_number_neurons;
            
            // Loop through each neurons in the layer.
            for(tmp_ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_last_neuron_unit; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                   ++tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array,
                                                                                                                                                                   ++tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array)
            {
                // Result.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array = tmp_ptr_array_neuron_units_reduce_batch_mean_results;
                *tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array = tmp_ptr_array_neuron_units_reduce_batch_variance_results;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_reduce_mean = tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array;
                tmp_ptr_neuron_unit_it->ptr_array_reduce_variance = tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array;
                // |END| Result. |END|
                
                // Number elements to reduce equal the size of batch
                tmp_total_elements_to_reduce = batch_size_received;
                
                // If the neuron is a bias. Number of elements to reduce equal zero.
                if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections == 0u)
                { tmp_total_elements_to_reduce = 0u; }
                
                // If is not the bias. (The bias have no elements to reduce.)
                if(tmp_total_elements_to_reduce != 0u)
                {
                    // Assign dim3 grid to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_threads = tmp_ptr_array_neuron_units_dim3_grid_batch++;
                    // Assign dim3 block to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_threads = tmp_ptr_array_neuron_units_dim3_block_batch++;

                    // Initialize index to zero.
                    tmp_index_dim3 = 0u;

                    // Loop to reduce "number of elements" to one at the end.
                    do
                    {
                        // Compute remaining results to reduce.
                        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                                0u,
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_threads[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)],
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_threads[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)]);

                        // Get the remaining results to reduce.
                        tmp_total_elements_to_reduce = tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_threads[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)].x;

                        // Increment index to dim3.
                        ++tmp_index_dim3;
                    } while(tmp_total_elements_to_reduce != 1u);
                    // |END| dim3. |END|

                    // Increment the begining results by the layer reduce batch size.
                    tmp_ptr_array_neuron_units_reduce_batch_mean_results += tmp_layer_reduce_batch_size;
                    tmp_ptr_array_neuron_units_reduce_batch_variance_results += tmp_layer_reduce_batch_size;
                }
            }
            
            // If some elements need to be reduce in the layer.
            if(tmp_layer_reduce_batch_size != 0u)
            {
                // Increment pointer by (number of neurons in layer minus bias) times (layer reduce batch size minus one).
                tmp_ptr_array_neuron_units_dim3_grid_batch += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_batch_size - 1u);
                tmp_ptr_array_neuron_units_dim3_block_batch += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_batch_size - 1u);
            }
        }
        // |END| COMPUTE DIMENSION REDUCE BATCH. |END|

        this->neurons_total_reduce_batch_size = tmp_neurons_reduce_batch_size_so_far;
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate__Thread__Parameter(size_t const number_threads_received)
{
    if(this->total_parameters_allocated != 0u)
    {
        struct dim3 tmp_dim3_grid_zero(1u,1u, 1u),
                         tmp_dim3_block_zero(1u,1u, 1u),
                         tmp_dim3_grid_copy(1u,1u, 1u),
                         tmp_dim3_block_copy(1u,1u, 1u);
        
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_threads_received * this->total_parameters_allocated,
                                                                                                  this->number_threads * this->total_parameters_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        T_ *tmp_ptr_array_derivatives_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_derivatives_parameters,
                                                                                                                number_threads_received * this->total_parameters_allocated,
                                                                                                                this->number_threads * this->total_parameters_allocated,
                                                                                                                &tmp_dim3_grid_zero,
                                                                                                                &tmp_dim3_block_zero,
                                                                                                                &tmp_dim3_grid_copy,
                                                                                                                &tmp_dim3_block_copy,
                                                                                                                false));
        if(tmp_ptr_array_derivatives_parameters == nullptr)
        {
            PRINT_FORMAT("ERROR: Can not allocate memory." NEW_LINE);

            return(false);
        }
        this->ptr_array_derivatives_parameters = tmp_ptr_array_derivatives_parameters;

        if(this->use_Batch_Renormalization)
        { this->Reset__Derivative_Parameter__Normalized_Unit(); }
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate__Parameter(size_t const number_parameters_received)
{
    if(this->total_parameters_allocated != 0u)
    {
        struct dim3 tmp_dim3_grid_zero(1u,1u, 1u),
                         tmp_dim3_block_zero(1u,1u, 1u),
                         tmp_dim3_grid_copy(1u,1u, 1u),
                         tmp_dim3_block_copy(1u,1u, 1u);
        
        class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        // Parameters.
        if(this->ptr_array_parameters != nullptr)
        {
            this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                      this->total_parameters_allocated,
                                                                                                      tmp_dim3_grid_zero,
                                                                                                      tmp_dim3_block_zero,
                                                                                                      tmp_dim3_grid_copy,
                                                                                                      tmp_dim3_block_copy,
                                                                                                      tmp_ptr_CUDA_Device);

            T_ *tmp_ptr_array_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_parameters,
                                                                                                    number_parameters_received,
                                                                                                    this->total_parameters_allocated,
                                                                                                    &tmp_dim3_grid_zero,
                                                                                                    &tmp_dim3_block_zero,
                                                                                                    &tmp_dim3_grid_copy,
                                                                                                    &tmp_dim3_block_copy));
            if(tmp_ptr_array_parameters == nullptr)
            {
                PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                            __FUNCTION__);

                return(false);
            }
            this->ptr_array_parameters = tmp_ptr_array_parameters;
                
            if(this->Reallocate__Parameter__Optimizer(number_parameters_received) == false)
            {
                PRINT_FORMAT("%s: ERROR: From \"Reallocate__Parameter__Optimizer\"." NEW_LINE, __FUNCTION__);

                return(false);
            }
            else if(this->Use__Regularization_Parameter() && this->Reallocate__Parameter__Regularization(number_parameters_received) == false)
            {
                PRINT_FORMAT("%s: ERROR: From \"Reallocate__Parameter__Regularization\"." NEW_LINE,
                                        __FUNCTION__);

                return(false);
            }
            else if(this->use_Dropout && this->Reallocate__Parameter__Dropout_Bernoulli(number_parameters_received) == false)
            {
                PRINT_FORMAT("%s: ERROR: From \"Reallocate__Parameter__Dropout_Bernoulli\"." NEW_LINE,
                                        __FUNCTION__);

                return(false);
            }
                
            if(this->use_Batch_Renormalization)
            { this->Reset__Parameter__Normalized_Unit(); }
        }
        // |END| Parameters. |END|

        // Derivates parameters.
        if(this->ptr_array_derivatives_parameters != nullptr)
        {
            this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(this->number_threads * number_parameters_received,
                                                                                                      this->number_threads * this->total_parameters_allocated,
                                                                                                      tmp_dim3_grid_zero,
                                                                                                      tmp_dim3_block_zero,
                                                                                                      tmp_dim3_grid_copy,
                                                                                                      tmp_dim3_block_copy,
                                                                                                      tmp_ptr_CUDA_Device,
                                                                                                      false);

            T_ *tmp_ptr_array_derivatives_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_derivatives_parameters,
                                                                                                                    this->number_threads * number_parameters_received,
                                                                                                                    this->number_threads * this->total_parameters_allocated,
                                                                                                                    &tmp_dim3_grid_zero,
                                                                                                                    &tmp_dim3_block_zero,
                                                                                                                    &tmp_dim3_grid_copy,
                                                                                                                    &tmp_dim3_block_copy,
                                                                                                                    false));
            if(tmp_ptr_array_derivatives_parameters == nullptr)
            {
                PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                            __FUNCTION__);

                return(false);
            }
            this->ptr_array_derivatives_parameters = tmp_ptr_array_derivatives_parameters;

            if(this->use_Batch_Renormalization)
            { this->Reset__Derivative_Parameter__Normalized_Unit(); }
        }
        // |END| Derivates parameters. |END|
            
        this->total_parameters = number_parameters_received;
        this->total_parameters_allocated = number_parameters_received;

        // Prepare grids and blocks dimensions.
        this->Prepare__Parameters__Grids_Blocks_Dimensions();

        this->Prepare__Threads_Parameters__Grids_Blocks_Dimensions(this->number_threads);
        // |END| Prepare grids and blocks dimensions. |END|
    }

    return(true);
}
    
__device__ bool CUDA_Neural_Network::Reallocate__Parameter__Regularization(size_t const number_parameters_received)
{
    if(this->ptr_array_mask_regularized_parameters != nullptr)
    {
        struct dim3 tmp_dim3_grid_zero(1u,1u, 1u),
                         tmp_dim3_block_zero(1u,1u, 1u),
                         tmp_dim3_grid_copy(1u,1u, 1u),
                         tmp_dim3_block_copy(1u,1u, 1u);
        
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                  this->total_parameters_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        T_ *tmp_ptr_array_mask_rergularization_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_mask_regularized_parameters,
                                                                                                                                number_parameters_received,
                                                                                                                                this->total_parameters_allocated,
                                                                                                                                &tmp_dim3_grid_zero,
                                                                                                                                &tmp_dim3_block_zero,
                                                                                                                                &tmp_dim3_grid_copy,
                                                                                                                                &tmp_dim3_block_copy,
                                                                                                                                false));
        if(tmp_ptr_array_mask_rergularization_parameters == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        this->ptr_array_mask_regularized_parameters = tmp_ptr_array_mask_rergularization_parameters;
    }

    return(true);
}
    
__device__ bool CUDA_Neural_Network::Reallocate__Parameter__Dropout_Bernoulli(size_t const number_parameters_received)
{
    if(this->ptr_array_mask_dropout_parameters != nullptr)
    {
        struct dim3 tmp_dim3_grid_zero(1u,1u, 1u),
                         tmp_dim3_block_zero(1u,1u, 1u),
                         tmp_dim3_grid_copy(1u,1u, 1u),
                         tmp_dim3_block_copy(1u,1u, 1u);
        
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                  this->total_parameters_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        T_ *tmp_ptr_array_mask_dropout_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_mask_dropout_parameters,
                                                                                                                      number_parameters_received,
                                                                                                                      this->total_parameters_allocated,
                                                                                                                      &tmp_dim3_grid_zero,
                                                                                                                      &tmp_dim3_block_zero,
                                                                                                                      &tmp_dim3_grid_copy,
                                                                                                                      &tmp_dim3_block_copy,
                                                                                                                      false));
        if(tmp_ptr_array_mask_dropout_parameters == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        this->ptr_array_mask_dropout_parameters = tmp_ptr_array_mask_dropout_parameters;
        
        // If array increase in size, initialize the new entries to one.
        if(this->total_weights_allocated < number_parameters_received)
        {
            this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(number_parameters_received - this->total_weights_allocated,
                                                                                                                                                  0u,
                                                                                                                                                  tmp_dim3_grid_copy,
                                                                                                                                                  tmp_dim3_block_copy);

            Memory::Fill_1D<T_>(number_parameters_received - this->total_weights_allocated,
                                                                 tmp_ptr_array_mask_dropout_parameters + this->total_weights_allocated,
                                                                 1_T,
                                                                 &tmp_dim3_grid_copy,
                                                                 &tmp_dim3_block_copy);
        }
    }

    return(true);
}
    
__device__ bool CUDA_Neural_Network::Reallocate__Parameter__Optimizer(size_t const number_parameters_received)
{
    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD: return(this->Reallocate__Parameter__Gradient_Descent(number_parameters_received));
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus: return(this->Reallocate__Parameter__iRPROP_minus(number_parameters_received));
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus: return(this->Reallocate__Parameter__iRPROP_plus(number_parameters_received));
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: return(this->Reallocate__Parameter__Adam(number_parameters_received));
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: return(this->Reallocate__Parameter__AMSGrad(number_parameters_received));
        default: return(true);
    }
}

__device__ bool CUDA_Neural_Network::Reallocate__Parameter__Gradient_Descent(size_t const number_parameters_received)
{
    if(this->learning_momentum != 0_T
        &&
        this->ptr_array_previous_delta_parameters != nullptr)
    {
        struct dim3 tmp_dim3_grid_zero(1u,1u, 1u),
                         tmp_dim3_block_zero(1u,1u, 1u),
                         tmp_dim3_grid_copy(1u,1u, 1u),
                         tmp_dim3_block_copy(1u,1u, 1u);
        
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                  this->total_parameters_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        // Previous delta parameters.
        T_ *tmp_ptr_array_previous_delta_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_previous_delta_parameters,
                                                                                                                      number_parameters_received,
                                                                                                                      this->total_parameters_allocated,
                                                                                                                      &tmp_dim3_grid_zero,
                                                                                                                      &tmp_dim3_block_zero,
                                                                                                                      &tmp_dim3_grid_copy,
                                                                                                                      &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_delta_parameters == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        this->ptr_array_previous_delta_parameters = tmp_ptr_array_previous_delta_parameters;
        // |END| Previous delta parameters. |END|
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate__Parameter__iRPROP_minus(size_t const number_parameters_received)
{
    struct dim3 tmp_dim3_grid_zero(1u,1u, 1u),
                     tmp_dim3_block_zero(1u,1u, 1u),
                     tmp_dim3_grid_copy(1u,1u, 1u),
                     tmp_dim3_block_copy(1u,1u, 1u);
        
    if(this->ptr_array_previous_steps != nullptr || this->ptr_array_previous_derivatives_parameters != nullptr)
    {
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                  this->total_parameters_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
    }

    if(this->ptr_array_previous_steps != nullptr)
    {
        T_ *tmp_ptr_array_previous_steps(Memory::reallocate_cpp<T_>(this->ptr_array_previous_steps,
                                                                                                     number_parameters_received,
                                                                                                     this->total_parameters_allocated,
                                                                                                     &tmp_dim3_grid_zero,
                                                                                                     &tmp_dim3_block_zero,
                                                                                                     &tmp_dim3_grid_copy,
                                                                                                     &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_steps == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        this->ptr_array_previous_steps = tmp_ptr_array_previous_steps;
        
        if(this->total_parameters_allocated < number_parameters_received)
        {
            Memory::Fill_1D<T_>(number_parameters_received - this->total_parameters_allocated,
                                                                 tmp_ptr_array_previous_steps + this->total_parameters_allocated,
                                                                 this->rprop_delta_zero,
                                                                 &tmp_dim3_grid_zero,
                                                                 &tmp_dim3_block_zero);
        }
    }
    
    if(this->ptr_array_previous_derivatives_parameters != nullptr)
    {
        T_ *tmp_ptr_array_previous_derivatives_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_previous_derivatives_parameters,
                                                                                                                              number_parameters_received,
                                                                                                                              this->total_parameters_allocated,
                                                                                                                              &tmp_dim3_grid_zero,
                                                                                                                              &tmp_dim3_block_zero,
                                                                                                                              &tmp_dim3_grid_copy,
                                                                                                                              &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_derivatives_parameters == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        this->ptr_array_previous_derivatives_parameters = tmp_ptr_array_previous_derivatives_parameters;
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate__Parameter__iRPROP_plus(size_t const number_parameters_received)
{
    struct dim3 tmp_dim3_grid_zero(1u,1u, 1u),
                     tmp_dim3_block_zero(1u,1u, 1u),
                     tmp_dim3_grid_copy(1u,1u, 1u),
                     tmp_dim3_block_copy(1u,1u, 1u);
        
    if(this->ptr_array_previous_steps != nullptr
      ||
      this->ptr_array_previous_delta_parameters != nullptr
      ||
      this->ptr_array_previous_derivatives_parameters != nullptr)
    {
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                    this->total_parameters_allocated,
                                                                                                    tmp_dim3_grid_zero,
                                                                                                    tmp_dim3_block_zero,
                                                                                                    tmp_dim3_grid_copy,
                                                                                                    tmp_dim3_block_copy,
                                                                                                    this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
    }

    if(this->ptr_array_previous_steps != nullptr)
    {
        T_ *tmp_ptr_array_previous_steps(Memory::reallocate_cpp<T_>(this->ptr_array_previous_steps,
                                                                                                    number_parameters_received,
                                                                                                    this->total_parameters_allocated,
                                                                                                    &tmp_dim3_grid_zero,
                                                                                                    &tmp_dim3_block_zero,
                                                                                                    &tmp_dim3_grid_copy,
                                                                                                    &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_steps == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        this->ptr_array_previous_steps = tmp_ptr_array_previous_steps;
        
        if(this->total_parameters_allocated < number_parameters_received)
        {
            Memory::Fill_1D<T_>(number_parameters_received - this->total_parameters_allocated,
                                                                 tmp_ptr_array_previous_steps + this->total_parameters_allocated,
                                                                 this->rprop_delta_zero,
                                                                 &tmp_dim3_grid_zero,
                                                                 &tmp_dim3_block_zero);
        }
    }
    
    if(this->ptr_array_previous_delta_parameters != nullptr)
    {
        T_ *tmp_ptr_array_previous_delta_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_previous_delta_parameters,
                                                                                                                    number_parameters_received,
                                                                                                                    this->total_parameters_allocated,
                                                                                                                    &tmp_dim3_grid_zero,
                                                                                                                    &tmp_dim3_block_zero,
                                                                                                                    &tmp_dim3_grid_copy,
                                                                                                                    &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_delta_parameters == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        this->ptr_array_previous_delta_parameters = tmp_ptr_array_previous_delta_parameters;
    }
    
    if(this->ptr_array_previous_derivatives_parameters != nullptr)
    {
        T_ *tmp_ptr_array_previous_derivatives_parameters(Memory::reallocate_cpp<T_>(this->ptr_array_previous_derivatives_parameters,
                                                                                                                            number_parameters_received,
                                                                                                                            this->total_parameters_allocated,
                                                                                                                            &tmp_dim3_grid_zero,
                                                                                                                            &tmp_dim3_block_zero,
                                                                                                                            &tmp_dim3_grid_copy,
                                                                                                                            &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_derivatives_parameters == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        this->ptr_array_previous_derivatives_parameters = tmp_ptr_array_previous_derivatives_parameters;
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate__Parameter__Adam(size_t const number_parameters_received)
{
    struct dim3 tmp_dim3_grid_zero(1u,1u, 1u),
                     tmp_dim3_block_zero(1u,1u, 1u),
                     tmp_dim3_grid_copy(1u,1u, 1u),
                     tmp_dim3_block_copy(1u,1u, 1u);
        
    if(this->ptr_array_previous_biased_first_moment != nullptr || this->ptr_array_previous_biased_second_moment != nullptr)
    {
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                    this->total_parameters_allocated,
                                                                                                    tmp_dim3_grid_zero,
                                                                                                    tmp_dim3_block_zero,
                                                                                                    tmp_dim3_grid_copy,
                                                                                                    tmp_dim3_block_copy,
                                                                                                    this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
    }

    if(this->ptr_array_previous_biased_first_moment != nullptr)
    {
        T_ *tmp_ptr_array_previous_biased_first_moment(Memory::reallocate_cpp<T_>(this->ptr_array_previous_biased_first_moment,
                                                                                                                            number_parameters_received,
                                                                                                                            this->total_parameters_allocated,
                                                                                                                            &tmp_dim3_grid_zero,
                                                                                                                            &tmp_dim3_block_zero,
                                                                                                                            &tmp_dim3_grid_copy,
                                                                                                                            &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_biased_first_moment == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        this->ptr_array_previous_biased_first_moment = tmp_ptr_array_previous_biased_first_moment;
    }
    
    if(this->ptr_array_previous_biased_second_moment != nullptr)
    {
        T_ *tmp_ptr_array_previous_biased_second_moment(Memory::reallocate_cpp<T_>(this->ptr_array_previous_biased_second_moment,
                                                                                                                                number_parameters_received,
                                                                                                                                this->total_parameters_allocated,
                                                                                                                                &tmp_dim3_grid_zero,
                                                                                                                                &tmp_dim3_block_zero,
                                                                                                                                &tmp_dim3_grid_copy,
                                                                                                                                &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_biased_second_moment == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        this->ptr_array_previous_biased_second_moment = tmp_ptr_array_previous_biased_second_moment;
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate__Parameter__AMSGrad(size_t const number_parameters_received)
{
    struct dim3 tmp_dim3_grid_zero(1u,1u, 1u),
                     tmp_dim3_block_zero(1u,1u, 1u),
                     tmp_dim3_grid_copy(1u,1u, 1u),
                     tmp_dim3_block_copy(1u,1u, 1u);
        
    if(this->ptr_array_previous_biased_first_moment != nullptr
      ||
      this->ptr_array_previous_biased_second_moment != nullptr
      ||
      this->ptr_array_previous_biased_second_moment_hat != nullptr)
    {
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                  this->total_parameters_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
    }
    
    if(this->ptr_array_previous_biased_first_moment != nullptr)
    {
        T_ *tmp_ptr_array_previous_biased_first_moment(Memory::reallocate_cpp<T_>(this->ptr_array_previous_biased_first_moment,
                                                                                                                            number_parameters_received,
                                                                                                                            this->total_parameters_allocated,
                                                                                                                            &tmp_dim3_grid_zero,
                                                                                                                            &tmp_dim3_block_zero,
                                                                                                                            &tmp_dim3_grid_copy,
                                                                                                                            &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_biased_first_moment == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        this->ptr_array_previous_biased_first_moment = tmp_ptr_array_previous_biased_first_moment;
    }
    
    if(this->ptr_array_previous_biased_second_moment != nullptr)
    {
        T_ *tmp_ptr_array_previous_biased_second_moment(Memory::reallocate_cpp<T_>(this->ptr_array_previous_biased_second_moment,
                                                                                                                                number_parameters_received,
                                                                                                                                this->total_parameters_allocated,
                                                                                                                                &tmp_dim3_grid_zero,
                                                                                                                                &tmp_dim3_block_zero,
                                                                                                                                &tmp_dim3_grid_copy,
                                                                                                                                &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_biased_second_moment == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        this->ptr_array_previous_biased_second_moment = tmp_ptr_array_previous_biased_second_moment;
    }

    if(this->ptr_array_previous_biased_second_moment_hat != nullptr)
    {
        T_ *tmp_ptr_array_previous_biased_second_moment_hat(Memory::reallocate_cpp<T_>(this->ptr_array_previous_biased_second_moment_hat,
                                                                                                                                        number_parameters_received,
                                                                                                                                        this->total_parameters_allocated,
                                                                                                                                        &tmp_dim3_grid_zero,
                                                                                                                                        &tmp_dim3_block_zero,
                                                                                                                                        &tmp_dim3_grid_copy,
                                                                                                                                        &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_biased_second_moment_hat == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        this->ptr_array_previous_biased_second_moment_hat = tmp_ptr_array_previous_biased_second_moment_hat;
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate_Connections(size_t const total_connections_received)
{
    PRINT_FORMAT("%s: [FUNCTION DEPRECATED] Need to Fix \"Reallocate_Connections\" algorithm." NEW_LINE, __FUNCTION__);

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate_Neurons(size_t const total_neuron_units_received, bool const reSet__neuron_position_received)
{
    PRINT_FORMAT("%s: [FUNCTION DEPRECATED] Need to Fix \"Reallocate_Neurons\" algorithm." NEW_LINE, __FUNCTION__);

    return(true);
}

__device__ bool CUDA_Neural_Network::Reallocate_Layers(size_t const total_layers_received)
{
    PRINT_FORMAT("%s: [FUNCTION DEPRECATED] Need to Fix \"Reallocate_Layers\" algorithm." NEW_LINE, __FUNCTION__);

    return(true);
}
