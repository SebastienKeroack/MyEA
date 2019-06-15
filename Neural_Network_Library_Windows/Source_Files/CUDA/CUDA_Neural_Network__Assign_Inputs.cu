#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

template<typename T>
__global__ void kernel__CNeural_Network__Assign_Inputs__Dropout_Bernoulli__Training(bool const *const ptr_array_mask_dropout_received,
                                                                                                                 T *const ptr_array_input_layer_value_received,
                                                                                                                 T const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(ptr_array_mask_dropout_received[tmp_thread_global_index])
    { ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index]; }
    else
    { ptr_array_input_layer_value_received[tmp_thread_global_index] = T(0); }
}

template<typename T>
__global__ void kernel__CNeural_Network__Assign_Inputs__Dropout_Bernoulli__Training(size_t const size_received,
                                                                                                                 bool const *const ptr_array_mask_dropout_received,
                                                                                                                 T *const ptr_array_input_layer_value_received,
                                                                                                                 T const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(tmp_thread_global_index < size_received && ptr_array_mask_dropout_received[tmp_thread_global_index])
    { ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index]; }
    else
    { ptr_array_input_layer_value_received[tmp_thread_global_index] = T(0); }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Assign_Inputs__Dropout_Bernoulli__Training(size_t const size_received,
                                                                                                                          bool const *const ptr_array_mask_dropout_received,
                                                                                                                          T *const ptr_array_input_layer_value_received,
                                                                                                                          T const *const ptr_array_inputs_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        if(ptr_array_mask_dropout_received[tmp_thread_global_index])
        { ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index]; }
        else
        { ptr_array_input_layer_value_received[tmp_thread_global_index] = T(0); }

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__CNeural_Network__Assign_Inputs__Dropout_Bernoulli__Testing(T const dropout_values,
                                                                                                                                T *const ptr_array_input_layer_value_received,
                                                                                                                                T const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index] * dropout_values;
}

template<typename T>
__global__ void kernel__CNeural_Network__Assign_Inputs__Dropout_Bernoulli__Testing(size_t const size_received,
                                                                                                                                T const dropout_values,
                                                                                                                                T *const ptr_array_input_layer_value_received,
                                                                                                                                T const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(tmp_thread_global_index < size_received)
    { ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index] * dropout_values; }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Assign_Inputs__Dropout_Bernoulli__Testing(size_t const size_received,
                                                                                                                         T const dropout_values,
                                                                                                                         T *const ptr_array_input_layer_value_received,
                                                                                                                         T const *const ptr_array_inputs_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index] * dropout_values;

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__CNeural_Network__Assign_Inputs(T *const ptr_array_input_layer_value_received, T const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index];
}

template<typename T>
__global__ void kernel__CNeural_Network__Assign_Inputs(size_t const size_received,
                                                                                     T *const ptr_array_input_layer_value_received,
                                                                                     T const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
        
    if(tmp_thread_global_index < size_received)
    { ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index]; }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Assign_Inputs(size_t const size_received,
                                                                                              T *const ptr_array_input_layer_value_received,
                                                                                              T const *const ptr_array_inputs_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
        
    do
    {
        ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index];

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Assign_Inputs(bool &ref_synchronized_received,
                                                                                 size_t const thread_index_received,
                                                                                 T_ const *ptr_array_inputs_received)
{
    struct CUDA_Layer const *const tmp_ptr_input_layer(this->ptr_array_layers);
    
    bool const *tmp_ptr_array_input_layer_mask_dropout;

    T_ *tmp_ptr_array_input_layer_values(tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values + thread_index_received * *tmp_ptr_input_layer->ptr_number_neurons),
         tmp_probability_retained_unit;
    T_ const *const tmp_ptr_array_input_layers_values_end(tmp_ptr_array_input_layer_values + this->number_inputs);

    if(this->use_Dropout)
    {
        if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
        {
            tmp_ptr_array_input_layer_mask_dropout = tmp_ptr_input_layer->ptr_array_neuron_units->ptr_mask_dropout_bernoulli;

            // Condition to enter into dynamic parallelisme of each.
            if(USE_PARALLEL && this->number_inputs >= warpSize)
            {
                // Set the synchronisation state to false. Because we launch a kernel.
                ref_synchronized_received = false;
                
                LAUNCH_KERNEL_POINTER_1D(CNeural_Network__Assign_Inputs__Dropout_Bernoulli__Training<T_>,
                                                                  tmp_ptr_input_layer->ptr_dim3_grid_neurons,
                                                                  tmp_ptr_input_layer->ptr_dim3_block_neurons,
                                                                  0_zu,
                                                                  this->number_inputs,
                                                                  tmp_ptr_array_input_layer_mask_dropout,
                                                                  tmp_ptr_array_input_layer_values,
                                                                  ptr_array_inputs_received)
            }
            // Standard assignment inputs.
            else
            {
                for(; tmp_ptr_array_input_layer_values != tmp_ptr_array_input_layers_values_end; ++tmp_ptr_array_input_layer_values,
                                                                                                                                      ++tmp_ptr_array_input_layer_mask_dropout,
                                                                                                                                      ++ptr_array_inputs_received)
                {
                    if(*tmp_ptr_array_input_layer_mask_dropout)
                    { *tmp_ptr_array_input_layer_values = *ptr_array_inputs_received; }
                    else
                    { *tmp_ptr_array_input_layer_values = 0_T; }
                }
            }
        }
        else
        {
            // Condition to enter into dynamic parallelisme of each.
            if(USE_PARALLEL && this->number_inputs >= warpSize)
            {
                // Set the synchronisation state to false. Because we launch a kernel.
                ref_synchronized_received = false;
                
                LAUNCH_KERNEL_POINTER_1D(CNeural_Network__Assign_Inputs__Dropout_Bernoulli__Testing<T_>,
                                                                  tmp_ptr_input_layer->ptr_dim3_grid_neurons,
                                                                  tmp_ptr_input_layer->ptr_dim3_block_neurons,
                                                                  0_zu,
                                                                  this->number_inputs,
                                                                  tmp_ptr_input_layer->dropout_values[0u],
                                                                  tmp_ptr_array_input_layer_values,
                                                                  ptr_array_inputs_received)
            }
            // Standard assignment inputs.
            else
            {
                tmp_probability_retained_unit = tmp_ptr_input_layer->dropout_values[0u];
                
                for(; tmp_ptr_array_input_layer_values != tmp_ptr_array_input_layers_values_end; ++tmp_ptr_array_input_layer_values,
                                                                                                                                      ++ptr_array_inputs_received)
                { *tmp_ptr_array_input_layer_values = *ptr_array_inputs_received * tmp_probability_retained_unit; }
            }
        }
    }
    else
    {
        // Condition to enter into dynamic parallelisme of each.
        if(USE_PARALLEL && this->number_inputs >= warpSize)
        {
            // Set the synchronisation state to false. Because we launch a kernel.
            ref_synchronized_received = false;
            
            LAUNCH_KERNEL_POINTER_1D(CNeural_Network__Assign_Inputs<T_>,
                                                              tmp_ptr_input_layer->ptr_dim3_grid_neurons,
                                                              tmp_ptr_input_layer->ptr_dim3_block_neurons,
                                                              0_zu,
                                                              this->number_inputs,
                                                              tmp_ptr_array_input_layer_values,
                                                              ptr_array_inputs_received)
        }
        // Standard assignment inputs.
        else
        {
            for(; tmp_ptr_array_input_layer_values != tmp_ptr_array_input_layers_values_end; ++tmp_ptr_array_input_layer_values,
                                                                                                                                    ++ptr_array_inputs_received)
            { *tmp_ptr_array_input_layer_values = *ptr_array_inputs_received; }
        }
    }
}