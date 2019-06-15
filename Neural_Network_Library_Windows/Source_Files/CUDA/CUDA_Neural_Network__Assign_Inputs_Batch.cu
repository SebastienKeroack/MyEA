#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

template<typename T>
__global__ void kernel__CNeural_Network__Assign_Input_Batch(size_t const size_inputs_received,
                                                                                              size_t const number_neurons_received,
                                                                                              T *const ptr_array_input_layer_value_received,
                                                                                              T const *const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
                       tmp_data_index(tmp_thread_global_index / size_inputs_received),
                       tmp_input_index(tmp_thread_global_index % size_inputs_received);
    
    ptr_array_input_layer_value_received[tmp_data_index * number_neurons_received + tmp_input_index] = ptr_array_inputs_received[tmp_data_index][tmp_input_index];
}

template<typename T>
__global__ void kernel__CNeural_Network__Assign_Input_Batch(size_t const size_received,
                                                                                              size_t const size_inputs_received,
                                                                                              size_t const number_neurons_received,
                                                                                              T *const ptr_array_input_layer_value_received,
                                                                                              T const *const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
        
    if(tmp_thread_global_index < size_received)
    {
        size_t const tmp_data_index((tmp_thread_global_index / size_inputs_received)),
                           tmp_input_index(tmp_thread_global_index % size_inputs_received);

        ptr_array_input_layer_value_received[tmp_data_index * number_neurons_received + tmp_input_index] = ptr_array_inputs_received[tmp_data_index][tmp_input_index];
    }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Assign_Input_Batch(size_t const size_received,
                                                                                                       size_t const size_inputs_received,
                                                                                                       size_t const number_neurons_received,
                                                                                                       T *const ptr_array_input_layer_value_received,
                                                                                                       T const *const *const ptr_array_inputs_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
              tmp_data_index,
              tmp_input_index;
        
    do
    {
        tmp_data_index = (tmp_thread_global_index / size_inputs_received);
        tmp_input_index = tmp_thread_global_index % size_inputs_received;

        ptr_array_input_layer_value_received[tmp_data_index * number_neurons_received + tmp_input_index] = ptr_array_inputs_received[tmp_data_index][tmp_input_index];

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__CNeural_Network__Assign_Input_Batch__Dropout_Bernoulli__Testing(size_t const size_inputs_received,
                                                                                                                        size_t const number_neurons_received,
                                                                                                                        T const probability_retained_unit_received,
                                                                                                                        T *const ptr_array_input_layer_value_received,
                                                                                                                        T const *const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
                       tmp_data_index((tmp_thread_global_index / size_inputs_received)),
                       tmp_input_index(tmp_thread_global_index % size_inputs_received);
    
    ptr_array_input_layer_value_received[tmp_data_index * number_neurons_received + tmp_input_index] = ptr_array_inputs_received[tmp_data_index][tmp_input_index] * probability_retained_unit_received;
}

template<typename T>
__global__ void kernel__CNeural_Network__Assign_Input_Batch__Dropout_Bernoulli__Testing(size_t const size_received,
                                                                                                                        size_t const size_inputs_received,
                                                                                                                        size_t const number_neurons_received,
                                                                                                                        T const probability_retained_unit_received,
                                                                                                                        T *const ptr_array_input_layer_value_received,
                                                                                                                        T const *const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
        
    if(tmp_thread_global_index < size_received)
    {
        size_t const tmp_data_index((tmp_thread_global_index / size_inputs_received)),
                           tmp_input_index(tmp_thread_global_index % size_inputs_received);
        
        ptr_array_input_layer_value_received[tmp_data_index * number_neurons_received + tmp_input_index] = ptr_array_inputs_received[tmp_data_index][tmp_input_index] * probability_retained_unit_received;
    }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Assign_Input_Batch__Dropout_Bernoulli__Testing(size_t const size_received,
                                                                                                                                  size_t const size_inputs_received,
                                                                                                                                  size_t const number_neurons_received,
                                                                                                                                  T const probability_retained_unit_received,
                                                                                                                                  T *const ptr_array_input_layer_value_received,
                                                                                                                                  T const *const *const ptr_array_inputs_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
              tmp_data_index,
              tmp_input_index;
    
    do
    {
        tmp_data_index = (tmp_thread_global_index / size_inputs_received);
        tmp_input_index = tmp_thread_global_index % size_inputs_received;
        
        ptr_array_input_layer_value_received[tmp_data_index * number_neurons_received + tmp_input_index] = ptr_array_inputs_received[tmp_data_index][tmp_input_index] * probability_retained_unit_received;

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__CNeural_Network__Assign_Input_Batch__Dropout_Bernoulli__Training(size_t const size_inputs_received,
                                                                                                                        size_t const number_neurons_received,
                                                                                                                        bool const *const ptr_array_input_layer_mask_dropout_received,
                                                                                                                        T *const ptr_array_input_layer_value_received,
                                                                                                                        T const *const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
                       tmp_data_index((tmp_thread_global_index / size_inputs_received)),
                       tmp_input_index(tmp_thread_global_index % size_inputs_received);
    
    if(ptr_array_input_layer_mask_dropout_received[tmp_input_index])
    { ptr_array_input_layer_value_received[tmp_data_index * number_neurons_received + tmp_input_index] = ptr_array_inputs_received[tmp_data_index][tmp_input_index]; }
    else
    { ptr_array_input_layer_value_received[tmp_data_index * number_neurons_received + tmp_input_index] = T(0); }
}

template<typename T>
__global__ void kernel__CNeural_Network__Assign_Input_Batch__Dropout_Bernoulli__Training(size_t const size_received,
                                                                                                                        size_t const size_inputs_received,
                                                                                                                        size_t const number_neurons_received,
                                                                                                                        bool const *const ptr_array_input_layer_mask_dropout_received,
                                                                                                                        T *const ptr_array_input_layer_value_received,
                                                                                                                        T const *const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
        
    if(tmp_thread_global_index < size_received)
    {
        size_t const tmp_data_index((tmp_thread_global_index / size_inputs_received)),
                           tmp_input_index(tmp_thread_global_index % size_inputs_received);
        
        if(ptr_array_input_layer_mask_dropout_received[tmp_input_index])
        { ptr_array_input_layer_value_received[tmp_data_index * number_neurons_received + tmp_input_index] = ptr_array_inputs_received[tmp_data_index][tmp_input_index]; }
        else
        { ptr_array_input_layer_value_received[tmp_data_index * number_neurons_received + tmp_input_index] = T(0); }
    }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Assign_Input_Batch__Dropout_Bernoulli__Training(size_t const size_received,
                                                                                                                                  size_t const size_inputs_received,
                                                                                                                                  size_t const number_neurons_received,
                                                                                                                                  bool const *const ptr_array_input_layer_mask_dropout_received,
                                                                                                                                  T *const ptr_array_input_layer_value_received,
                                                                                                                                  T const *const *const ptr_array_inputs_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
              tmp_data_index,
              tmp_input_index;
    
    do
    {
        tmp_data_index = (tmp_thread_global_index / size_inputs_received);
        tmp_input_index = tmp_thread_global_index % size_inputs_received;
        
        if(ptr_array_input_layer_mask_dropout_received[tmp_input_index])
        { ptr_array_input_layer_value_received[tmp_data_index * number_neurons_received + tmp_input_index] = ptr_array_inputs_received[tmp_data_index][tmp_input_index]; }
        else
        { ptr_array_input_layer_value_received[tmp_data_index * number_neurons_received + tmp_input_index] = T(0); }

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Assign_Inputs_Batch(bool &ref_synchronized_received,
                                                                                  size_t const batch_size_received,
                                                                                  T_ const *const *const ptr_matrix_inputs_received)
{
    size_t const tmp_batch_size_times_number_inputs(batch_size_received * this->number_inputs);
    size_t tmp_data_index;

    T_ const *tmp_ptr_array_inputs;
    
    // Variable to cache optimal size to launch dynamic parallelisme through the GPU.
    struct dim3 tmp_dim3_grid,
                     tmp_dim3_block;

    struct CUDA_Layer *const tmp_ptr_input_layer(this->ptr_array_layers);
    
    bool const *tmp_ptr_array_input_layer_mask_dropout;

    T_ const *tmp_ptr_array_input_layers_values_end;
    T_ *tmp_ptr_array_input_layer_values,
         tmp_probability_retained_unit;

    // Condition to enter into dynamic parallelisme.
    if(USE_PARALLEL && tmp_batch_size_times_number_inputs >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;

        // Get or compute the optimal size to launch dynamic parallelisme through the GPU.
        tmp_ptr_input_layer->ptr_Class_Storage_Dim3_Batch->Get__Dim3_1D(tmp_batch_size_times_number_inputs,
                                                                                                             tmp_dim3_grid,
                                                                                                             tmp_dim3_block,
                                                                                                             this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
        
        // Condition to know if we use dropout. For droped inputs.
        if(this->use_Dropout)
        {
            if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
            {
                LAUNCH_KERNEL_1D(CNeural_Network__Assign_Input_Batch__Dropout_Bernoulli__Training<T_>,
                                                  tmp_dim3_grid,
                                                  tmp_dim3_block,
                                                  0_zu,
                                                  tmp_batch_size_times_number_inputs,
                                                  this->number_inputs,
                                                  *tmp_ptr_input_layer->ptr_number_neurons,
                                                  tmp_ptr_input_layer->ptr_array_neuron_units->ptr_mask_dropout_bernoulli,
                                                  tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values,
                                                  ptr_matrix_inputs_received)
            }
            else
            {
                LAUNCH_KERNEL_1D(CNeural_Network__Assign_Input_Batch__Dropout_Bernoulli__Testing<T_>,
                                                  tmp_dim3_grid,
                                                  tmp_dim3_block,
                                                  0_zu,
                                                  tmp_batch_size_times_number_inputs,
                                                  this->number_inputs,
                                                  *tmp_ptr_input_layer->ptr_number_neurons,
                                                  tmp_ptr_input_layer->dropout_values[0u],
                                                  tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values,
                                                  ptr_matrix_inputs_received)
            }
        }
        // Standard assignment inputs.
        else
        {
            LAUNCH_KERNEL_1D(CNeural_Network__Assign_Input_Batch<T_>,
                                              tmp_dim3_grid,
                                              tmp_dim3_block,
                                              0_zu,
                                              tmp_batch_size_times_number_inputs,
                                              this->number_inputs,
                                              *tmp_ptr_input_layer->ptr_number_neurons,
                                              tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values,
                                              ptr_matrix_inputs_received)
        }
    }
    // If we don't enter into dynamic parallelisme, we serialize the computation.
    else
    {
        // Condition to know if we use dropout. For droped inputs.
        if(this->use_Dropout)
        {
            if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
            {
                // Loop through each sample data.
                for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
                {
                    // Get inputs array from sample.
                    tmp_ptr_array_inputs = ptr_matrix_inputs_received[tmp_data_index];
                
                    // Assign value position.
                    tmp_ptr_array_input_layer_values = tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values + tmp_data_index * *tmp_ptr_input_layer->ptr_number_neurons;

                    // Assign value end pointer.
                    tmp_ptr_array_input_layers_values_end = tmp_ptr_array_input_layer_values + this->number_inputs;
                
                    // Assign mask dropout.
                    tmp_ptr_array_input_layer_mask_dropout = tmp_ptr_input_layer->ptr_array_neuron_units->ptr_mask_dropout_bernoulli;
                
                    // Loop through each input.
                    for(; tmp_ptr_array_input_layer_values != tmp_ptr_array_input_layers_values_end; ++tmp_ptr_array_input_layer_values,
                                                                                                                                           ++tmp_ptr_array_input_layer_mask_dropout,
                                                                                                                                           ++tmp_ptr_array_inputs)
                    {
                        // Condition to see if the entry is alive. If yes, assign an input from sample.
                        if(*tmp_ptr_array_input_layer_mask_dropout)
                        { *tmp_ptr_array_input_layer_values = *tmp_ptr_array_inputs; }
                        // Entry dead. Give it a zero value.
                        else
                        { *tmp_ptr_array_input_layer_values = 0_T; }
                    }
                }
            }
            else
            {
                tmp_probability_retained_unit = tmp_ptr_input_layer->dropout_values[0u];
                
                // Loop through each sample data.
                for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
                {
                    // Get inputs array from sample.
                    tmp_ptr_array_inputs = ptr_matrix_inputs_received[tmp_data_index];
                
                    // Assign value position.
                    tmp_ptr_array_input_layer_values = tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values + tmp_data_index * *tmp_ptr_input_layer->ptr_number_neurons;

                    // Assign value end pointer.
                    tmp_ptr_array_input_layers_values_end = tmp_ptr_array_input_layer_values + this->number_inputs;
                
                    // Assign mask dropout.
                    tmp_ptr_array_input_layer_mask_dropout = tmp_ptr_input_layer->ptr_array_neuron_units->ptr_mask_dropout_bernoulli;
                
                    // Loop through each input.
                    for(; tmp_ptr_array_input_layer_values != tmp_ptr_array_input_layers_values_end; ++tmp_ptr_array_input_layer_values,
                                                                                                                                           ++tmp_ptr_array_input_layer_mask_dropout,
                                                                                                                                           ++tmp_ptr_array_inputs)
                    { *tmp_ptr_array_input_layer_values = *tmp_ptr_array_inputs * tmp_probability_retained_unit; }
                }
            }
        }
        // Standard assignment inputs.
        else
        {
            // Loop through each sample data.
            for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
            {
                // Get inputs array from sample.
                tmp_ptr_array_inputs = ptr_matrix_inputs_received[tmp_data_index];
                
                // Assign value position.
                tmp_ptr_array_input_layer_values = tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values + tmp_data_index * *tmp_ptr_input_layer->ptr_number_neurons;

                // Assign value end pointer.
                tmp_ptr_array_input_layers_values_end = tmp_ptr_array_input_layer_values + this->number_inputs;

                // Loop through each input.
                for(; tmp_ptr_array_input_layer_values != tmp_ptr_array_input_layers_values_end; ++tmp_ptr_array_input_layer_values,
                                                                                                                                       ++tmp_ptr_array_inputs)
                { *tmp_ptr_array_input_layer_values = *tmp_ptr_array_inputs; }
            }
        }
    }
}