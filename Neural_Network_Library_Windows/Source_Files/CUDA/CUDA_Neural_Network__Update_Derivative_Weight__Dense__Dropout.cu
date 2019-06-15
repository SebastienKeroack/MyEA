#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Multiply_1D.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

template<typename T>
__global__ void kernel__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons(size_t const number_neurons_received,
                                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                                        bool const *const ptr_array_mask_dropout_received,
                                                                                                                                                                                        T *const ptr_array_derivatives_paramters_received,
                                                                                                                                                                                        T const *const ptr_array_neuron_units_errors_received,
                                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(ptr_array_mask_dropout_received[tmp_thread_global_index])
    {
        T const tmp_error(ptr_array_neuron_units_errors_received[tmp_thread_global_index]);

        Multiply::FMAC_X_YX_1D<T>(number_connections_received,
                                                    ptr_array_derivatives_paramters_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                    tmp_error,
                                                    ptr_array_previous_layer_outputs_received,
                                                    ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                                    ptr_array_dim3_block_connections_received + tmp_thread_global_index);
    
        ptr_array_derivatives_paramters_received[tmp_thread_global_index * (number_connections_received + 1u) + number_connections_received] += tmp_error; // Bias.
    }
}

template<typename T>
__global__ void kernel__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                                        bool const *const ptr_array_mask_dropout_received,
                                                                                                                                                                                        T *const ptr_array_derivatives_paramters_received,
                                                                                                                                                                                        T const *const ptr_array_neuron_units_errors_received,
                                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(tmp_thread_global_index < size_received && ptr_array_mask_dropout_received[tmp_thread_global_index])
    {
        T const tmp_error(ptr_array_neuron_units_errors_received[tmp_thread_global_index]);
        
        Multiply::FMAC_X_YX_1D<T>(number_connections_received,
                                                    ptr_array_derivatives_paramters_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                    tmp_error,
                                                    ptr_array_previous_layer_outputs_received,
                                                    ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                                    ptr_array_dim3_block_connections_received + tmp_thread_global_index);
        
        ptr_array_derivatives_paramters_received[tmp_thread_global_index * (number_connections_received + 1u) + number_connections_received] += tmp_error; // Bias.
    }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                                bool const *const ptr_array_mask_dropout_received,
                                                                                                                                                                                                T *const ptr_array_derivatives_paramters_received,
                                                                                                                                                                                                T const *const ptr_array_neuron_units_errors_received,
                                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_error;

    do
    {
        if(ptr_array_mask_dropout_received[tmp_thread_global_index])
        {
            tmp_error = ptr_array_neuron_units_errors_received[tmp_thread_global_index];
        
            Multiply::FMAC_X_YX_1D<T>(number_connections_received,
                                                        ptr_array_derivatives_paramters_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                        tmp_error,
                                                        ptr_array_previous_layer_outputs_received,
                                                        ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                                        ptr_array_dim3_block_connections_received + tmp_thread_global_index);
        
            ptr_array_derivatives_paramters_received[tmp_thread_global_index * (number_connections_received + 1u) + number_connections_received] += tmp_error; // Bias.
        }

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Parallel_Neurons(size_t const number_neurons_received,
                                                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                                                    size_t const total_parameters_allocated_received,
                                                                                                                                                                                    T *const ptr_array_derivatives_parameters_received,
                                                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                    struct CUDA_Layer *const ptr_layer_it_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    LAUNCH_KERNEL_POINTER_1D(CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<T>,
                                                      ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                      ptr_layer_it_received->ptr_dim3_block_neurons,
                                                      0_zu,
                                                      number_neurons_received - 1u, // Subtract bias.
                                                      number_neurons_received,
                                                      number_connections_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                      ptr_array_derivatives_parameters_received + blockIdx.x * total_parameters_allocated_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                      ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                      tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                      tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections)
}

template<typename T>
__global__ void kernel__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                                                    size_t const total_parameters_allocated_received,
                                                                                                                                                                                    T *const ptr_array_derivatives_parameters_received,
                                                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                    struct CUDA_Layer *const ptr_layer_it_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);
    
    if(tmp_thread_global_index < size_received)
    {
        LAUNCH_KERNEL_POINTER_1D(CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<T>,
                                                          ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                          ptr_layer_it_received->ptr_dim3_block_neurons,
                                                          0_zu,
                                                          number_neurons_received - 1u, // Subtract bias.
                                                          number_neurons_received,
                                                          number_connections_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                          ptr_array_derivatives_parameters_received + blockIdx.x * total_parameters_allocated_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                          ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                          tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                          tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections)
    }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                                size_t const total_parameters_allocated_received,
                                                                                                                                                                                                T *const ptr_array_derivatives_parameters_received,
                                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                struct CUDA_Layer *const ptr_layer_it_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    // KERNEL LAUNCH
    //    1: Launching do-while elements.
    if(ptr_layer_it_received->ptr_dim3_grid_neurons->x * ptr_layer_it_received->ptr_dim3_block_neurons->x < number_neurons_received - 1u) // Subtract bias.
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel_while__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons, *ptr_layer_it_received->ptr_dim3_block_neurons >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                      number_neurons_received,
                                                                                                                                                                                                                                                                                                                                                      number_connections_received,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                                      ptr_array_derivatives_parameters_received + blockIdx.x * total_parameters_allocated_received,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                                      ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    //    2: Launching size condition.
    else if(ptr_layer_it_received->ptr_dim3_grid_neurons->x * ptr_layer_it_received->ptr_dim3_block_neurons->x > number_neurons_received - 1u) // Subtract bias.
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons, *ptr_layer_it_received->ptr_dim3_block_neurons >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                             number_neurons_received,
                                                                                                                                                                                                                                                                                                                                             number_connections_received,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                             ptr_array_derivatives_parameters_received + blockIdx.x * total_parameters_allocated_received,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                             ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    //    3: Standard.
    else
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons, *ptr_layer_it_received->ptr_dim3_block_neurons >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                          number_neurons_received,
                                                                                                                                                                                                                                                                                                                                          number_connections_received,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                          ptr_array_derivatives_parameters_received + blockIdx.x * total_parameters_allocated_received,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                          ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    // |END| KERNEL LAUNCH |END|
}

template<typename T>
__global__ void kernel__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Serialize_Neurons(size_t const number_neurons_received,
                                                                                                                                                                                         size_t const number_connections_received,
                                                                                                                                                                                         size_t const total_parameters_allocated_received,
                                                                                                                                                                                         bool const *ptr_array_mask_dropout_received,
                                                                                                                                                                                         T *const ptr_array_derivatives_parameters_received,
                                                                                                                                                                                         T const *const ptr_array_neuron_units_errors_received,
                                                                                                                                                                                         T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                         struct dim3 const *ptr_array_dim3_grid_connections_received,
                                                                                                                                                                                         struct dim3 const *ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T_ const *const tmp_ptr_array_previous_layer_neurons_values(ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u)), // Add bias.
                 *tmp_ptr_array_errors(ptr_array_neuron_units_errors_received + tmp_thread_global_index * number_neurons_received),
                 *const tmp_ptr_array_errors_end(tmp_ptr_array_errors + number_neurons_received - 1u); // Subtract bias.
    T_ *tmp_ptr_array_derivatives_parameters(ptr_array_derivatives_parameters_received + tmp_thread_global_index * total_parameters_allocated_received),
         tmp_error;

    for(; tmp_ptr_array_errors != tmp_ptr_array_errors_end; ++tmp_ptr_array_errors,
                                                                                    ++ptr_array_mask_dropout_received,
                                                                                    ++ptr_array_dim3_grid_connections_received,
                                                                                    ++ptr_array_dim3_block_connections_received,
                                                                                    tmp_ptr_array_derivatives_parameters += number_connections_received + 1u) // Add bias.
    {
        if(*ptr_array_mask_dropout_received)
        {
            tmp_error = *tmp_ptr_array_errors;

            Multiply::FMAC_X_YX_1D<T_>(number_connections_received,
                                                            tmp_ptr_array_derivatives_parameters,
                                                            tmp_error,
                                                            tmp_ptr_array_previous_layer_neurons_values,
                                                            ptr_array_dim3_grid_connections_received,
                                                            ptr_array_dim3_block_connections_received);
            
            tmp_ptr_array_derivatives_parameters[number_connections_received] += tmp_error; // Bias.
        }
    }
}

template<typename T>
__global__ void kernel__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                                        size_t const total_parameters_allocated_received,
                                                                                                                                                                                        bool const *ptr_array_mask_dropout_received,
                                                                                                                                                                                        T *const ptr_array_derivatives_parameters_received,
                                                                                                                                                                                        T const *const ptr_array_neuron_units_errors_received,
                                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                        struct dim3 const *ptr_array_dim3_grid_connections_received,
                                                                                                                                                                                        struct dim3 const *ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T_ const *tmp_ptr_array_previous_layer_neurons_values,
                 *tmp_ptr_array_errors,
                 *tmp_ptr_array_errors_end;
    T_ *tmp_ptr_array_derivatives_parameters,
         tmp_error;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_previous_layer_neurons_values = ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u); // Add bias.

        tmp_ptr_array_errors = ptr_array_neuron_units_errors_received + tmp_thread_global_index * number_neurons_received;
        tmp_ptr_array_errors_end = tmp_ptr_array_errors + number_neurons_received - 1_zu; // Subtract bias.
        
        tmp_ptr_array_derivatives_parameters = ptr_array_derivatives_parameters_received + tmp_thread_global_index * total_parameters_allocated_received;

        for(; tmp_ptr_array_errors != tmp_ptr_array_errors_end; ++tmp_ptr_array_errors,
                                                                                        ++ptr_array_mask_dropout_received,
                                                                                        ++ptr_array_dim3_grid_connections_received,
                                                                                        ++ptr_array_dim3_block_connections_received,
                                                                                        tmp_ptr_array_derivatives_parameters += number_connections_received + 1u) // Add bias.
        {
            if(*ptr_array_mask_dropout_received)
            {
                tmp_error = *tmp_ptr_array_errors;

                Multiply::FMAC_X_YX_1D<T_>(number_connections_received,
                                                              tmp_ptr_array_derivatives_parameters,
                                                              tmp_error,
                                                              tmp_ptr_array_previous_layer_neurons_values,
                                                              ptr_array_dim3_grid_connections_received,
                                                              ptr_array_dim3_block_connections_received);
            
                tmp_ptr_array_derivatives_parameters[number_connections_received] += tmp_error; // Bias.
            }
        }
    }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                                size_t const total_parameters_allocated_received,
                                                                                                                                                                                                bool const *const ptr_array_mask_dropout_received,
                                                                                                                                                                                                T *const ptr_array_derivatives_parameters_received,
                                                                                                                                                                                                T const *const ptr_array_neuron_units_errors_received,
                                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_grid_index(blockIdx.x * blockDim.x + threadIdx.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    bool const *tmp_ptr_array_mask_dropout;

    T_ const *tmp_ptr_array_previous_layer_neurons_values,
                 *tmp_ptr_array_errors,
                 *tmp_ptr_array_errors_end;
    T_ *tmp_ptr_array_derivatives_parameters,
         tmp_error;

    struct dim3 const *tmp_ptr_array_dim3_grid_connections,
                              *tmp_ptr_array_dim3_block_connections;
    
    // Loop through each sample.
    do
    {
        tmp_ptr_array_mask_dropout = ptr_array_mask_dropout_received;

        tmp_ptr_array_previous_layer_neurons_values = ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u); // Add bias.

        tmp_ptr_array_errors = ptr_array_neuron_units_errors_received + tmp_thread_global_index * number_neurons_received;
        tmp_ptr_array_errors_end = tmp_ptr_array_errors + number_neurons_received - 1_zu; // Subtract bias.
        
        tmp_ptr_array_derivatives_parameters = ptr_array_derivatives_parameters_received + tmp_thread_grid_index * total_parameters_allocated_received;

        tmp_ptr_array_dim3_grid_connections = ptr_array_dim3_grid_connections_received;
        tmp_ptr_array_dim3_block_connections = ptr_array_dim3_block_connections_received;

        for(; tmp_ptr_array_errors != tmp_ptr_array_errors_end; ++tmp_ptr_array_errors,
                                                                                        ++tmp_ptr_array_mask_dropout,
                                                                                        ++tmp_ptr_array_dim3_grid_connections,
                                                                                        ++tmp_ptr_array_dim3_block_connections,
                                                                                        tmp_ptr_array_derivatives_parameters += number_connections_received + 1u) // Add bias.
        {
            if(*tmp_ptr_array_mask_dropout)
            {
                tmp_error = *tmp_ptr_array_errors;

                Multiply::FMAC_X_YX_1D<T_>(number_connections_received,
                                                              tmp_ptr_array_derivatives_parameters,
                                                              tmp_error,
                                                              tmp_ptr_array_previous_layer_neurons_values,
                                                              tmp_ptr_array_dim3_grid_connections,
                                                              tmp_ptr_array_dim3_block_connections);
            
                tmp_ptr_array_derivatives_parameters[number_connections_received] += tmp_error; // Bias.
            }
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Update_Derivative_Weight__FC_to_FC__Dropout(bool &ref_synchronized_received,
                                                                                                                                    size_t const batch_size_received,
                                                                                                                                    struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                    struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                                    struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                    struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    size_t tmp_data_index;
    
    struct CUDA_Neuron const *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units),
                                                      *const tmp_ptr_previous_layer_first_neuron(ptr_previous_layer_it_received->ptr_array_neuron_units);
    
    // TODO: Remove bias term in nConnections.
    // By subtracting the bias the variable "ptr_dim3_grid_connections" become a false dimension.
    size_t const tmp_number_connections(*tmp_ptr_layer_it_first_neuron->ptr_number_forward_connections - 1u), // Subtract bias.
                                tmp_number_neuron_units(*ptr_layer_it_received->ptr_number_neurons);

    bool const *tmp_ptr_array_mask_dropout;

    T_ const *tmp_ptr_array_previous_layer_neurons_values,
                 *tmp_ptr_array_errors,
                 *tmp_ptr_array_errors_end;
    T_ *tmp_ptr_array_derivatives_parameters,
         tmp_error;

    struct dim3 const *tmp_ptr_array_dim3_grid_connections,
                              *tmp_ptr_array_dim3_block_connections;
    
    // Condition to enter into dynamic parallelisme of each sample.
    if(USE_PARALLEL && batch_size_received >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;
        
        // Condition to enter into dynamic parallelisme of each sample and neurons.
        if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
        {
            LAUNCH_KERNEL_POINTER_1D(CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Parallel_Neurons<T_>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_zu,
                                                              batch_size_received,
                                                              tmp_number_neuron_units,
                                                              tmp_number_connections,
                                                              this->total_parameters_allocated,
                                                              this->ptr_array_derivatives_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_previous_layer_first_neuron->ptr_array_values,
                                                              ptr_layer_it_received)
        }
        // Condition to enter into dynamic parallelisme of each sample.
        else
        {
            LAUNCH_KERNEL_POINTER_1D(CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Serialize_Neurons<T_>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_zu,
                                                              batch_size_received,
                                                              tmp_number_neuron_units,
                                                              tmp_number_connections,
                                                              this->total_parameters_allocated,
                                                              tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                              this->ptr_array_derivatives_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_layer_it_first_neuron->ptr_array_errors,
                                                              tmp_ptr_previous_layer_first_neuron->ptr_array_values,
                                                              tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                              tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections)
        }
    }
    // Condition to enter into dynamic parallelisme of each neurons.
    else if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;
        
        // KERNEL LAUNCH
        //    1: Launching do-while elements.
        if(ptr_layer_it_received->ptr_dim3_grid_neurons_DP->x * ptr_layer_it_received->ptr_dim3_block_neurons_DP->x < tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel_while__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP, *ptr_layer_it_received->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                                    tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                                    tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                                                    this->ptr_array_derivatives_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);
            }
        }
        //    2: Launching size condition.
        else if(ptr_layer_it_received->ptr_dim3_grid_neurons_DP->x * ptr_layer_it_received->ptr_dim3_block_neurons_DP->x > tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP, *ptr_layer_it_received->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                            tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                            tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                                            this->ptr_array_derivatives_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);
            }
        }
        //    3: Standard.
        else
        {
            // Loop through each sample.
            for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__CNeural_Network__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP, *ptr_layer_it_received->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                            tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                                            this->ptr_array_derivatives_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);
            }
        }
        // |END| KERNEL LAUNCH |END|
    }
    // If we don't enter into dynamic parallelisme, we serialize the computation.
    else
    {
        // Loop through each sample.
        for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
        {
            tmp_ptr_array_mask_dropout = tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli;

            tmp_ptr_array_previous_layer_neurons_values = tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u); // Add bias.

            tmp_ptr_array_errors = tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units;
            tmp_ptr_array_errors_end = tmp_ptr_array_errors + tmp_number_neuron_units - 1_zu; // Subtract bias.
            
            tmp_ptr_array_derivatives_parameters = this->ptr_array_derivatives_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index;

            tmp_ptr_array_dim3_grid_connections = tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections;
            tmp_ptr_array_dim3_block_connections = tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections;

            for(; tmp_ptr_array_errors != tmp_ptr_array_errors_end; ++tmp_ptr_array_errors,
                                                                                            ++tmp_ptr_array_mask_dropout,
                                                                                            ++tmp_ptr_array_dim3_grid_connections,
                                                                                            ++tmp_ptr_array_dim3_block_connections,
                                                                                            tmp_ptr_array_derivatives_parameters += tmp_number_connections + 1u) // Add bias.
            {
                if(*tmp_ptr_array_mask_dropout)
                {
                    tmp_error = *tmp_ptr_array_errors;

                    Multiply::FMAC_X_YX_1D<T_>(tmp_number_connections,
                                                                  tmp_ptr_array_derivatives_parameters,
                                                                  tmp_error,
                                                                  tmp_ptr_array_previous_layer_neurons_values,
                                                                  tmp_ptr_array_dim3_grid_connections,
                                                                  tmp_ptr_array_dim3_block_connections);
                
                    tmp_ptr_array_derivatives_parameters[tmp_number_connections] += tmp_error; // Bias.
                }
            }
        }
    }
}
