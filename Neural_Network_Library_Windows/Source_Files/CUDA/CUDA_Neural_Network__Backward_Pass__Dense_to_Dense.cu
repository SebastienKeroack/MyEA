#include <Enums/Enum_Type_Activation_Functions.hpp>

#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Reduce.cuh>
#include <CUDA/CUDA_Multiply_1D.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons(size_t const data_index_received,
                                                                                                                                              size_t const number_neurons_received,
                                                                                                                                              size_t const next_layer_number_neurons_received,
                                                                                                                                              size_t const neurons_total_reduce_error_size_received,
                                                                                                                                              T const *const ptr_array_layer_it_summations_received,
                                                                                                                                              T const *const ptr_array_layer_it_values_received,
                                                                                                                                              T *const ptr_array_layer_it_errors_received,
                                                                                                                                              T **const ptr_array_layer_it_reduce_errors_received,
                                                                                                                                              T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                              T const *const ptr_array_next_layer_errors_received,
                                                                                                                                              enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                              struct dim3 const *const ptr_array_dim3_grid_reduce_errors_received,
                                                                                                                                              struct dim3 const *const ptr_array_dim3_block_reduce_errors_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T *const tmp_ptr_array_reduce_error(ptr_array_layer_it_reduce_errors_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_error_size_received);
    
    Reduce::Reduce_XX<T>(next_layer_number_neurons_received,
                                        number_neurons_received,
                                        tmp_ptr_array_reduce_error,
                                        ptr_array_next_layer_parameters_received + tmp_thread_global_index * next_layer_number_neurons_received,
                                        ptr_array_next_layer_errors_received,
                                        ptr_array_dim3_grid_reduce_errors_received + tmp_thread_global_index,
                                        ptr_array_dim3_block_reduce_errors_received + tmp_thread_global_index);

    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced error of the neuron.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    ptr_array_layer_it_errors_received[tmp_thread_global_index] = *tmp_ptr_array_reduce_error; // Reduced error.

    ptr_array_layer_it_errors_received[tmp_thread_global_index] *= Activation_Derived(1_T,
                                                                                                                           ptr_array_layer_it_summations_received[tmp_thread_global_index],
                                                                                                                           ptr_array_layer_it_values_received[tmp_thread_global_index],
                                                                                                                           ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                              size_t const data_index_received,
                                                                                                                                              size_t const number_neurons_received,
                                                                                                                                              size_t const next_layer_number_neurons_received,
                                                                                                                                              size_t const neurons_total_reduce_error_size_received,
                                                                                                                                              T  const *const ptr_array_layer_it_summations_received,
                                                                                                                                              T const *const ptr_array_layer_it_values_received,
                                                                                                                                              T *const ptr_array_layer_it_errors_received,
                                                                                                                                              T **const ptr_array_layer_it_reduce_errors_received,
                                                                                                                                              T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                              T const *const ptr_array_next_layer_errors_received,
                                                                                                                                              enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                              struct dim3 const *const ptr_array_dim3_grid_reduce_errors_received,
                                                                                                                                              struct dim3 const *const ptr_array_dim3_block_reduce_errors_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T *tmp_ptr_array_reduce_error;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_reduce_error = ptr_array_layer_it_reduce_errors_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_error_size_received;
        
        Reduce::Reduce_XX<T>(next_layer_number_neurons_received,
                                            number_neurons_received,
                                            tmp_ptr_array_reduce_error,
                                            ptr_array_next_layer_parameters_received + tmp_thread_global_index * next_layer_number_neurons_received,
                                            ptr_array_next_layer_errors_received,
                                            ptr_array_dim3_grid_reduce_errors_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_errors_received + tmp_thread_global_index);
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced error of the neuron.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_layer_it_errors_received[tmp_thread_global_index] = *tmp_ptr_array_reduce_error; // Reduced error.
        
        ptr_array_layer_it_errors_received[tmp_thread_global_index] *= Activation_Derived(1_T,
                                                                                                                               ptr_array_layer_it_summations_received[tmp_thread_global_index],
                                                                                                                               ptr_array_layer_it_values_received[tmp_thread_global_index],
                                                                                                                               ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
    }
}

template<typename T>
__global__ void kernel_while__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                        size_t const data_index_received,
                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                        size_t const next_layer_number_neurons_received,
                                                                                                                                                        size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                        T const *const ptr_array_layer_it_summations_received,
                                                                                                                                                        T const *const ptr_array_layer_it_values_received,
                                                                                                                                                        T *const ptr_array_layer_it_errors_received,
                                                                                                                                                        T **const ptr_array_layer_it_reduce_errors_received,
                                                                                                                                                        T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                                        T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                        enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_reduce_errors_received,
                                                                                                                                                        struct dim3 const *const ptr_array_dim3_block_reduce_errors_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    // Loop through each neurons.
    do
    {
        Reduce::Reduce_XX<T>(next_layer_number_neurons_received,
                                            number_neurons_received,
                                            ptr_array_layer_it_reduce_errors_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_error_size_received,
                                            ptr_array_next_layer_parameters_received + tmp_thread_global_index * next_layer_number_neurons_received,
                                            ptr_array_next_layer_errors_received,
                                            ptr_array_dim3_grid_reduce_errors_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_errors_received + tmp_thread_global_index);
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced error of the neuron.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // Reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop through each neurons.
    do
    {
        ptr_array_layer_it_errors_received[tmp_thread_global_index] = *(ptr_array_layer_it_reduce_errors_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_error_size_received); // Reduced error.
        
        ptr_array_layer_it_errors_received[tmp_thread_global_index] *= Activation_Derived(1_T,
                                                                                                                               ptr_array_layer_it_summations_received[tmp_thread_global_index],
                                                                                                                               ptr_array_layer_it_values_received[tmp_thread_global_index],
                                                                                                                               ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Parallel_Batch__Parallel_Neurons(size_t const number_neurons_received,
                                                                                                                                            size_t const next_layer_number_neurons_received,
                                                                                                                                            size_t const neurons_total_reduce_error_size_received,
                                                                                                                                            T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                            T const *const ptr_array_next_layer_errors_received,
                                                                                                                                            struct CUDA_Layer *const ptr_layer_it_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T>,
                                                      ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                      ptr_layer_it_received->ptr_dim3_block_neurons,
                                                      0_zu,
                                                      number_neurons_received - 1u, // Subtract bias.
                                                      tmp_thread_global_index,
                                                      number_neurons_received,
                                                      next_layer_number_neurons_received,
                                                      neurons_total_reduce_error_size_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                      ptr_array_next_layer_parameters_received,
                                                      ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                      tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error)
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                            size_t const next_layer_number_neurons_received,
                                                                                                                                            size_t const neurons_total_reduce_error_size_received,
                                                                                                                                            T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                            T const *const ptr_array_next_layer_errors_received,
                                                                                                                                            struct CUDA_Layer *const ptr_layer_it_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);
    
    if(tmp_thread_global_index < size_received)
    {
        LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T>,
                                                          ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                          ptr_layer_it_received->ptr_dim3_block_neurons,
                                                          0_zu,
                                                          number_neurons_received - 1u, // Subtract bias.
                                                          tmp_thread_global_index,
                                                          number_neurons_received,
                                                          next_layer_number_neurons_received,
                                                          neurons_total_reduce_error_size_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                          ptr_array_next_layer_parameters_received,
                                                          ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                          tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error)
    }
}

template<typename T>
__global__ void kernel_while__Backward_Pass__FC_to_FC__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                     size_t const number_neurons_received,
                                                                                                                                                     size_t const next_layer_number_neurons_received,
                                                                                                                                                     size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                     T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                                     T const *const ptr_array_next_layer_errors_received,
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
            kernel_while__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons, *ptr_layer_it_received->ptr_dim3_block_neurons >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                        tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                        number_neurons_received,
                                                                                                                                                                                                                                                                                                        next_layer_number_neurons_received,
                                                                                                                                                                                                                                                                                                        neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                        ptr_array_next_layer_parameters_received,
                                                                                                                                                                                                                                                                                                        ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
        
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
            kernel__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons, *ptr_layer_it_received->ptr_dim3_block_neurons >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                number_neurons_received,
                                                                                                                                                                                                                                                                                                next_layer_number_neurons_received,
                                                                                                                                                                                                                                                                                                neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                ptr_array_next_layer_parameters_received,
                                                                                                                                                                                                                                                                                                ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
        
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
            kernel__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons, *ptr_layer_it_received->ptr_dim3_block_neurons >>> (tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                number_neurons_received,
                                                                                                                                                                                                                                                                                                next_layer_number_neurons_received,
                                                                                                                                                                                                                                                                                                neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                ptr_array_next_layer_parameters_received,
                                                                                                                                                                                                                                                                                                ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    // |END| KERNEL LAUNCH |END|
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Parallel_Batch__Serialize_Neurons(size_t const number_neurons_received,
                                                                                                                                            size_t const next_layer_number_neurons_received,
                                                                                                                                            size_t const neurons_total_reduce_error_size_received,
                                                                                                                                            T const *ptr_array_next_layer_parameters_received,
                                                                                                                                            T const *const ptr_array_next_layer_errors_received,
                                                                                                                                            struct CUDA_Neuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                            struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T_ const *const tmp_ptr_next_layer_errors(ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u)); // Add bias.

    struct CUDA_Neuron *tmp_ptr_neuron_unit_it;

    // Loop through each neurons.
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                            ptr_array_next_layer_parameters_received += next_layer_number_neurons_received)
    {
        Reduce::Reduce_XX<T_>(next_layer_number_neurons_received,
                                                number_neurons_received,
                                                *tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received,
                                                ptr_array_next_layer_parameters_received,
                                                tmp_ptr_next_layer_errors,
                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error,
                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error);
    }

    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    // Loop through each neurons.
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
    {
        tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received); // Reduced error.

        tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] *= Activation_Derived(*tmp_ptr_neuron_unit_it->ptr_activation_steepness,
                                                                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                                                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                                                                                                                                                *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
    }
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                            size_t const next_layer_number_neurons_received,
                                                                                                                                            size_t const neurons_total_reduce_error_size_received,
                                                                                                                                            T const *ptr_array_next_layer_parameters_received,
                                                                                                                                            T const *const ptr_array_next_layer_errors_received,
                                                                                                                                            struct CUDA_Neuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                            struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T_ const *tmp_ptr_next_layer_errors;
    
    struct CUDA_Neuron *tmp_ptr_neuron_unit_it;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_next_layer_errors = ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u); // Add bias.

        // Loop through each neurons.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                             ptr_array_next_layer_parameters_received += next_layer_number_neurons_received)
        {
            Reduce::Reduce_XX<T_>(next_layer_number_neurons_received,
                                                 number_neurons_received,
                                                 *tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received,
                                                 ptr_array_next_layer_parameters_received,
                                                 tmp_ptr_next_layer_errors,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error);
        }
    }

    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    if(tmp_thread_global_index < size_received)
    {
        // Loop through each neurons.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received); // Reduced error.

            tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] *= Activation_Derived(*tmp_ptr_neuron_unit_it->ptr_activation_steepness,
                                                                                                                                                                    tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                                                                                                                                                    tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                                                                                                                                                    *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
        }
    }
}

template<typename T>
__global__ void kernel_while__Backward_Pass__FC_to_FC__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                    size_t const next_layer_number_neurons_received,
                                                                                                                                                    size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                    T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                                    T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                    struct CUDA_Neuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                    struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T_ const *tmp_ptr_next_layer_parameters,
                 *tmp_ptr_next_layer_errors;
    
    struct CUDA_Neuron *tmp_ptr_neuron_unit_it;

    // Loop through each sample.
    do
    {
        tmp_ptr_next_layer_parameters = ptr_array_next_layer_parameters_received;
        tmp_ptr_next_layer_errors = ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u); // Add bias.

        // Loop through each neurons.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                             tmp_ptr_next_layer_parameters += next_layer_number_neurons_received)
        {
            Reduce::Reduce_XX<T_>(next_layer_number_neurons_received,
                                                 number_neurons_received,
                                                 *tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received,
                                                 tmp_ptr_next_layer_parameters,
                                                 tmp_ptr_next_layer_errors,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error);
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);

    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    // Reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Loop through each sample.
    do
    {
        // Loop through each neurons.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received); // Reduced error.

            tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] *= Activation_Derived(*tmp_ptr_neuron_unit_it->ptr_activation_steepness,
                                                                                                                                                                    tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                                                                                                                                                    tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                                                                                                                                                    *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Backward_Pass__FC_to_FC(bool &ref_synchronized_received,
                                                                                                        size_t const batch_size_received,
                                                                                                        struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                        struct CUDA_Layer *const ptr_next_layer_received,
                                                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    size_t const tmp_number_neuron_units(*ptr_layer_it_received->ptr_number_neurons),
                                tmp_next_layer_number_neurons(*ptr_next_layer_received->ptr_number_neurons - 1u); // Subtract bias.
    size_t tmp_data_index;
    
    struct CUDA_Neuron const *const tmp_ptr_next_layer_first_neuron(ptr_next_layer_received->ptr_array_neuron_units);
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units),
                                             *tmp_ptr_neuron_unit_it;
    
    T_ const *tmp_ptr_next_layer_parameters,
                  *tmp_ptr_next_layer_errors;
    
    // Condition to enter into dynamic parallelisme of each sample.
    if(USE_PARALLEL && batch_size_received >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;
        
        // Condition to enter into dynamic parallelisme of each sample and neurons.
        if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
        {
            LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Parallel_Batch__Parallel_Neurons<T_>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_zu,
                                                              batch_size_received,
                                                              tmp_number_neuron_units,
                                                              tmp_next_layer_number_neurons,
                                                              this->neurons_total_reduce_error_size,
                                                              this->ptr_array_transposed_weights + *tmp_ptr_next_layer_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_next_layer_first_neuron->ptr_array_errors,
                                                              ptr_layer_it_received)
        }
        // Condition to enter into dynamic parallelisme of each sample.
        else
        {
            LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Parallel_Batch__Serialize_Neurons<T_>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_zu,
                                                              batch_size_received,
                                                              tmp_number_neuron_units,
                                                              tmp_next_layer_number_neurons,
                                                              this->neurons_total_reduce_error_size,
                                                              this->ptr_array_transposed_weights + *tmp_ptr_next_layer_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_next_layer_first_neuron->ptr_array_errors,
                                                              tmp_ptr_layer_it_first_neuron,
                                                              ptr_layer_it_received->ptr_last_neuron_unit - 1u) // Subtract bias.
        }
    }
    // Condition to enter into dynamic parallelisme of each neurons.
    if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
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
                kernel_while__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP, *ptr_layer_it_received->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                            tmp_data_index,
                                                                                                                                                                                                                                                                                                                            tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                            tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                            this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                                            this->ptr_array_transposed_weights + *tmp_ptr_next_layer_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_next_layer_first_neuron->ptr_array_errors + tmp_data_index * (tmp_next_layer_number_neurons + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
            }
        }
        //    2: Launching size condition.
        else if(ptr_layer_it_received->ptr_dim3_grid_neurons_DP->x * ptr_layer_it_received->ptr_dim3_block_neurons_DP->x > tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP, *ptr_layer_it_received->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                    tmp_data_index,
                                                                                                                                                                                                                                                                                                                    tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                    this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                                    this->ptr_array_transposed_weights + *tmp_ptr_next_layer_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_next_layer_first_neuron->ptr_array_errors + tmp_data_index * (tmp_next_layer_number_neurons + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
            }
        }
        //    3: Standard.
        else
        {
            // Loop through each sample.
            for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP, *ptr_layer_it_received->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                    tmp_data_index,
                                                                                                                                                                                                                                                                                                                    tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                    this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                                    this->ptr_array_transposed_weights + *tmp_ptr_next_layer_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_next_layer_first_neuron->ptr_array_errors + tmp_data_index * (tmp_next_layer_number_neurons + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
            }
        }
        // |END| KERNEL LAUNCH |END|
    }
    // If we don't enter into dynamic parallelisme, we serialize the computation.
    else
    {
        struct CUDA_Neuron const *const tmp_ptr_last_neuron_unit(ptr_layer_it_received->ptr_last_neuron_unit - 1);

        // Synchronisation before using the transposed error of the layer.
        CUDA__Device_Synchronise(ref_synchronized_received, MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);
        
        // Loop through each sample.
        for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
        {
            tmp_ptr_next_layer_parameters = this->ptr_array_transposed_weights + *tmp_ptr_next_layer_first_neuron->ptr_first_forward_connection_index;
            tmp_ptr_next_layer_errors = tmp_ptr_next_layer_first_neuron->ptr_array_errors + tmp_data_index * (tmp_next_layer_number_neurons + 1u); // Add bias.

            // Loop through each neurons.
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                         tmp_ptr_next_layer_parameters += tmp_next_layer_number_neurons)
            {
                Reduce::Reduce_XX<T_>(tmp_next_layer_number_neurons,
                                                      tmp_number_neuron_units,
                                                      *tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_data_index * this->neurons_total_reduce_error_size,
                                                      tmp_ptr_next_layer_parameters,
                                                      tmp_ptr_next_layer_errors,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error);
            }
        }

        // Do we need to synchronise? Based on "Reduce_XX" Function.
        // => Synchronisation before using the reduced error of the neuron.
        if(tmp_next_layer_number_neurons >= warpSize * 2u) { CUDA__Check_Error(); }
        
        // Loop through each sample.
        for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
        {
            // Loop through each neurons.
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
            {
                tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_data_index * tmp_number_neuron_units] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_data_index * this->neurons_total_reduce_error_size); // Reduced error.

                tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_data_index * tmp_number_neuron_units] *= Activation_Derived(*tmp_ptr_neuron_unit_it->ptr_activation_steepness,
                                                                                                                                                     tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units],
                                                                                                                                                     tmp_ptr_neuron_unit_it->ptr_array_values[tmp_data_index * tmp_number_neuron_units],
                                                                                                                                                     *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
            }
        }
    }
}
