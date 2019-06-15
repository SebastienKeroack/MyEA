#include <Enums/Enum_Type_Activation_Functions.hpp>

#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Reduce.cuh>
#include <CUDA/CUDA_Multiply_1D.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons(size_t const data_index_received,
                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                            size_t const number_connections_received,
                                                                                                                                            size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                            T *const ptr_array_layer_it_summations_received,
                                                                                                                                            T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                            T *const ptr_array_layer_it_values_received,
                                                                                                                                            T const *const ptr_array_parameters_received,
                                                                                                                                            T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                            enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                            struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                            struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u), // Add bias.
                               tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *const tmp_ptr_array_parameters(ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased);
    T *const tmp_ptr_array_reduce_summation(ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received);

    Reduce::Reduce_XX<T>(number_connections_received,
                                        number_neurons_received,
                                        tmp_ptr_array_reduce_summation,
                                        tmp_ptr_array_parameters,
                                        ptr_array_previous_layer_outputs_received,
                                        ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                        ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);

    ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    ptr_array_layer_it_summations_received[tmp_thread_global_index] += *tmp_ptr_array_reduce_summation; // Reduced summation.
    
    Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                            ptr_array_layer_it_summations_received[tmp_thread_global_index],
                            ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                            size_t const data_index_received,
                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                            size_t const number_connections_received,
                                                                                                                                            size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                            T *const ptr_array_layer_it_summations_received,
                                                                                                                                            T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                            T *const ptr_array_layer_it_values_received,
                                                                                                                                            T const *const ptr_array_parameters_received,
                                                                                                                                            T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                            enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                            struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                            struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u), // Add bias.
                               tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *const tmp_ptr_array_parameters(ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased);
    T *tmp_ptr_array_reduce_summation;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_reduce_summation = ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received;

        PRINT_FORMAT("tmp_thread_global_index: %u" NEW_LINE, tmp_thread_global_index);
        Reduce::Reduce_XX<T>(number_connections_received,
                                            number_neurons_received,
                                            tmp_ptr_array_reduce_summation,
                                            tmp_ptr_array_parameters,
                                            ptr_array_previous_layer_outputs_received,
                                            ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);

        ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_layer_it_summations_received[tmp_thread_global_index] += *tmp_ptr_array_reduce_summation; // Reduced summation.
        
        Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                               ptr_array_layer_it_summations_received[tmp_thread_global_index],
                               ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                    size_t const data_index_received,
                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                    size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                    T *const ptr_array_layer_it_summations_received,
                                                                                                                                                    T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                                    T *const ptr_array_layer_it_values_received,
                                                                                                                                                    T const *const ptr_array_parameters_received,
                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                    struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                                    struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u); // Add bias.
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *tmp_ptr_array_parameters;

    // Loop through each neurons.
    do
    {
        tmp_ptr_array_parameters = ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased;

        Reduce::Reduce_XX<T>(number_connections_received,
                                            number_neurons_received,
                                            ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received,
                                            tmp_ptr_array_parameters,
                                             ptr_array_previous_layer_outputs_received,
                                            ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);
        
        ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // Reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop through each neurons.
    do
    {
        ptr_array_layer_it_summations_received[tmp_thread_global_index] += *(ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received); // Reduced summation.
        
        Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                               ptr_array_layer_it_summations_received[tmp_thread_global_index],
                               ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Parallel_Batch__Serialize_Neurons(size_t const number_neurons_received,
                                                                                                                                            size_t const number_connections_received,
                                                                                                                                            size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                            T const *ptr_array_parameters_received,
                                                                                                                                            T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                            struct CUDA_Neuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                            struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T_ const *const tmp_ptr_array_previous_layer_outputs(ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u)); // Add bias.
    
    struct CUDA_Neuron *tmp_ptr_neuron_unit_it;
    
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                         ptr_array_parameters_received += number_connections_received + 1u) // Add bias.
    {
        Reduce::Reduce_XX<T_>(number_connections_received,
                                              number_neurons_received,
                                              *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                              ptr_array_parameters_received,
                                              tmp_ptr_array_previous_layer_outputs,
                                              tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                              tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);

        tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = ptr_array_parameters_received[number_connections_received]; // Bias.
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    // Loop through each neurons for retrieve reduced summation and then do the activation function.
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
    {
        tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] += *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received); // Reduced summation.
        
        Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
    }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                        size_t const number_connections_received,
                                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                        T const *ptr_array_parameters_received,
                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                        struct CUDA_Neuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                        struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T_ const *tmp_ptr_array_previous_layer_outputs;
    
    struct CUDA_Neuron *tmp_ptr_neuron_unit_it;
    
    if(tmp_thread_global_index < size_received)
    {
        for(tmp_ptr_array_previous_layer_outputs = ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
            tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                             ptr_array_parameters_received += number_connections_received + 1u) // Add bias.
        {
            Reduce::Reduce_XX<T_>(number_connections_received,
                                                  number_neurons_received,
                                                  *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                                  ptr_array_parameters_received,
                                                  tmp_ptr_array_previous_layer_outputs,
                                                  tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                  tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);

            tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = ptr_array_parameters_received[number_connections_received]; // Bias.
        }
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    if(tmp_thread_global_index < size_received)
    {
        // Loop through each neurons for retrieve reduced summation and then do the activation function.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] += *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received); // Reduced summation.
        
            Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                   tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                   *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
        }
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                    size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                    T const *const ptr_array_parameters_received,
                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                    struct CUDA_Neuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                    struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T_ const *tmp_ptr_array_parameters,
                  *tmp_ptr_array_previous_layer_outputs;
    
    struct CUDA_Neuron *tmp_ptr_neuron_unit_it;

    // Loop through each sample.
    do
    {
        tmp_ptr_array_parameters = ptr_array_parameters_received;
        tmp_ptr_array_previous_layer_outputs = ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u); // Add bias.

        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                             tmp_ptr_array_parameters += number_connections_received + 1u) // Add bias.
        {
            Reduce::Reduce_XX<T_>(number_connections_received,
                                                  number_neurons_received,
                                                  *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                                  tmp_ptr_array_parameters,
                                                  tmp_ptr_array_previous_layer_outputs,
                                                  tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                  tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);

            tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    // Reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Loop through each sample.
    do
    {
        // Loop through each neurons for retrieve reduced summation and then do the activation function.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] += *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received); // Reduced summation.
        
            Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                   tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                   *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Parallel_Batch__Parallel_Neurons(size_t const number_neurons_received,
                                                                                                                                        size_t const number_connections_received,
                                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                        T const *const ptr_array_parameters_received,
                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                        struct CUDA_Layer *const ptr_layer_it_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T>,
                                                        ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                        ptr_layer_it_received->ptr_dim3_block_neurons,
                                                        0_zu,
                                                        number_neurons_received - 1u, // Subtract bias.
                                                        tmp_thread_global_index,
                                                        number_neurons_received,
                                                        number_connections_received,
                                                        neurons_total_reduce_summation_size_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                        ptr_array_parameters_received,
                                                        ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation)
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                        size_t const number_connections_received,
                                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                        T const *const ptr_array_parameters_received,
                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                        struct CUDA_Layer *const ptr_layer_it_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    if(tmp_thread_global_index < size_received)
    {
        LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T>,
                                                          ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                          ptr_layer_it_received->ptr_dim3_block_neurons,
                                                          0_zu,
                                                          number_neurons_received - 1u, // Subtract bias.
                                                          tmp_thread_global_index,
                                                          number_neurons_received,
                                                          number_connections_received,
                                                          neurons_total_reduce_summation_size_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                          ptr_array_parameters_received,
                                                          ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                          tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation)
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                T const *const ptr_array_parameters_received,
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
            kernel_while__Forward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons, *ptr_layer_it_received->ptr_dim3_block_neurons >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                        tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                        number_neurons_received,
                                                                                                                                                                                                                                                                                                        number_connections_received,
                                                                                                                                                                                                                                                                                                        neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                        ptr_array_parameters_received,
                                                                                                                                                                                                                                                                                                        ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
        
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
            kernel__Forward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons, *ptr_layer_it_received->ptr_dim3_block_neurons >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                number_neurons_received,
                                                                                                                                                                                                                                                                                                number_connections_received,
                                                                                                                                                                                                                                                                                                neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                ptr_array_parameters_received,
                                                                                                                                                                                                                                                                                                ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
        
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
            kernel__Forward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons, *ptr_layer_it_received->ptr_dim3_block_neurons >>> (tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                number_neurons_received,
                                                                                                                                                                                                                                                                                                number_connections_received,
                                                                                                                                                                                                                                                                                                neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                ptr_array_parameters_received,
                                                                                                                                                                                                                                                                                                ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    // |END| KERNEL LAUNCH |END|
}

__device__ void CUDA_Neural_Network::Forward_Pass__FC_to_FC(bool &ref_synchronized_received,
                                                                                                    size_t const batch_size_received,
                                                                                                    struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                    struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                    struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                    struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    size_t tmp_data_index;
    
    struct CUDA_Neuron const *const tmp_ptr_previous_layer_first_neuron(ptr_previous_layer_it_received->ptr_array_neuron_units);
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units),
                                             *tmp_ptr_neuron_unit_it;
    
    // TODO: Remove bias term in nConnections.
    // By subtracting the bias the variable "ptr_dim3_grid_connections" become a false dimension.
    size_t const tmp_number_connections(*tmp_ptr_layer_it_first_neuron->ptr_number_forward_connections - 1u), // Subtract bias.
                                tmp_number_neuron_units(*ptr_layer_it_received->ptr_number_neurons);

    T_ const *tmp_ptr_array_previous_layer_outputs,
                  *tmp_ptr_array_parameters;
    
    // Condition to enter into dynamic parallelisme of each sample.
    if(USE_PARALLEL && batch_size_received >= warpSize && ptr_layer_it_received == this->ptr_array_layers + 1)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;
        
        // Condition to enter into dynamic parallelisme of each sample and neurons.
        if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
        {
            LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Parallel_Batch__Parallel_Neurons<T_>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_zu,
                                                              batch_size_received,
                                                              tmp_number_neuron_units,
                                                              tmp_number_connections,
                                                              this->neurons_total_reduce_summation_size,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_previous_layer_first_neuron->ptr_array_values,
                                                              ptr_layer_it_received)
        }
        // Condition to enter into dynamic parallelisme of each sample.
        else
        {
            LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Parallel_Batch__Serialize_Neurons<T_>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_zu,
                                                              batch_size_received,
                                                              tmp_number_neuron_units,
                                                              tmp_number_connections,
                                                              this->neurons_total_reduce_summation_size,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_previous_layer_first_neuron->ptr_array_values,
                                                              tmp_ptr_layer_it_first_neuron,
                                                              ptr_layer_it_received->ptr_last_neuron_unit - 1) // Subtract bias.
        }
    }
    // Condition to enter into dynamic parallelisme of each neurons.
    else if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize || true)
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
                kernel_while__Forward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP, *ptr_layer_it_received->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                           tmp_data_index,
                                                                                                                                                                                                                                                                                                                           tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                           tmp_number_connections,
                                                                                                                                                                                                                                                                                                                           this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                           tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                           tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                           tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                           this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                           tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                           tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                           tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                           tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
            }
        }
        //    2: Launching size condition.
        else if(ptr_layer_it_received->ptr_dim3_grid_neurons_DP->x * ptr_layer_it_received->ptr_dim3_block_neurons_DP->x > tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__Forward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP, *ptr_layer_it_received->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                tmp_data_index,
                                                                                                                                                                                                                                                                                                                tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                tmp_number_connections,
                                                                                                                                                                                                                                                                                                                this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
            }
        }
        //    3: Standard.
        else
        {
            // Loop through each sample.
            for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__Forward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP, *ptr_layer_it_received->ptr_dim3_block_neurons_DP >>> (tmp_data_index,
                                                                                                                                                                                                                                                                                                                  tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                  tmp_number_connections,
                                                                                                                                                                                                                                                                                                                  this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                  this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                  tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                  tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
            }
        }
        // |END| KERNEL LAUNCH |END|
    }
    // If we don't enter into dynamic parallelisme, we serialize the computation.
    else
    {
        struct CUDA_Neuron const *const tmp_ptr_last_neuron_unit(ptr_layer_it_received->ptr_last_neuron_unit - 1); // Subtract bias.
        
        // Synchronize if needed to see the output of the previous layer.
        CUDA__Device_Synchronise(ref_synchronized_received, MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);
        
        // Loop through each sample.
        for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
        {
            tmp_ptr_array_parameters = this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index;
            tmp_ptr_array_previous_layer_outputs = tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u);

            // Loop through each neurons for doing a reduction of summation.
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                         tmp_ptr_array_parameters += tmp_number_connections + 1u) // Add bias.
            {
                Reduce::Reduce_XX<T_>(tmp_number_connections,
                                                      tmp_number_neuron_units,
                                                      *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_data_index * this->neurons_total_reduce_summation_size,
                                                      tmp_ptr_array_parameters,
                                                      tmp_ptr_array_previous_layer_outputs,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);

                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units] = tmp_ptr_array_parameters[tmp_number_connections]; // Bias.
            }
        }

        // Do we need to synchronise? Based on "Reduce_XX" Function.
        // => Synchronize if needed to see the summation reduced of the layer.
        if(tmp_number_connections >= warpSize * 2u) { CUDA__Check_Error(); }

        // Loop through each sample.
        for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
        {
            // Loop through each neurons for retrieve reduced summation and then do the activation function.
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
            {
                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units] += *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_data_index * this->neurons_total_reduce_summation_size); // Reduced summation.
        
                Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_data_index * tmp_number_neuron_units],
                                       tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units],
                                       *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
            }
        }
    }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Softmax__Summation(size_t const data_index_received,
                                                                                                                        size_t const number_neurons_received,
                                                                                                                        size_t const number_connections_received,
                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                        T *const ptr_array_layer_it_summations_received,
                                                                                                                        T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                        T const *const ptr_array_parameters_received,
                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                        struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u), // Add bias.
                               tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *const tmp_ptr_array_parameters(ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased);
    T *const tmp_ptr_array_reduce_summation(ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received);

    Reduce::Reduce_XX<T>(number_connections_received,
                                        number_neurons_received,
                                        tmp_ptr_array_reduce_summation,
                                        tmp_ptr_array_parameters,
                                        ptr_array_previous_layer_outputs_received,
                                        ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                        ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);

    ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    ptr_array_layer_it_summations_received[tmp_thread_global_index] += *tmp_ptr_array_reduce_summation; // Reduced summation.
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Softmax__Summation(size_t const size_received,
                                                                                                                        size_t const data_index_received,
                                                                                                                        size_t const number_neurons_received,
                                                                                                                        size_t const number_connections_received,
                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                        T *const ptr_array_layer_it_summations_received,
                                                                                                                        T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                        T const *const ptr_array_parameters_received,
                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                        struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u), // Add bias.
                               tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *const tmp_ptr_array_parameters(ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased);
    T *tmp_ptr_array_reduce_summation;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_reduce_summation = ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received;

        Reduce::Reduce_XX<T>(number_connections_received,
                                            number_neurons_received,
                                            tmp_ptr_array_reduce_summation,
                                            tmp_ptr_array_parameters,
                                             ptr_array_previous_layer_outputs_received,
                                            ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);
        
        ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    if(tmp_thread_global_index < size_received)
    { ptr_array_layer_it_summations_received[tmp_thread_global_index] += *tmp_ptr_array_reduce_summation; } // Reduced summation.
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Softmax__Summation(size_t const size_received,
                                                                                                                                size_t const data_index_received,
                                                                                                                                size_t const number_neurons_received,
                                                                                                                                size_t const number_connections_received,
                                                                                                                                size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                T *const ptr_array_layer_it_summations_received,
                                                                                                                                T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                T const *const ptr_array_parameters_received,
                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u); // Add bias.
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *tmp_ptr_array_parameters;

    // Loop through each neurons.
    do
    {
        tmp_ptr_array_parameters = ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased;

        Reduce::Reduce_XX<T>(number_connections_received,
                                            number_neurons_received,
                                            ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received,
                                            tmp_ptr_array_parameters,
                                             ptr_array_previous_layer_outputs_received,
                                            ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);
        
        ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // Reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop through each neurons.
    do
    {
        ptr_array_layer_it_summations_received[tmp_thread_global_index] += *(ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received); // Reduced summation.

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Softmax__Activation(size_t const size_received,
                                                                                                                             size_t const data_index_received,
                                                                                                                             T const layer_maximum_summation_received,
                                                                                                                             T *const ptr_array_layer_it_summations_received,
                                                                                                                             T *const ptr_array_layer_it_values_received,
                                                                                                                             struct CUDA_Neuron *const ptr_layer_it_first_neuron_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        ptr_array_layer_it_values_received[tmp_thread_global_index] = exp(ptr_array_layer_it_summations_received[tmp_thread_global_index] - layer_maximum_summation_received);
        
        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

// Template overload [double]
__global__ void kernel_while__Forward_Pass__FC_to_FC__Softmax__Activation(size_t const size_received,
                                                                                                                             size_t const data_index_received,
                                                                                                                             double const layer_maximum_summation_received,
                                                                                                                             double *const ptr_array_layer_it_summations_received,
                                                                                                                             double *const ptr_array_layer_it_values_received,
                                                                                                                             struct CUDA_Neuron *const ptr_layer_it_first_neuron_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        ptr_array_layer_it_values_received[tmp_thread_global_index] = exp(ptr_array_layer_it_summations_received[tmp_thread_global_index] - layer_maximum_summation_received);
        
        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Forward_Pass__FC_to_FC__Softmax(bool &ref_synchronized_received,
                                                                                                                  size_t const batch_size_received,
                                                                                                                  struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                  struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                  struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                  struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    PRINT_FORMAT("%s: [FUNCTION DEPRECATED] TODO: Fix \"Forward_Pass__FC_to_FC__Softmax\" algorithm." NEW_LINE, __FUNCTION__);

    /*
    struct CUDA_Neuron const *const tmp_ptr_last_neuron_unit(ptr_layer_it_received->ptr_last_neuron_unit - 1); // Subtract bias.
    struct CUDA_Neuron *tmp_ptr_neuron_unit_it(ptr_layer_it_received->ptr_array_neuron_units);
    
    // TODO: Remove bias term in nConnections.
    // By subtracting the bias the variable "ptr_dim3_grid_connections" become a false dimension.
    size_t const tmp_number_connections(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections - 1u), // Subtract bias.
                                tmp_number_neuron_units(*ptr_layer_it_received->ptr_number_neurons);

    T_ const *const tmp_ptr_array_previous_layer_outputs(ptr_previous_layer_it_received->ptr_array_neuron_units->ptr_array_values + thread_index_received * (tmp_number_connections + 1u)), // Add bias.
                  *tmp_ptr_array_parameters(this->ptr_array_parameters + *tmp_ptr_neuron_unit_it->ptr_first_forward_connection_index);
    T_ tmp_summation(0),
         tmp_layer_maximum_summation(MyEA::Math::NUMERIC_LIMITS_MIN<T_>),
        *tmp_ptr_array_reduce_summation;
    
    if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;
        
        LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Softmax__Summation<T_>,
                                                            ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                            ptr_layer_it_received->ptr_dim3_block_neurons,
                                                            0_zu,
                                                            tmp_number_neuron_units - 1u, // Subtract bias.
                                                            thread_index_received,
                                                            tmp_number_neuron_units,
                                                            tmp_number_connections,
                                                            this->neurons_total_reduce_summation_size,
                                                            tmp_ptr_neuron_unit_it->ptr_array_summations + thread_index_received * tmp_number_neuron_units,
                                                            tmp_ptr_neuron_unit_it->ptr_array_reduce_summation,
                                                            tmp_ptr_array_parameters,
                                                            tmp_ptr_array_previous_layer_outputs,
                                                            tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                            tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation)
    }
    else
    {
        // Synchronize if needed to see the output of the previous layer.
        CUDA__Device_Synchronise(ref_synchronized_received, MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);

        // Loop through each neurons for doing a reduction of summation.
        for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                tmp_ptr_array_parameters += tmp_number_connections + 1u) // Add bias.
        {
            tmp_ptr_array_reduce_summation = *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + thread_index_received * this->neurons_total_reduce_summation_size;

            Reduce::Reduce_XX<T_>(tmp_number_connections,
                                                 tmp_number_neuron_units,
                                                 tmp_ptr_array_reduce_summation,
                                                 tmp_ptr_array_parameters,
                                                 tmp_ptr_array_previous_layer_outputs,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);

            tmp_ptr_neuron_unit_it->ptr_array_summations[thread_index_received * tmp_number_neuron_units] = tmp_ptr_array_parameters[tmp_number_connections]; // Bias.
        }
    
        // Do we need to synchronise? Based on "Reduce_XX" Function.
        // => Synchronize if needed to see the summation reduced of the layer.
        if(tmp_number_connections >= warpSize) { CUDA__Check_Error(); }
    
        // Loop through each neurons for retrieve reduced summation.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_received->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
        {
            tmp_ptr_neuron_unit_it->ptr_array_summations[thread_index_received * tmp_number_neuron_units] += *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + thread_index_received * this->neurons_total_reduce_summation_size); // Reduced summation.
            
            tmp_layer_maximum_summation = MyEA::Math::Maximum(tmp_layer_maximum_summation, tmp_ptr_neuron_unit_it->ptr_array_summations[thread_index_received * tmp_number_neuron_units]);
        }
    
        // Loop through each neurons for activation function.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_received->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
        {
        #if defined(COMPILE_DEBUG)
            if(*tmp_ptr_neuron_unit_it->ptr_type_activation_function != MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_SOFTMAX)
            {
                PRINT_FORMAT("%s: ERROR : Can not use a activation function different than softmax." NEW_LINE, __FUNCTION__);

                return;
            }
        #endif
            
            tmp_summation += tmp_ptr_neuron_unit_it->ptr_array_values[thread_index_received * tmp_number_neuron_units] = exp(tmp_ptr_neuron_unit_it->ptr_array_summations[thread_index_received * tmp_number_neuron_units] - tmp_layer_maximum_summation);
        }
        
        // Probability on one.
        tmp_summation = 1_T / tmp_summation;
        
        // Loop through each neurons for normalizing probability.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_received->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
        { tmp_ptr_neuron_unit_it->ptr_array_values[thread_index_received * tmp_number_neuron_units] *= tmp_summation; }
    }
    */
}
