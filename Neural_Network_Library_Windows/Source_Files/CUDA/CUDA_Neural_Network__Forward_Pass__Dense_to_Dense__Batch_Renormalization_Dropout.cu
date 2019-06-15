#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Reduce.cuh>
#include <CUDA/CUDA_Multiply_1D.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons(size_t const data_index_received,
                                                                                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                                                                                            size_t const number_connections_received,
                                                                                                                                                                                                            size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                            T const probability_retained_unit_received,
                                                                                                                                                                                                            T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                            T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                                                                                            T *const ptr_array_layer_it_values_received,
                                                                                                                                                                                                            T const *const ptr_array_layer_it_scales_received,
                                                                                                                                                                                                            T const *const ptr_array_layer_it_shifts_received,
                                                                                                                                                                                                            T const *const ptr_array_layer_it_means_averages_received,
                                                                                                                                                                                                            T const *const ptr_array_layer_it_variances_averages_received,
                                                                                                                                                                                                            T const *const ptr_array_parameters_received,
                                                                                                                                                                                                            T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                            enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                                                                            struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                                                                                            struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u), // Add bias.
                                tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T *const tmp_ptr_array_reduce_summation(ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received);

    Reduce::Reduce_XX<T>(number_connections_received,
                                        number_neurons_received,
                                        tmp_ptr_array_reduce_summation,
                                        ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased,
                                        ptr_array_previous_layer_outputs_received,
                                        ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                        ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // Normalize input, scale and shift.
    // value_normalize = scale * ( (summation - mean) / variance ) + shift
    ptr_array_layer_it_summations_received[tmp_thread_global_index] = ptr_array_layer_it_scales_received[tmp_thread_global_index] * ( (*tmp_ptr_array_reduce_summation - ptr_array_layer_it_means_averages_received[tmp_thread_global_index]) / ptr_array_layer_it_variances_averages_received[tmp_thread_global_index]) + ptr_array_layer_it_shifts_received[tmp_thread_global_index];

    Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                           ptr_array_layer_it_summations_received[tmp_thread_global_index],
                           ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);

    ptr_array_layer_it_values_received[tmp_thread_global_index] *= probability_retained_unit_received;
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                                            size_t const data_index_received,
                                                                                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                                                                                            size_t const number_connections_received,
                                                                                                                                                                                                            size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                            T const probability_retained_unit_received,
                                                                                                                                                                                                            T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                            T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                                                                                            T *const ptr_array_layer_it_values_received,
                                                                                                                                                                                                            T const *const ptr_array_layer_it_scales_received,
                                                                                                                                                                                                            T const *const ptr_array_layer_it_shifts_received,
                                                                                                                                                                                                            T const *const ptr_array_layer_it_means_averages_received,
                                                                                                                                                                                                            T const *const ptr_array_layer_it_variances_averages_received,
                                                                                                                                                                                                            T const *const ptr_array_parameters_received,
                                                                                                                                                                                                            T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                            enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                                                                            struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                                                                                            struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u), // Add bias.
                                tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T *tmp_ptr_array_reduce_summation;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_reduce_summation = ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received;

        Reduce::Reduce_XX<T>(number_connections_received,
                                            number_neurons_received,
                                            tmp_ptr_array_reduce_summation,
                                            ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased,
                                            ptr_array_previous_layer_outputs_received,
                                            ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    if(tmp_thread_global_index < size_received)
    {
        // Normalize input, scale and shift.
        // value_normalize = scale * ( (summation - mean) / variance ) + shift
        ptr_array_layer_it_summations_received[tmp_thread_global_index] = ptr_array_layer_it_scales_received[tmp_thread_global_index] * ( (*tmp_ptr_array_reduce_summation - ptr_array_layer_it_means_averages_received[tmp_thread_global_index]) / ptr_array_layer_it_variances_averages_received[tmp_thread_global_index]) + ptr_array_layer_it_shifts_received[tmp_thread_global_index];

        Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                               ptr_array_layer_it_summations_received[tmp_thread_global_index],
                               ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);

        ptr_array_layer_it_values_received[tmp_thread_global_index] *= probability_retained_unit_received;
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                                                      size_t const data_index_received,
                                                                                                                                                                                                                      size_t const number_neurons_received,
                                                                                                                                                                                                                      size_t const number_connections_received,
                                                                                                                                                                                                                      size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                      T const probability_retained_unit_received,
                                                                                                                                                                                                                      T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                                      T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                                                                                                      T *const ptr_array_layer_it_values_received,
                                                                                                                                                                                                                      T const *const ptr_array_layer_it_scales_received,
                                                                                                                                                                                                                      T const *const ptr_array_layer_it_shifts_received,
                                                                                                                                                                                                                      T const *const ptr_array_layer_it_means_averages_received,
                                                                                                                                                                                                                      T const *const ptr_array_layer_it_variances_averages_received,
                                                                                                                                                                                                                      T const *const ptr_array_parameters_received,
                                                                                                                                                                                                                      T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                                      enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                                                                                      struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                                                                                                      struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u); // Add bias.
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    // Loop through each neurons.
    do
    {
        Reduce::Reduce_XX<T>(number_connections_received,
                                            number_neurons_received,
                                            ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received,
                                            ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased,
                                            ptr_array_previous_layer_outputs_received,
                                            ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);
        
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
        // Normalize input, scale and shift.
        // value_normalize = scale * ( (summation - mean) / variance ) + shift
        ptr_array_layer_it_summations_received[tmp_thread_global_index] = ptr_array_layer_it_scales_received[tmp_thread_global_index] * ( (*(ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received) - ptr_array_layer_it_means_averages_received[tmp_thread_global_index]) / ptr_array_layer_it_variances_averages_received[tmp_thread_global_index]) + ptr_array_layer_it_shifts_received[tmp_thread_global_index];

        Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                               ptr_array_layer_it_summations_received[tmp_thread_global_index],
                               ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);

        ptr_array_layer_it_values_received[tmp_thread_global_index] *= probability_retained_unit_received;
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Parallel_Batch__Serialize_Neurons(size_t const number_neurons_received,
                                                                                                                                                                                                             size_t const number_connections_received,
                                                                                                                                                                                                             size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                             T const probability_retained_unit_received,
                                                                                                                                                                                                             T const *ptr_array_parameters_received,
                                                                                                                                                                                                             T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                             struct CUDA_Neuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                             struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron *tmp_ptr_neuron_unit_it(ptr_layer_it_first_neuron_received);

    T const *const tmp_ptr_array_previous_layer_outputs(ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u)); // Add bias.
    
    // Loop through each neurons for doing a reduction of summation.
    for(; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                           ptr_array_parameters_received += number_connections_received + 1u) // Add bias.
    {
        Reduce::Reduce_XX<T>(number_connections_received,
                                            number_neurons_received,
                                            *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                            ptr_array_parameters_received,
                                            tmp_ptr_array_previous_layer_outputs,
                                            tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                            tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);
    }

    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // Loop through each neurons for retrieve reduced summation and then do a summation of mean and variance.
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
    {
        // Normalize input, scale and shift.
        // value_normalize = scale * ( (summation - mean) / variance ) + shift
        tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = *tmp_ptr_neuron_unit_it->ptr_scale * ( (*(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received) - *tmp_ptr_neuron_unit_it->ptr_mean_average) / *tmp_ptr_neuron_unit_it->ptr_variance_average ) + *tmp_ptr_neuron_unit_it->ptr_shift;
        
        Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                              tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                              *tmp_ptr_neuron_unit_it->ptr_type_activation_function);

        tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received] *= probability_retained_unit_received;
    }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                                                                                            size_t const number_connections_received,
                                                                                                                                                                                                            size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                            T const probability_retained_unit_received,
                                                                                                                                                                                                            T const *ptr_array_parameters_received,
                                                                                                                                                                                                            T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                            struct CUDA_Neuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                            struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron * tmp_ptr_neuron_unit_it(ptr_layer_it_first_neuron_received);

    T const *const tmp_ptr_array_previous_layer_outputs(ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u)); // Add bias.
    
    if(tmp_thread_global_index < size_received)
    {
        // Loop through each neurons for doing a reduction of summation.
        for(; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                               ptr_array_parameters_received += number_connections_received + 1u) // Add bias.
        {
            Reduce::Reduce_XX<T>(number_connections_received,
                                                number_neurons_received,
                                                *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                                ptr_array_parameters_received,
                                                tmp_ptr_array_previous_layer_outputs,
                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);
        }
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    if(tmp_thread_global_index < size_received)
    {
        // Loop through each neurons for retrieve reduced summation and then do a summation of mean and variance.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            // Normalize input, scale and shift.
            // value_normalize = scale * ( (summation - mean) / variance ) + shift
            tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = *tmp_ptr_neuron_unit_it->ptr_scale * ( (*(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received) - *tmp_ptr_neuron_unit_it->ptr_mean_average) / *tmp_ptr_neuron_unit_it->ptr_variance_average ) + *tmp_ptr_neuron_unit_it->ptr_shift;

            Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                    tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                    *tmp_ptr_neuron_unit_it->ptr_type_activation_function);

            tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received] *= probability_retained_unit_received;
        }
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                                                                                    size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                    T const probability_retained_unit_received,
                                                                                                                                                                                                                    T const *ptr_array_parameters_received,
                                                                                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                                    struct CUDA_Neuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                                    struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron *tmp_ptr_neuron_unit_it;

    T const *tmp_ptr_array_previous_layer_outputs;

    // Loop through each sample.
    do
    {
        // Loop through each neurons for doing a reduction of summation.
        for(tmp_ptr_array_previous_layer_outputs = ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
            tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                             ptr_array_parameters_received += number_connections_received + 1u) // Add bias.
        {
            Reduce::Reduce_XX<T>(number_connections_received,
                                                number_neurons_received,
                                                *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                                ptr_array_parameters_received,
                                                tmp_ptr_array_previous_layer_outputs,
                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // Reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop through each sample.
    do
    {
        // Loop through each neurons for retrieve reduced summation and then do a summation of mean and variance.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            // Normalize input, scale and shift.
            // value_normalize = scale * ( (summation - mean) / variance ) + shift
            tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = *tmp_ptr_neuron_unit_it->ptr_scale * ( (*(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received) - *tmp_ptr_neuron_unit_it->ptr_mean_average) / *tmp_ptr_neuron_unit_it->ptr_variance_average ) + *tmp_ptr_neuron_unit_it->ptr_shift;

            Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                   tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                   *tmp_ptr_neuron_unit_it->ptr_type_activation_function);

            tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received] *= probability_retained_unit_received;
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Parallel_Batch__Parallel_Neurons(size_t const number_neurons_received,
                                                                                                                                                                                                            size_t const number_connections_received,
                                                                                                                                                                                                            size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                            T const *const ptr_array_parameters_received,
                                                                                                                                                                                                            T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                            struct CUDA_Layer *const ptr_layer_it_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<T>,
                                                      ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                      ptr_layer_it_received->ptr_dim3_block_neurons,
                                                      0_zu,
                                                      number_neurons_received - 1u, // Subtract bias.
                                                      tmp_thread_global_index,
                                                      number_neurons_received,
                                                      number_connections_received,
                                                      neurons_total_reduce_summation_size_received,
                                                      ptr_layer_it_received->dropout_values[0u],
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                      tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                      tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                      tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                      ptr_array_parameters_received,
                                                      ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                      tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation)
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Parallel_Batch__Parallel_Neurons(size_t const size_received,
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
        LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<T>,
                                                          ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                          ptr_layer_it_received->ptr_dim3_block_neurons,
                                                          0_zu,
                                                          number_neurons_received - 1u, // Subtract bias.
                                                          tmp_thread_global_index,
                                                          number_neurons_received,
                                                          number_connections_received,
                                                          neurons_total_reduce_summation_size_received,
                                                          ptr_layer_it_received->dropout_values[0u],
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                          tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                          tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                          tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                          ptr_array_parameters_received,
                                                          ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                          tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation)
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                                                                                    size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                    T const *const ptr_array_parameters_received,
                                                                                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                                    struct CUDA_Layer *const ptr_layer_it_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T_ const tmp_probability_retained_unit(ptr_layer_it_received->dropout_values[0u]);

    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    // KERNEL LAUNCH
    //    1: Launching do-while elements.
    if(ptr_layer_it_received->ptr_dim3_grid_neurons->x * ptr_layer_it_received->ptr_dim3_block_neurons->x < number_neurons_received - 1u) // Subtract bias.
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons, *ptr_layer_it_received->ptr_dim3_block_neurons >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                          tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                                                          number_neurons_received,
                                                                                                                                                                                                                                                                                                                                                          number_connections_received,
                                                                                                                                                                                                                                                                                                                                                          neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                                                                          tmp_probability_retained_unit,
                                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_variance_average,
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
            kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons, *ptr_layer_it_received->ptr_dim3_block_neurons >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                                                number_neurons_received,
                                                                                                                                                                                                                                                                                                                                                number_connections_received,
                                                                                                                                                                                                                                                                                                                                                neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                                                                tmp_probability_retained_unit,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_variance_average,
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
            kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons, *ptr_layer_it_received->ptr_dim3_block_neurons >>> (tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                                                number_neurons_received,
                                                                                                                                                                                                                                                                                                                                                number_connections_received,
                                                                                                                                                                                                                                                                                                                                                neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                                                                tmp_probability_retained_unit,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_variance_average,
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

__device__ void CUDA_Neural_Network::Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing(bool &ref_synchronized_received,
                                                                                                                                                                      size_t const batch_size_received,
                                                                                                                                                                      struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                                                      struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                                                                      struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                                                      struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    size_t tmp_data_index;
    
    T_ const *const tmp_ptr_array_previous_layer_outputs_begin(ptr_previous_layer_it_received->ptr_array_neuron_units->ptr_array_values),
                  *tmp_ptr_array_previous_layer_outputs,
                  *tmp_ptr_array_parameters,
                  tmp_probability_retained_unit(ptr_layer_it_received->dropout_values[0u]);

    struct CUDA_Neuron const *const tmp_ptr_layer_it_last_neuron(ptr_layer_it_received->ptr_last_neuron_unit - 1); // Subtract bias.
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units),
                                             *tmp_ptr_neuron_unit_it;

    size_t const tmp_number_connections(*tmp_ptr_layer_it_first_neuron->ptr_number_forward_connections - 1u), // Subtract bias.
                                tmp_number_neuron_units(*ptr_layer_it_received->ptr_number_neurons);

    // Condition to enter into dynamic parallelisme of each sample.
    if(USE_PARALLEL && batch_size_received >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;
        
        // Condition to enter into dynamic parallelisme of each sample and neurons.
        if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
        {
            LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Parallel_Batch__Parallel_Neurons<T_>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_zu,
                                                              batch_size_received,
                                                              tmp_number_neuron_units,
                                                              tmp_number_connections,
                                                              this->neurons_total_reduce_summation_size,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_array_previous_layer_outputs_begin,
                                                              ptr_layer_it_received)
        }
        // Condition to enter into dynamic parallelisme of each sample.
        else
        {
            LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Parallel_Batch__Serialize_Neurons<T_>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_zu,
                                                              batch_size_received,
                                                              tmp_number_neuron_units,
                                                              tmp_number_connections,
                                                              this->neurons_total_reduce_summation_size,
                                                              tmp_probability_retained_unit,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_array_previous_layer_outputs_begin,
                                                              tmp_ptr_layer_it_first_neuron,
                                                              tmp_ptr_layer_it_last_neuron)
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
                kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP, *ptr_layer_it_received->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1u,
                                                                                                                                                                                                                                                                                                                                                                tmp_data_index,
                                                                                                                                                                                                                                                                                                                                                                tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                                tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                                                this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                                                                tmp_probability_retained_unit,
                                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                                                                                                                                                                                                                                                                                                                                this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                                                tmp_ptr_array_previous_layer_outputs_begin + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
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
                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP, *ptr_layer_it_received->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1u,
                                                                                                                                                                                                                                                                                                                                                    tmp_data_index,
                                                                                                                                                                                                                                                                                                                                                    tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                    tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                                    this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                                                    tmp_probability_retained_unit,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                                                                                                                                                                                                                                                                                                                    this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_array_previous_layer_outputs_begin + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
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
                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP, *ptr_layer_it_received->ptr_dim3_block_neurons_DP >>> (tmp_data_index,
                                                                                                                                                                                                                                                                                                                                                      tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                      tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                                      this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                                                      tmp_probability_retained_unit,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                                                                                                                                                                                                                                                                                                                      this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_array_previous_layer_outputs_begin + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
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
        // Synchronize if needed to see the output of the previous layer.
        CUDA__Device_Synchronise(ref_synchronized_received, MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);

        // Loop through each sample.
        for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
        {
            tmp_ptr_array_previous_layer_outputs = tmp_ptr_array_previous_layer_outputs_begin + tmp_data_index * (tmp_number_connections + 1u); // Add bias.

            // Loop through each neurons for doing a reduction of summation.
            for(tmp_ptr_array_parameters = this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_layer_it_last_neuron; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                    tmp_ptr_array_parameters += tmp_number_connections + 1u) // Add bias.
            {
                Reduce::Reduce_XX<T_>(tmp_number_connections,
                                                     tmp_number_neuron_units,
                                                     *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_data_index * this->neurons_total_reduce_summation_size,
                                                     tmp_ptr_array_parameters,
                                                     tmp_ptr_array_previous_layer_outputs,
                                                     tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);
            }
        }
        
        // Do we need to synchronise? Based on "Reduce_XX" Function.
        // => Synchronize if needed to see the summation reduced of the layer.
        if(tmp_number_connections >= warpSize * 2u) { CUDA__Check_Error(); }

        // Activation function.
        for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
        {
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_layer_it_last_neuron; ++tmp_ptr_neuron_unit_it)
            {
                // Normalize input, scale and shift.
                // value_normalize = scale * ( (summation - mean) / variance ) + shift
                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units] = *tmp_ptr_neuron_unit_it->ptr_scale * ( (*(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_data_index * this->neurons_total_reduce_summation_size) - *tmp_ptr_neuron_unit_it->ptr_mean_average) / *tmp_ptr_neuron_unit_it->ptr_variance_average ) + *tmp_ptr_neuron_unit_it->ptr_shift;
                
                Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_data_index * tmp_number_neuron_units],
                                       tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units],
                                       *tmp_ptr_neuron_unit_it->ptr_type_activation_function);

                tmp_ptr_neuron_unit_it->ptr_array_values[tmp_data_index * tmp_number_neuron_units] *= tmp_probability_retained_unit;
            }
        }
    }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation(size_t const data_index_received,
                                                                                                                                                                                                                  size_t const number_neurons_received,
                                                                                                                                                                                                                  size_t const number_connections_received,
                                                                                                                                                                                                                  size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                  bool const *const ptr_array_layer_it_masks_dropouts_received,
                                                                                                                                                                                                                  T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                                  T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                                                                                                  T *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                                  T *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                                  T const *const ptr_array_parameters_received,
                                                                                                                                                                                                                  T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                                  struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                                                                                                  struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation__0[];
    T (&tmp_ptr_array_summations)[] = tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation__0;

    if(ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
    {
        Reduce::Reduce_XX<T>(number_connections_received,
                                            number_neurons_received,
                                            ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received,
                                            ptr_array_parameters_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                            ptr_array_previous_layer_outputs_received,
                                            ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);
    }

    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    if(ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
    {
        tmp_ptr_array_summations[threadIdx.x] = *(ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received);
    
        // mean += summation
        ptr_array_layer_it_means_received[tmp_thread_global_index] += tmp_ptr_array_summations[threadIdx.x];
        // variance += pow(summation, 2)
        ptr_array_layer_it_variances_received[tmp_thread_global_index] += tmp_ptr_array_summations[threadIdx.x] * tmp_ptr_array_summations[threadIdx.x];

        ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_summations[threadIdx.x];
    }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation(size_t const size_received,
                                                                                                                                                                                                                    size_t const data_index_received,
                                                                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                                                                                    size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                    bool const *const ptr_array_layer_it_masks_dropouts_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                                    T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                                    T const *const ptr_array_parameters_received,
                                                                                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                                    struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                                                                                                    struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation__1[];
    T (&tmp_ptr_array_summations)[] = tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation__1;

    if(tmp_thread_global_index < size_received && ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
    {
        Reduce::Reduce_XX<T>(number_connections_received,
                                            number_neurons_received,
                                            ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received,
                                            ptr_array_parameters_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                            ptr_array_previous_layer_outputs_received,
                                            ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    if(tmp_thread_global_index < size_received && ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
    {
        tmp_ptr_array_summations[threadIdx.x] = *(ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received);
        
        // mean += summation
        ptr_array_layer_it_means_received[tmp_thread_global_index] += tmp_ptr_array_summations[threadIdx.x];
        // variance += pow(summation, 2)
        ptr_array_layer_it_variances_received[tmp_thread_global_index] += tmp_ptr_array_summations[threadIdx.x] * tmp_ptr_array_summations[threadIdx.x];

        ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_summations[threadIdx.x];
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation(size_t const size_received,
                                                                                                                                                                                                                            size_t const data_index_received,
                                                                                                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                                                                                                            size_t const number_connections_received,
                                                                                                                                                                                                                            size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                            bool const *const ptr_array_layer_it_masks_dropouts_received,
                                                                                                                                                                                                                            T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                                            T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                                                                                                            T *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                                            T *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                                            T const *const ptr_array_parameters_received,
                                                                                                                                                                                                                            T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                                            struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                                                                                                            struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation[];
    T (&tmp_ptr_array_summations)[] = tmp_shared_T__kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation;

    // Loop through each neurons.
    do
    {
        if(ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
        {
            Reduce::Reduce_XX<T>(number_connections_received,
                                                number_neurons_received,
                                                ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received,
                                                ptr_array_parameters_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                ptr_array_previous_layer_outputs_received,
                                                ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                                ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);
        }

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
        if(ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
        {
            tmp_ptr_array_summations[threadIdx.x] = *(ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received);
        
            // mean += summation
            ptr_array_layer_it_means_received[tmp_thread_global_index] += tmp_ptr_array_summations[threadIdx.x];
            // variance += pow(summation, 2)
            ptr_array_layer_it_variances_received[tmp_thread_global_index] += tmp_ptr_array_summations[threadIdx.x] * tmp_ptr_array_summations[threadIdx.x];

            ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_summations[threadIdx.x];
        }

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Average(bool const *const ptr_array_layer_it_masks_dropouts_received,
                                                                                                                                                                                   T const T_batch_size_received,
                                                                                                                                                                                   T const normalization_momentum_average_received,
                                                                                                                                                                                   T const epsilon_received,
                                                                                                                                                                                   T const r_correction_maximum_received,
                                                                                                                                                                                   T const d_correction_maximum_received,
                                                                                                                                                                                   T *const ptr_array_layer_it_means_received,
                                                                                                                                                                                   T *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                   T *const ptr_array_layer_it_means_averages_received,
                                                                                                                                                                                   T *const ptr_array_layer_it_variances_averages_received,
                                                                                                                                                                                   T *const ptr_array_layer_it_r_correction_received,
                                                                                                                                                                                   T *const ptr_array_layer_it_d_correction_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Average__0[];
    /* Index map:
        0: mean, value (d_correction)
        1: mean_average
        2: variance, value (r_correction)
        3: variance_average
        4: low (r_correction) */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Average__0;

    if(ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
    {
        // Average batch mean.
        // mean_b = sum(summation, N) / N
        ptr_array_layer_it_means_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x] = ptr_array_layer_it_means_received[tmp_thread_global_index] / T_batch_size_received;

        // Average exponentialy global mean.
        // mean += momentum * (mean_b - mean)
        tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_layer_it_means_averages_received[tmp_thread_global_index] += normalization_momentum_average_received * (tmp_ptr_array_smem[threadIdx.x] - ptr_array_layer_it_means_averages_received[tmp_thread_global_index]); // Exponential moving average.
        
        // Average batch variance.
        // variance_b = sqrt( ((sum(pow(summation, 2), N) / N) - pow(mean_b, 2) + epsilon )
        ptr_array_layer_it_variances_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = sqrt(ptr_array_layer_it_variances_received[tmp_thread_global_index] / T_batch_size_received - tmp_ptr_array_smem[threadIdx.x] * tmp_ptr_array_smem[threadIdx.x] + epsilon_received);
        
        // Average exponentialy global variance.
        // variance += momentum * (variance_b - variance)   "Risk of NaN if momentum equal 1.0"
        tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x] = ptr_array_layer_it_variances_averages_received[tmp_thread_global_index] += normalization_momentum_average_received * (tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] - ptr_array_layer_it_variances_averages_received[tmp_thread_global_index]); // Exponential moving average.
    
        // r correction.
        // value = variance_b / variance
        tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x];
        // low = 1 / r_correction_max
        tmp_ptr_array_smem[threadIdx.x + 4u * blockDim.x] = T(1) / r_correction_maximum_received;
        // high = r_correction_max
        // r_correction = clip(value, low, high)
        ptr_array_layer_it_r_correction_received[tmp_thread_global_index] = MyEA::Math::Clip<T_>(tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x], tmp_ptr_array_smem[threadIdx.x + 4u * blockDim.x], r_correction_maximum_received);

        // d correction.
        // value = (mean_b - mean) / variance
        tmp_ptr_array_smem[threadIdx.x] = (tmp_ptr_array_smem[threadIdx.x] - tmp_ptr_array_smem[threadIdx.x + blockDim.x]) / tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x];
        // low = -d_correction_max
        // high = d_correction_max
        // d_correction = clip(value, low, high)
        ptr_array_layer_it_d_correction_received[tmp_thread_global_index] = MyEA::Math::Clip<T_>(tmp_ptr_array_smem[threadIdx.x], -d_correction_maximum_received, d_correction_maximum_received);
    }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Average(size_t const size_received,
                                                                                                                                                                                    bool const *const ptr_array_layer_it_masks_dropouts_received,
                                                                                                                                                                                    T const T_batch_size_received,
                                                                                                                                                                                    T const normalization_momentum_average_received,
                                                                                                                                                                                    T const epsilon_received,
                                                                                                                                                                                    T const r_correction_maximum_received,
                                                                                                                                                                                    T const d_correction_maximum_received,
                                                                                                                                                                                    T *const ptr_array_layer_it_means_received,
                                                                                                                                                                                    T *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                    T *const ptr_array_layer_it_means_averages_received,
                                                                                                                                                                                    T *const ptr_array_layer_it_variances_averages_received,
                                                                                                                                                                                    T *const ptr_array_layer_it_r_correction_received,
                                                                                                                                                                                    T *const ptr_array_layer_it_d_correction_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Average__1[];
    /* Index map:
        0: mean, value (d_correction)
        1: mean_average
        2: variance, value (r_correction)
        3: variance_average
        4: low (r_correction) */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Average__1;

    if(tmp_thread_global_index < size_received && ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
    {
        // Average batch mean.
        // mean_b = sum(summation, N) / N
        ptr_array_layer_it_means_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x] = ptr_array_layer_it_means_received[tmp_thread_global_index] / T_batch_size_received;

        // Average exponentialy global mean.
        // mean += momentum * (mean_b - mean)
        tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_layer_it_means_averages_received[tmp_thread_global_index] += normalization_momentum_average_received * (tmp_ptr_array_smem[threadIdx.x] - ptr_array_layer_it_means_averages_received[tmp_thread_global_index]); // Exponential moving average.
        
        // Average batch variance.
        // variance_b = sqrt( ((sum(pow(summation, 2), N) / N) - pow(mean_b, 2) + epsilon )
        ptr_array_layer_it_variances_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = sqrt(ptr_array_layer_it_variances_received[tmp_thread_global_index] / T_batch_size_received - tmp_ptr_array_smem[threadIdx.x] * tmp_ptr_array_smem[threadIdx.x] + epsilon_received);
        
        // Average exponentialy global variance.
        // variance += momentum * (variance_b - variance)   "Risk of NaN if momentum equal 1.0"
        tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x] = ptr_array_layer_it_variances_averages_received[tmp_thread_global_index] += normalization_momentum_average_received * (tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] - ptr_array_layer_it_variances_averages_received[tmp_thread_global_index]); // Exponential moving average.
    
        // r correction.
        // value = variance_b / variance
        tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x];
        // low = 1 / r_correction_max
        tmp_ptr_array_smem[threadIdx.x + 4u * blockDim.x] = T(1) / r_correction_maximum_received;
        // high = r_correction_max
        // r_correction = clip(value, low, high)
        ptr_array_layer_it_r_correction_received[tmp_thread_global_index] = MyEA::Math::Clip<T_>(tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x], tmp_ptr_array_smem[threadIdx.x + 4u * blockDim.x], r_correction_maximum_received);

        // d correction.
        // value = (mean_b - mean) / variance
        tmp_ptr_array_smem[threadIdx.x] = (tmp_ptr_array_smem[threadIdx.x] - tmp_ptr_array_smem[threadIdx.x + blockDim.x]) / tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x];
        // low = -d_correction_max
        // high = d_correction_max
        // d_correction = clip(value, low, high)
        ptr_array_layer_it_d_correction_received[tmp_thread_global_index] = MyEA::Math::Clip<T_>(tmp_ptr_array_smem[threadIdx.x], -d_correction_maximum_received, d_correction_maximum_received);
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Average(size_t const size_received,
                                                                                                                                                                                            bool const *const ptr_array_layer_it_masks_dropouts_received,
                                                                                                                                                                                            T const T_batch_size_received,
                                                                                                                                                                                            T const normalization_momentum_average_received,
                                                                                                                                                                                            T const epsilon_received,
                                                                                                                                                                                            T const r_correction_maximum_received,
                                                                                                                                                                                            T const d_correction_maximum_received,
                                                                                                                                                                                            T *const ptr_array_layer_it_means_received,
                                                                                                                                                                                            T *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                            T *const ptr_array_layer_it_means_averages_received,
                                                                                                                                                                                            T *const ptr_array_layer_it_variances_averages_received,
                                                                                                                                                                                            T *const ptr_array_layer_it_r_correction_received,
                                                                                                                                                                                            T *const ptr_array_layer_it_d_correction_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Average[];
    /* Index map:
        0: mean, value (d_correction)
        1: mean_average
        2: variance, value (r_correction)
        3: variance_average
        4: low (r_correction) */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Average;

    // Loop through each neurons.
    do
    {
        if(ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
        {
            // Average batch mean.
            // mean_b = sum(summation, N) / N
            ptr_array_layer_it_means_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x] = ptr_array_layer_it_means_received[tmp_thread_global_index] / T_batch_size_received;

            // Average exponentialy global mean.
            // mean += momentum * (mean_b - mean)
            tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_layer_it_means_averages_received[tmp_thread_global_index] += normalization_momentum_average_received * (tmp_ptr_array_smem[threadIdx.x] - ptr_array_layer_it_means_averages_received[tmp_thread_global_index]); // Exponential moving average.
        
            // Average batch variance.
            // variance_b = sqrt( ((sum(pow(summation, 2), N) / N) - pow(mean_b, 2) + epsilon )
            ptr_array_layer_it_variances_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = sqrt(ptr_array_layer_it_variances_received[tmp_thread_global_index] / T_batch_size_received - tmp_ptr_array_smem[threadIdx.x] * tmp_ptr_array_smem[threadIdx.x] + epsilon_received);
        
            // Average exponentialy global variance.
            // variance += momentum * (variance_b - variance)   "Risk of NaN if momentum equal 1.0"
            tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x] = ptr_array_layer_it_variances_averages_received[tmp_thread_global_index] += normalization_momentum_average_received * (tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] - ptr_array_layer_it_variances_averages_received[tmp_thread_global_index]); // Exponential moving average.
    
            // r correction.
            // value = variance_b / variance
            tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x];
            // low = 1 / r_correction_max
            tmp_ptr_array_smem[threadIdx.x + 4u * blockDim.x] = T(1) / r_correction_maximum_received;
            // high = r_correction_max
            // r_correction = clip(value, low, high)
            ptr_array_layer_it_r_correction_received[tmp_thread_global_index] = MyEA::Math::Clip<T_>(tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x], tmp_ptr_array_smem[threadIdx.x + 4u * blockDim.x], r_correction_maximum_received);

            // d correction.
            // value = (mean_b - mean) / variance
            tmp_ptr_array_smem[threadIdx.x] = (tmp_ptr_array_smem[threadIdx.x] - tmp_ptr_array_smem[threadIdx.x + blockDim.x]) / tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x];
            // low = -d_correction_max
            // high = d_correction_max
            // d_correction = clip(value, low, high)
            ptr_array_layer_it_d_correction_received[tmp_thread_global_index] = MyEA::Math::Clip<T_>(tmp_ptr_array_smem[threadIdx.x], -d_correction_maximum_received, d_correction_maximum_received);
        }

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation(bool const *const ptr_array_layer_it_masks_dropouts_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_values_hats_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_values_normalizes_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_values_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_scales_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_shifts_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_r_correction_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_d_correction_received,
                                                                                                                                                                                                                enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation__0[];
    T (&tmp_ptr_array_summations)[] = tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation__0;

    if(ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
    {
        tmp_ptr_array_summations[threadIdx.x] = ptr_array_layer_it_summations_received[tmp_thread_global_index];
    
        // Normalize.
        // value_hat = (summation - mean_b) / variance_b * r_correction + d_correction
        ptr_array_layer_it_values_hats_received[tmp_thread_global_index] = tmp_ptr_array_summations[threadIdx.x] = (tmp_ptr_array_summations[threadIdx.x] - ptr_array_layer_it_means_received[tmp_thread_global_index]) / ptr_array_layer_it_variances_received[tmp_thread_global_index] * ptr_array_layer_it_r_correction_received[tmp_thread_global_index] + ptr_array_layer_it_d_correction_received[tmp_thread_global_index];
        
        // Scale and shift.
        // value_normalize = scale * value_hat + shift
        ptr_array_layer_it_values_normalizes_received[tmp_thread_global_index] = tmp_ptr_array_summations[threadIdx.x] = ptr_array_layer_it_scales_received[tmp_thread_global_index] * tmp_ptr_array_summations[threadIdx.x] + ptr_array_layer_it_shifts_received[tmp_thread_global_index];
        
        // Activation.
        // value = AF(value_normalize)
        Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                                tmp_ptr_array_summations[threadIdx.x],
                                ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
    }
    else { ptr_array_layer_it_values_received[tmp_thread_global_index] = T(0); }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation(size_t const size_received,
                                                                                                                                                                                                                bool const *const ptr_array_layer_it_masks_dropouts_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_values_hats_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_values_normalizes_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_values_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_scales_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_shifts_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_r_correction_received,
                                                                                                                                                                                                                T *const ptr_array_layer_it_d_correction_received,
                                                                                                                                                                                                                enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation__1[];
    T (&tmp_ptr_array_summations)[] = tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation__1;

    // Loop through each neurons.
    if(tmp_thread_global_index < size_received)
    {
        if(ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
        {
            tmp_ptr_array_summations[threadIdx.x] = ptr_array_layer_it_summations_received[tmp_thread_global_index];
        
            // Normalize.
            // value_hat = (summation - mean_b) / variance_b * r_correction + d_correction
            ptr_array_layer_it_values_hats_received[tmp_thread_global_index] = tmp_ptr_array_summations[threadIdx.x] = (tmp_ptr_array_summations[threadIdx.x] - ptr_array_layer_it_means_received[tmp_thread_global_index]) / ptr_array_layer_it_variances_received[tmp_thread_global_index] * ptr_array_layer_it_r_correction_received[tmp_thread_global_index] + ptr_array_layer_it_d_correction_received[tmp_thread_global_index];
        
            // Scale and shift.
            // value_normalize = scale * value_hat + shift
            ptr_array_layer_it_values_normalizes_received[tmp_thread_global_index] = tmp_ptr_array_summations[threadIdx.x] = ptr_array_layer_it_scales_received[tmp_thread_global_index] * tmp_ptr_array_summations[threadIdx.x] + ptr_array_layer_it_shifts_received[tmp_thread_global_index];
        
            // Activation.
            // value = AF(value_normalize)
            Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                                   tmp_ptr_array_summations[threadIdx.x],
                                   ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
        }
        else { ptr_array_layer_it_values_received[tmp_thread_global_index] = T(0); }
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation(size_t const size_received,
                                                                                                                                                                                                                        bool const *const ptr_array_layer_it_masks_dropouts_received,
                                                                                                                                                                                                                        T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                                        T *const ptr_array_layer_it_values_hats_received,
                                                                                                                                                                                                                        T *const ptr_array_layer_it_values_normalizes_received,
                                                                                                                                                                                                                        T *const ptr_array_layer_it_values_received,
                                                                                                                                                                                                                        T *const ptr_array_layer_it_scales_received,
                                                                                                                                                                                                                        T *const ptr_array_layer_it_shifts_received,
                                                                                                                                                                                                                        T *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                                        T *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                                        T *const ptr_array_layer_it_r_correction_received,
                                                                                                                                                                                                                        T *const ptr_array_layer_it_d_correction_received,
                                                                                                                                                                                                                        enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation[];
    T (&tmp_ptr_array_summations)[] = tmp_shared_T__kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation;

    // Loop through each neurons.
    do
    {
        if(ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
        {
            tmp_ptr_array_summations[threadIdx.x] = ptr_array_layer_it_summations_received[tmp_thread_global_index];
        
            // Normalize.
            // value_hat = (summation - mean_b) / variance_b * r_correction + d_correction
            ptr_array_layer_it_values_hats_received[tmp_thread_global_index] = tmp_ptr_array_summations[threadIdx.x] = (tmp_ptr_array_summations[threadIdx.x] - ptr_array_layer_it_means_received[tmp_thread_global_index]) / ptr_array_layer_it_variances_received[tmp_thread_global_index] * ptr_array_layer_it_r_correction_received[tmp_thread_global_index] + ptr_array_layer_it_d_correction_received[tmp_thread_global_index];
        
            // Scale and shift.
            // value_normalize = scale * value_hat + shift
            ptr_array_layer_it_values_normalizes_received[tmp_thread_global_index] = tmp_ptr_array_summations[threadIdx.x] = ptr_array_layer_it_scales_received[tmp_thread_global_index] * tmp_ptr_array_summations[threadIdx.x] + ptr_array_layer_it_shifts_received[tmp_thread_global_index];
        
            // Activation.
            // value = AF(value_normalize)
            Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                                   tmp_ptr_array_summations[threadIdx.x],
                                   ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
        }
        else { ptr_array_layer_it_values_received[tmp_thread_global_index] = T(0); }

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Summation(size_t const number_neurons_received,
                                                                                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                                                                                    size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                    T const *ptr_array_parameters_received,
                                                                                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                                    struct CUDA_Neuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                                    struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron *tmp_ptr_neuron_unit_it(ptr_layer_it_first_neuron_received);

    T const *const tmp_ptr_array_previous_layer_outputs(ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u)); // Add bias.
    
    extern __shared__ T tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Summation_0[];
    T (&tmp_ptr_array_summations)[] = tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Summation_0;

    // Loop through each neurons for doing a reduction of summation.
    for(; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                           ptr_array_parameters_received += number_connections_received + 1u) // Add bias.
    {
        if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
        {
            Reduce::Reduce_XX<T>(number_connections_received,
                                                number_neurons_received,
                                                *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                                ptr_array_parameters_received,
                                                tmp_ptr_array_previous_layer_outputs,
                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);
        }
        else { tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received] = T(0); }
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // Loop through each neurons for retrieve reduced summation and then do a summation of mean and variance.
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
    {
        if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
        {
            tmp_ptr_array_summations[threadIdx.x] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received);
        
            // mean += summation
            tmp_ptr_neuron_unit_it->ptr_array_means[tmp_thread_global_index * number_neurons_received] += tmp_ptr_array_summations[threadIdx.x];
            // variance += pow(summation, 2)
            tmp_ptr_neuron_unit_it->ptr_array_variances[tmp_thread_global_index * number_neurons_received] += tmp_ptr_array_summations[threadIdx.x] * tmp_ptr_array_summations[threadIdx.x];

            tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_summations[threadIdx.x];
        }
    }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Summation(size_t const size_received,
                                                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                                                size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                T const *ptr_array_parameters_received,
                                                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                                struct CUDA_Neuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                                struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron * tmp_ptr_neuron_unit_it(ptr_layer_it_first_neuron_received);

    T const *const tmp_ptr_array_previous_layer_outputs(ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u)); // Add bias.
    
    extern __shared__ T tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Summation_1[];
    T (&tmp_ptr_array_summations)[] = tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Summation_1;

    if(tmp_thread_global_index < size_received)
    {
        // Loop through each neurons for doing a reduction of summation.
        for(; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                               ptr_array_parameters_received += number_connections_received + 1u) // Add bias.
        {
            if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
            {
                Reduce::Reduce_XX<T>(number_connections_received,
                                                    number_neurons_received,
                                                    *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                                    ptr_array_parameters_received,
                                                    tmp_ptr_array_previous_layer_outputs,
                                                    tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                    tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);
            }
            else { tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received] = T(0); }
        }
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    if(tmp_thread_global_index < size_received)
    {
        // Loop through each neurons for retrieve reduced summation and then do a summation of mean and variance.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
            {
                tmp_ptr_array_summations[threadIdx.x] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received);
            
                // mean += summation
                tmp_ptr_neuron_unit_it->ptr_array_means[tmp_thread_global_index * number_neurons_received] += tmp_ptr_array_summations[threadIdx.x];
                // variance += pow(summation, 2)
                tmp_ptr_neuron_unit_it->ptr_array_variances[tmp_thread_global_index * number_neurons_received] += tmp_ptr_array_summations[threadIdx.x] * tmp_ptr_array_summations[threadIdx.x];

                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_summations[threadIdx.x];
            }
        }
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Summation(size_t const size_received,
                                                                                                                                                                                                                           size_t const number_neurons_received,
                                                                                                                                                                                                                           size_t const number_connections_received,
                                                                                                                                                                                                                           size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                           T const *ptr_array_parameters_received,
                                                                                                                                                                                                                           T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                                           struct CUDA_Neuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                                           struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron *tmp_ptr_neuron_unit_it;

    T const *tmp_ptr_array_previous_layer_outputs;
    
    extern __shared__ T tmp_shared_T__kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Summation[];
    T (&tmp_ptr_array_summations)[] = tmp_shared_T__kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Summation;

    // Loop through each sample.
    do
    {
        // Loop through each neurons for doing a reduction of summation.
        for(tmp_ptr_array_previous_layer_outputs = ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
            tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                             ptr_array_parameters_received += number_connections_received + 1u) // Add bias.
        {
            if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
            {
                Reduce::Reduce_XX<T>(number_connections_received,
                                                    number_neurons_received,
                                                    *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                                    ptr_array_parameters_received,
                                                    tmp_ptr_array_previous_layer_outputs,
                                                    tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                    tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);
            }
            else { tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received] = T(0); }
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // Reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop through each sample.
    do
    {
        // Loop through each neurons for retrieve reduced summation and then do a summation of mean and variance.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
            {
                tmp_ptr_array_summations[threadIdx.x] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received);
            
                // mean += summation
                tmp_ptr_neuron_unit_it->ptr_array_means[tmp_thread_global_index * number_neurons_received] += tmp_ptr_array_summations[threadIdx.x];
                // variance += pow(summation, 2)
                tmp_ptr_neuron_unit_it->ptr_array_variances[tmp_thread_global_index * number_neurons_received] += tmp_ptr_array_summations[threadIdx.x] * tmp_ptr_array_summations[threadIdx.x];

                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_summations[threadIdx.x];
            }
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Parallel_Summation(size_t const number_neurons_received,
                                                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                                                size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                T const *const ptr_array_parameters_received,
                                                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                                struct CUDA_Layer *const ptr_layer_it_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron const *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation<T>,
                                                        ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                        ptr_layer_it_received->ptr_dim3_block_neurons,
                                                        ptr_layer_it_received->ptr_dim3_block_neurons->x * sizeof(T),
                                                        number_neurons_received - 1u, // Subtract bias.
                                                        tmp_thread_global_index,
                                                        number_neurons_received,
                                                        number_connections_received,
                                                        neurons_total_reduce_summation_size_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_means + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_variances + tmp_thread_global_index * number_neurons_received,
                                                        ptr_array_parameters_received,
                                                        ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation)
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Parallel_Summation(size_t const size_received,
                                                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                                                size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                T const *const ptr_array_parameters_received,
                                                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                                struct CUDA_Layer *const ptr_layer_it_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron const *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    if(tmp_thread_global_index < size_received)
    {
        LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation<T>,
                                                            ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                            ptr_layer_it_received->ptr_dim3_block_neurons,
                                                            ptr_layer_it_received->ptr_dim3_block_neurons->x * sizeof(T),
                                                            number_neurons_received - 1u, // Subtract bias.
                                                            tmp_thread_global_index,
                                                            number_neurons_received,
                                                            number_connections_received,
                                                            neurons_total_reduce_summation_size_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_means + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_variances + tmp_thread_global_index * number_neurons_received,
                                                            ptr_array_parameters_received,
                                                            ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation)
    }
    // |END| KERNEL LAUNCH |END|
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Parallel_Summation(size_t const size_received,
                                                                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                        T const *const ptr_array_parameters_received,
                                                                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                                        struct CUDA_Layer *const ptr_layer_it_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron const *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    // KERNEL LAUNCH
    //    1: Launching do-while elements.
    if(ptr_layer_it_received->ptr_dim3_grid_neurons->x * ptr_layer_it_received->ptr_dim3_block_neurons->x < number_neurons_received - 1u) // Subtract bias.
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                             *ptr_layer_it_received->ptr_dim3_block_neurons,
                                                                                                                                                                                                                             ptr_layer_it_received->ptr_dim3_block_neurons->x * sizeof(T) >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                            tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                            number_neurons_received,
                                                                                                                                                                                                                                                                                                                            number_connections_received,
                                                                                                                                                                                                                                                                                                                            neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_means + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_variances + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                            ptr_array_parameters_received,
                                                                                                                                                                                                                                                                                                                            ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
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
            kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                    *ptr_layer_it_received->ptr_dim3_block_neurons,
                                                                                                                                                                                                                    ptr_layer_it_received->ptr_dim3_block_neurons->x * sizeof(T) >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                    tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                    number_neurons_received,
                                                                                                                                                                                                                                                                                                                    number_connections_received,
                                                                                                                                                                                                                                                                                                                    neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_means + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_variances + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                    ptr_array_parameters_received,
                                                                                                                                                                                                                                                                                                                    ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
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
            kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                    *ptr_layer_it_received->ptr_dim3_block_neurons,
                                                                                                                                                                                                                    ptr_layer_it_received->ptr_dim3_block_neurons->x * sizeof(T) >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                    tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                    number_neurons_received,
                                                                                                                                                                                                                                                                                                                    number_connections_received,
                                                                                                                                                                                                                                                                                                                    neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_means + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_variances + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                    ptr_array_parameters_received,
                                                                                                                                                                                                                                                                                                                    ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
            
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    // |END| KERNEL LAUNCH |END|
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average(size_t const number_neurons_received,
                                                                                                                                                                                                size_t const batch_size_received,
                                                                                                                                                                                                size_t const total_data_batch_received,
                                                                                                                                                                                                bool const *const ptr_array_layer_it_masks_dropouts_received,
                                                                                                                                                                                                T const T_batch_size_received,
                                                                                                                                                                                                T const normalization_momentum_average_received,
                                                                                                                                                                                                T const epsilon_received,
                                                                                                                                                                                                T const r_correction_maximum_received,
                                                                                                                                                                                                T const d_correction_maximum_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_transposed_means_received,
                                                                                                                                                                                                T **const ptr_array_layer_it_reduce_means_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_transposed_variances_received,
                                                                                                                                                                                                T **const ptr_array_layer_it_reduce_variances_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_means_averages_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_variances_averages_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_r_correction_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_d_correction_received,
                                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_grid_reduce_batch_received,
                                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_block_reduce_batch_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average__0[];
    /* Index map:
        0: mean, value (d_correction)
        1: mean_average
        2: variance, value (r_correction)
        3: variance_average
        4: low (r_correction) */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average__0;

    if(ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
    {
        Reduce::Reduce<T>(batch_size_received,
                                        number_neurons_received,
                                        ptr_array_layer_it_reduce_means_received[tmp_thread_global_index],
                                        ptr_array_layer_it_transposed_means_received + tmp_thread_global_index * total_data_batch_received,
                                        ptr_array_dim3_grid_reduce_batch_received + tmp_thread_global_index,
                                        ptr_array_dim3_block_reduce_batch_received + tmp_thread_global_index);

        Reduce::Reduce<T>(batch_size_received,
                                        number_neurons_received,
                                        ptr_array_layer_it_reduce_variances_received[tmp_thread_global_index],
                                        ptr_array_layer_it_transposed_variances_received + tmp_thread_global_index * total_data_batch_received,
                                        ptr_array_dim3_grid_reduce_batch_received + tmp_thread_global_index,
                                        ptr_array_dim3_block_reduce_batch_received + tmp_thread_global_index);
    }

    // Do we need to synchronise? Based on "Reduce" Function.
    // => Synchronisation before using the reduced mean/variance of the neuron.
    if(batch_size_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    if(ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
    {
        // Average batch mean.
        // mean_b = sum(summation, N) / N
        ptr_array_layer_it_means_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x] = *(ptr_array_layer_it_reduce_means_received[tmp_thread_global_index]) / T_batch_size_received;

        // Average exponentialy global mean.
        // mean += momentum * (mean_b - mean)
        tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_layer_it_means_averages_received[tmp_thread_global_index] += normalization_momentum_average_received * (tmp_ptr_array_smem[threadIdx.x] - ptr_array_layer_it_means_averages_received[tmp_thread_global_index]); // Exponential moving average.
        
        // Average batch variance.
        // variance_b = sqrt( ((sum(pow(summation, 2), N) / N) - pow(mean_b, 2) + epsilon )
        ptr_array_layer_it_variances_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = sqrt(*(ptr_array_layer_it_reduce_variances_received[tmp_thread_global_index]) / T_batch_size_received - tmp_ptr_array_smem[threadIdx.x] * tmp_ptr_array_smem[threadIdx.x] + epsilon_received);
        
        // Average exponentialy global variance.
        // variance += momentum * (variance_b - variance)   "Risk of NaN if momentum equal 1.0"
        tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x] = ptr_array_layer_it_variances_averages_received[tmp_thread_global_index] += normalization_momentum_average_received * (tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] - ptr_array_layer_it_variances_averages_received[tmp_thread_global_index]); // Exponential moving average.
    
        // r correction.
        // value = variance_b / variance
        tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x];
        // low = 1 / r_correction_max
        tmp_ptr_array_smem[threadIdx.x + 4u * blockDim.x] = T(1) / r_correction_maximum_received;
        // high = r_correction_max
        // r_correction = clip(value, low, high)
        ptr_array_layer_it_r_correction_received[tmp_thread_global_index] = MyEA::Math::Clip<T_>(tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x], tmp_ptr_array_smem[threadIdx.x + 4u * blockDim.x], r_correction_maximum_received);

        // d correction.
        // value = (mean_b - mean) / variance
        tmp_ptr_array_smem[threadIdx.x] = (tmp_ptr_array_smem[threadIdx.x] - tmp_ptr_array_smem[threadIdx.x + blockDim.x]) / tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x];
        // low = -d_correction_max
        // high = d_correction_max
        // d_correction = clip(value, low, high)
        ptr_array_layer_it_d_correction_received[tmp_thread_global_index] = MyEA::Math::Clip<T_>(tmp_ptr_array_smem[threadIdx.x], -d_correction_maximum_received, d_correction_maximum_received);
    }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average(size_t const size_received,
                                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                                size_t const batch_size_received,
                                                                                                                                                                                                size_t const total_data_batch_received,
                                                                                                                                                                                                bool const *const ptr_array_layer_it_masks_dropouts_received,
                                                                                                                                                                                                T const T_batch_size_received,
                                                                                                                                                                                                T const normalization_momentum_average_received,
                                                                                                                                                                                                T const epsilon_received,
                                                                                                                                                                                                T const r_correction_maximum_received,
                                                                                                                                                                                                T const d_correction_maximum_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_transposed_means_received,
                                                                                                                                                                                                T **const ptr_array_layer_it_reduce_means_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_transposed_variances_received,
                                                                                                                                                                                                T **const ptr_array_layer_it_reduce_variances_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_means_averages_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_variances_averages_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_r_correction_received,
                                                                                                                                                                                                T *const ptr_array_layer_it_d_correction_received,
                                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_grid_reduce_batch_received,
                                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_block_reduce_batch_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average__1[];
    /* Index map:
        0: mean, value (d_correction)
        1: mean_average
        2: variance, value (r_correction)
        3: variance_average
        4: low (r_correction) */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average__1;
    
    if(tmp_thread_global_index < size_received && ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
    {
        Reduce::Reduce<T>(batch_size_received,
                                      number_neurons_received,
                                      ptr_array_layer_it_reduce_means_received[tmp_thread_global_index],
                                      ptr_array_layer_it_transposed_means_received + tmp_thread_global_index * total_data_batch_received,
                                      ptr_array_dim3_grid_reduce_batch_received + tmp_thread_global_index,
                                      ptr_array_dim3_block_reduce_batch_received + tmp_thread_global_index);

        Reduce::Reduce<T>(batch_size_received,
                                      number_neurons_received,
                                      ptr_array_layer_it_reduce_variances_received[tmp_thread_global_index],
                                      ptr_array_layer_it_transposed_variances_received + tmp_thread_global_index * total_data_batch_received,
                                      ptr_array_dim3_grid_reduce_batch_received + tmp_thread_global_index,
                                      ptr_array_dim3_block_reduce_batch_received + tmp_thread_global_index);
    }
    
    // Do we need to synchronise? Based on "Reduce" Function.
    // => Synchronisation before using the reduced mean/variance of the neuron.
    if(batch_size_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    if(tmp_thread_global_index < size_received && ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
    {
        // Average batch mean.
        // mean_b = sum(summation, N) / N
        ptr_array_layer_it_means_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x] = *(ptr_array_layer_it_reduce_means_received[tmp_thread_global_index]) / T_batch_size_received;

        // Average exponentialy global mean.
        // mean += momentum * (mean_b - mean)
        tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_layer_it_means_averages_received[tmp_thread_global_index] += normalization_momentum_average_received * (tmp_ptr_array_smem[threadIdx.x] - ptr_array_layer_it_means_averages_received[tmp_thread_global_index]); // Exponential moving average.
        
        // Average batch variance.
        // variance_b = sqrt( ((sum(pow(summation, 2), N) / N) - pow(mean_b, 2) + epsilon )
        ptr_array_layer_it_variances_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = sqrt(*(ptr_array_layer_it_reduce_variances_received[tmp_thread_global_index]) / T_batch_size_received - tmp_ptr_array_smem[threadIdx.x] * tmp_ptr_array_smem[threadIdx.x] + epsilon_received);
        
        // Average exponentialy global variance.
        // variance += momentum * (variance_b - variance)   "Risk of NaN if momentum equal 1.0"
        tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x] = ptr_array_layer_it_variances_averages_received[tmp_thread_global_index] += normalization_momentum_average_received * (tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] - ptr_array_layer_it_variances_averages_received[tmp_thread_global_index]); // Exponential moving average.
    
        // r correction.
        // value = variance_b / variance
        tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x];
        // low = 1 / r_correction_max
        tmp_ptr_array_smem[threadIdx.x + 4u * blockDim.x] = T(1) / r_correction_maximum_received;
        // high = r_correction_max
        // r_correction = clip(value, low, high)
        ptr_array_layer_it_r_correction_received[tmp_thread_global_index] = MyEA::Math::Clip<T_>(tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x], tmp_ptr_array_smem[threadIdx.x + 4u * blockDim.x], r_correction_maximum_received);

        // d correction.
        // value = (mean_b - mean) / variance
        tmp_ptr_array_smem[threadIdx.x] = (tmp_ptr_array_smem[threadIdx.x] - tmp_ptr_array_smem[threadIdx.x + blockDim.x]) / tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x];
        // low = -d_correction_max
        // high = d_correction_max
        // d_correction = clip(value, low, high)
        ptr_array_layer_it_d_correction_received[tmp_thread_global_index] = MyEA::Math::Clip<T_>(tmp_ptr_array_smem[threadIdx.x], -d_correction_maximum_received, d_correction_maximum_received);
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average(size_t const size_received,
                                                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                                                        size_t const batch_size_received,
                                                                                                                                                                                                        size_t const total_data_batch_received,
                                                                                                                                                                                                        bool const *const ptr_array_layer_it_masks_dropouts_received,
                                                                                                                                                                                                        T const T_batch_size_received,
                                                                                                                                                                                                        T const normalization_momentum_average_received,
                                                                                                                                                                                                        T const epsilon_received,
                                                                                                                                                                                                        T const r_correction_maximum_received,
                                                                                                                                                                                                        T const d_correction_maximum_received,
                                                                                                                                                                                                        T *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                        T *const ptr_array_layer_it_transposed_means_received,
                                                                                                                                                                                                        T **const ptr_array_layer_it_reduce_means_received,
                                                                                                                                                                                                        T *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                        T *const ptr_array_layer_it_transposed_variances_received,
                                                                                                                                                                                                        T **const ptr_array_layer_it_reduce_variances_received,
                                                                                                                                                                                                        T *const ptr_array_layer_it_means_averages_received,
                                                                                                                                                                                                        T *const ptr_array_layer_it_variances_averages_received,
                                                                                                                                                                                                        T *const ptr_array_layer_it_r_correction_received,
                                                                                                                                                                                                        T *const ptr_array_layer_it_d_correction_received,
                                                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_reduce_batch_received,
                                                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_block_reduce_batch_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average[];
    /* Index map:
        0: mean, value (d_correction)
        1: mean_average
        2: variance, value (r_correction)
        3: variance_average
        4: low (r_correction) */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average;

    // Loop through each neurons.
    do
    {
        if(ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
        {
            Reduce::Reduce<T>(batch_size_received,
                                          number_neurons_received,
                                          ptr_array_layer_it_reduce_means_received[tmp_thread_global_index],
                                          ptr_array_layer_it_transposed_means_received + tmp_thread_global_index * total_data_batch_received,
                                          ptr_array_dim3_grid_reduce_batch_received + tmp_thread_global_index,
                                          ptr_array_dim3_block_reduce_batch_received + tmp_thread_global_index);

            Reduce::Reduce<T>(batch_size_received,
                                          number_neurons_received,
                                          ptr_array_layer_it_reduce_variances_received[tmp_thread_global_index],
                                          ptr_array_layer_it_transposed_variances_received + tmp_thread_global_index * total_data_batch_received,
                                          ptr_array_dim3_grid_reduce_batch_received + tmp_thread_global_index,
                                          ptr_array_dim3_block_reduce_batch_received + tmp_thread_global_index);
        }

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce" Function.
    // => Synchronisation before using the reduced mean/variance of the neuron.
    if(batch_size_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // Reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop through each neurons.
    do
    {
        if(ptr_array_layer_it_masks_dropouts_received[tmp_thread_global_index])
        {
            // Average batch mean.
            // mean_b = sum(summation, N) / N
            ptr_array_layer_it_means_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x] = *(ptr_array_layer_it_reduce_means_received[tmp_thread_global_index]) / T_batch_size_received;

            // Average exponentialy global mean.
            // mean += momentum * (mean_b - mean)
            tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_layer_it_means_averages_received[tmp_thread_global_index] += normalization_momentum_average_received * (tmp_ptr_array_smem[threadIdx.x] - ptr_array_layer_it_means_averages_received[tmp_thread_global_index]); // Exponential moving average.
        
            // Average batch variance.
            // variance_b = sqrt( ((sum(pow(summation, 2), N) / N) - pow(mean_b, 2) + epsilon )
            ptr_array_layer_it_variances_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = sqrt(*(ptr_array_layer_it_reduce_variances_received[tmp_thread_global_index]) / T_batch_size_received - tmp_ptr_array_smem[threadIdx.x] * tmp_ptr_array_smem[threadIdx.x] + epsilon_received);
        
            // Average exponentialy global variance.
            // variance += momentum * (variance_b - variance)   "Risk of NaN if momentum equal 1.0"
            tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x] = ptr_array_layer_it_variances_averages_received[tmp_thread_global_index] += normalization_momentum_average_received * (tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] - ptr_array_layer_it_variances_averages_received[tmp_thread_global_index]); // Exponential moving average.
    
            // r correction.
            // value = variance_b / variance
            tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x];
            // low = 1 / r_correction_max
            tmp_ptr_array_smem[threadIdx.x + 4u * blockDim.x] = T(1) / r_correction_maximum_received;
            // high = r_correction_max
            // r_correction = clip(value, low, high)
            ptr_array_layer_it_r_correction_received[tmp_thread_global_index] = MyEA::Math::Clip<T_>(tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x], tmp_ptr_array_smem[threadIdx.x + 4u * blockDim.x], r_correction_maximum_received);

            // d correction.
            // value = (mean_b - mean) / variance
            tmp_ptr_array_smem[threadIdx.x] = (tmp_ptr_array_smem[threadIdx.x] - tmp_ptr_array_smem[threadIdx.x + blockDim.x]) / tmp_ptr_array_smem[threadIdx.x + 3u * blockDim.x];
            // low = -d_correction_max
            // high = d_correction_max
            // d_correction = clip(value, low, high)
            ptr_array_layer_it_d_correction_received[tmp_thread_global_index] = MyEA::Math::Clip<T_>(tmp_ptr_array_smem[threadIdx.x], -d_correction_maximum_received, d_correction_maximum_received);
        }

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Activations(size_t const number_neurons_received,
                                                                                                                                                                                                                   struct CUDA_Neuron *ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                                   struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Activations_0[];
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Activations_0;
    
    // Loop through each neurons for doing a reduction of summation.
    for(; ptr_layer_it_first_neuron_received != ptr_layer_it_last_neuron_received; ++ptr_layer_it_first_neuron_received)
    {
        if(*ptr_layer_it_first_neuron_received->ptr_mask_dropout_bernoulli)
        {
            tmp_ptr_array_smem[threadIdx.x] = ptr_layer_it_first_neuron_received->ptr_array_summations[tmp_thread_global_index * number_neurons_received];
        
            // Normalize.
            // value_hat = (summation - mean_b) / variance_b * r_correction + d_correction
            ptr_layer_it_first_neuron_received->ptr_array_values_hats[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_smem[threadIdx.x] = (tmp_ptr_array_smem[threadIdx.x] - *ptr_layer_it_first_neuron_received->ptr_array_means) / *ptr_layer_it_first_neuron_received->ptr_array_variances * *ptr_layer_it_first_neuron_received->ptr_r_correction + *ptr_layer_it_first_neuron_received->ptr_d_correction;
        
            // Scale and shift.
            // value_normalize = scale * value_hat + shift
            ptr_layer_it_first_neuron_received->ptr_array_values_normalizes[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_smem[threadIdx.x] = *ptr_layer_it_first_neuron_received->ptr_scale * tmp_ptr_array_smem[threadIdx.x] + *ptr_layer_it_first_neuron_received->ptr_shift;
        
            // Activation.
            // value = AF(value_normalize)
            Activation_Real(ptr_layer_it_first_neuron_received->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                   tmp_ptr_array_smem[threadIdx.x],
                                   *ptr_layer_it_first_neuron_received->ptr_type_activation_function);
        }
    }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Activations(size_t const size_received,
                                                                                                                                                                                                                   size_t const number_neurons_received,
                                                                                                                                                                                                                   struct CUDA_Neuron *ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                                   struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel_if__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Activations_1[];
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel_if__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Activations_1;

    if(tmp_thread_global_index < size_received)
    {
        // Loop through each neurons for doing a reduction of summation.
        for(; ptr_layer_it_first_neuron_received != ptr_layer_it_last_neuron_received; ++ptr_layer_it_first_neuron_received)
        {
            if(*ptr_layer_it_first_neuron_received->ptr_mask_dropout_bernoulli)
            {
                tmp_ptr_array_smem[threadIdx.x] = ptr_layer_it_first_neuron_received->ptr_array_summations[tmp_thread_global_index * number_neurons_received];
            
                // Normalize.
                // value_hat = (summation - mean_b) / variance_b * r_correction + d_correction
                ptr_layer_it_first_neuron_received->ptr_array_values_hats[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_smem[threadIdx.x] = (tmp_ptr_array_smem[threadIdx.x] - *ptr_layer_it_first_neuron_received->ptr_array_means) / *ptr_layer_it_first_neuron_received->ptr_array_variances * *ptr_layer_it_first_neuron_received->ptr_r_correction + *ptr_layer_it_first_neuron_received->ptr_d_correction;
            
                // Scale and shift.
                // value_normalize = scale * value_hat + shift
                ptr_layer_it_first_neuron_received->ptr_array_values_normalizes[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_smem[threadIdx.x] = *ptr_layer_it_first_neuron_received->ptr_scale * tmp_ptr_array_smem[threadIdx.x] + *ptr_layer_it_first_neuron_received->ptr_shift;
            
                // Activation.
                // value = AF(value_normalize)
                Activation_Real(ptr_layer_it_first_neuron_received->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                        tmp_ptr_array_smem[threadIdx.x],
                                        *ptr_layer_it_first_neuron_received->ptr_type_activation_function);
            }
        }
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Activations(size_t const size_received,
                                                                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                                                                        struct CUDA_Neuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                                        struct CUDA_Neuron const *const ptr_layer_it_last_neuron_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron *tmp_ptr_neuron_unit_it;
    
    extern __shared__ T tmp_shared_T__kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Activations[];
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Activations;

    // Loop through each sample.
    do
    {
        // Loop through each neurons for doing a reduction of summation.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
            {
                tmp_ptr_array_smem[threadIdx.x] = tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received];
            
                // Normalize.
                // value_hat = (summation - mean_b) / variance_b * r_correction + d_correction
                tmp_ptr_neuron_unit_it->ptr_array_values_hats[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_smem[threadIdx.x] = (tmp_ptr_array_smem[threadIdx.x] - *tmp_ptr_neuron_unit_it->ptr_array_means) / *tmp_ptr_neuron_unit_it->ptr_array_variances * *tmp_ptr_neuron_unit_it->ptr_d_correction + *tmp_ptr_neuron_unit_it->ptr_d_correction;
            
                // Scale and shift.
                // value_normalize = scale * value_hat + shift
                tmp_ptr_neuron_unit_it->ptr_array_values_normalizes[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_smem[threadIdx.x] = *tmp_ptr_neuron_unit_it->ptr_scale * tmp_ptr_array_smem[threadIdx.x] + *tmp_ptr_neuron_unit_it->ptr_shift;
            
                // Activation.
                // value = AF(value_normalize)
                Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                        tmp_ptr_array_smem[threadIdx.x],
                                        *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
            }
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Parallel_Activation(size_t const number_neurons_received, struct CUDA_Layer const *const ptr_layer_it_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron const *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation<T>,
                                                        ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                        ptr_layer_it_received->ptr_dim3_block_neurons,
                                                        ptr_layer_it_received->ptr_dim3_block_neurons->x * sizeof(T),
                                                        number_neurons_received - 1u, // Subtract bias.
                                                        tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                        tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                        tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                        tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function)
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Parallel_Activation(size_t const size_received,
                                                                                                                                                                                                              size_t const number_neurons_received,
                                                                                                                                                                                                              struct CUDA_Layer const *const ptr_layer_it_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron const *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    if(tmp_thread_global_index < size_received)
    {
        LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation<T>,
                                                            ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                            ptr_layer_it_received->ptr_dim3_block_neurons,
                                                            ptr_layer_it_received->ptr_dim3_block_neurons->x * sizeof(T),
                                                            number_neurons_received - 1u, // Subtract bias.
                                                            tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                            tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                            tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                            tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                            tmp_ptr_layer_it_first_neuron->ptr_type_activation_function)
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Parallel_Activation(size_t const size_received,
                                                                                                                                                                                                                       size_t const number_neurons_received,
                                                                                                                                                                                                                       struct CUDA_Layer const *const ptr_layer_it_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct CUDA_Neuron const *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    // KERNEL LAUNCH
    //    1: Launching do-while elements.
    if(ptr_layer_it_received->ptr_dim3_grid_neurons->x * ptr_layer_it_received->ptr_dim3_block_neurons->x < number_neurons_received - 1u) // Subtract bias.
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a activation.
            kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                          *ptr_layer_it_received->ptr_dim3_block_neurons,
                                                                                                                                                                                                                          ptr_layer_it_received->ptr_dim3_block_neurons->x * sizeof(T) >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function);
            
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    //    2: Launching size condition.
    else if(ptr_layer_it_received->ptr_dim3_grid_neurons->x * ptr_layer_it_received->ptr_dim3_block_neurons->x > number_neurons_received - 1u) // Subtract bias.
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a activation.
            kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                *ptr_layer_it_received->ptr_dim3_block_neurons,
                                                                                                                                                                                                                ptr_layer_it_received->ptr_dim3_block_neurons->x * sizeof(T) >>> (number_neurons_received - 1u, // Subtract bias.
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_type_activation_function);
            
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    //    3: Standard.
    else
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a activation.
            kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation<T> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                *ptr_layer_it_received->ptr_dim3_block_neurons,
                                                                                                                                                                                                                ptr_layer_it_received->ptr_dim3_block_neurons->x * sizeof(T) >>> (tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_type_activation_function);
            
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    // |END| KERNEL LAUNCH |END|
}

__device__ void CUDA_Neural_Network::Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout(bool &ref_synchronized_received,
                                                                                                                                                                       size_t const batch_size_received,
                                                                                                                                                                       struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                                                       struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                                                                       struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                                                       struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    size_t tmp_data_index;
    
    T_ const *const tmp_ptr_array_previous_layer_outputs_begin(ptr_previous_layer_it_received->ptr_array_neuron_units->ptr_array_values),
                  *tmp_ptr_array_previous_layer_outputs,
                  *tmp_ptr_array_parameters,
                  tmp_r_correction_maximum(this->batch_renormalization_r_correction_maximum),
                  tmp_d_correction_maximum(this->batch_renormalization_d_correction_maximum),
                  tmp_epsilon(this->normalization_epsilon);
    T_ tmp_summation,
         tmp_gamma,
         tmp_mean,
         tmp_variance,
         tmp_mean_average,
         tmp_variance_average,
         tmp_r_correction,
         tmp_d_correction;

    struct CUDA_Neuron const *const tmp_ptr_layer_it_last_neuron(ptr_layer_it_received->ptr_last_neuron_unit - 1); // Subtract bias.
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units),
                                             *tmp_ptr_neuron_unit_it;

    size_t const tmp_number_connections(*tmp_ptr_layer_it_first_neuron->ptr_number_forward_connections - 1u), // Subtract bias.
                                tmp_number_neuron_units(*ptr_layer_it_received->ptr_number_neurons);

    // Condition to enter into dynamic parallelisme of each sample.
    if(USE_PARALLEL && batch_size_received >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;
        
        // Condition to enter into dynamic parallelisme of each sample and neurons.
        if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
        {
            // KERNEL LAUNCH
            //    1: Launching do-while elements.
            if(ptr_dim3_batch_size_grid_received->x * ptr_dim3_batch_size_block_received->x < batch_size_received)
            {
                kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Parallel_Summation<T_> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (batch_size_received,
                                                                                                                                                                                                                                                                                                                              tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                              tmp_number_connections,
                                                                                                                                                                                                                                                                                                                              this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                              this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                              tmp_ptr_array_previous_layer_outputs_begin,
                                                                                                                                                                                                                                                                                                                              ptr_layer_it_received);
                
                this->Transpose_Layer_Forward__Batch_Normalization(ptr_layer_it_received);
                
                LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average<T_>,
                                                                    ptr_layer_it_received->ptr_dim3_grid_neurons_DP,
                                                                    ptr_layer_it_received->ptr_dim3_block_neurons_DP,
                                                                    ptr_layer_it_received->ptr_dim3_block_neurons_DP->x * 5u * sizeof(T_),
                                                                    tmp_number_neuron_units - 1u, // Subtract bias.
                                                                    tmp_number_neuron_units,
                                                                    batch_size_received,
                                                                    this->batch_size,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                    static_cast<T_>(batch_size_received),
                                                                    this->normalization_momentum_average,
                                                                    tmp_epsilon,
                                                                    tmp_r_correction_maximum,
                                                                    tmp_d_correction_maximum,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_mean,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_variance,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_threads,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_threads)

                kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Parallel_Activation<T_> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (batch_size_received,
                                                                                                                                                                                                                                                                                                                                    tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                    ptr_layer_it_received);
            }
            //    2: Launching size condition.
            else if(ptr_dim3_batch_size_grid_received->x * ptr_dim3_batch_size_block_received->x > batch_size_received)
            {
                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Parallel_Summation<T_> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (batch_size_received,
                                                                                                                                                                                                                                                                                                                     tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                     tmp_number_connections,
                                                                                                                                                                                                                                                                                                                     this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                     this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                     tmp_ptr_array_previous_layer_outputs_begin,
                                                                                                                                                                                                                                                                                                                     ptr_layer_it_received);
                
                this->Transpose_Layer_Forward__Batch_Normalization(ptr_layer_it_received);
                
                LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average<T_>,
                                                                    ptr_layer_it_received->ptr_dim3_grid_neurons_DP,
                                                                    ptr_layer_it_received->ptr_dim3_block_neurons_DP,
                                                                    ptr_layer_it_received->ptr_dim3_block_neurons_DP->x * 5u * sizeof(T_),
                                                                    tmp_number_neuron_units - 1u, // Subtract bias.
                                                                    tmp_number_neuron_units,
                                                                    batch_size_received,
                                                                    this->batch_size,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                    static_cast<T_>(batch_size_received),
                                                                    this->normalization_momentum_average,
                                                                    tmp_epsilon,
                                                                    tmp_r_correction_maximum,
                                                                    tmp_d_correction_maximum,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_mean,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_variance,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_threads,
                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_threads)

                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Parallel_Activation<T_> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (batch_size_received,
                                                                                                                                                                                                                                                                                                                           tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                           ptr_layer_it_received);
            }
            //    3: Standard.
            else
            {
                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Parallel_Summation<T_> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                             tmp_number_connections,
                                                                                                                                                                                                                                                                                                                             this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                             this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                             tmp_ptr_array_previous_layer_outputs_begin,
                                                                                                                                                                                                                                                                                                                             ptr_layer_it_received);
                
                this->Transpose_Layer_Forward__Batch_Normalization(ptr_layer_it_received);
                
                LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average<T_>,
                                                                  ptr_layer_it_received->ptr_dim3_grid_neurons_DP,
                                                                  ptr_layer_it_received->ptr_dim3_block_neurons_DP,
                                                                  ptr_layer_it_received->ptr_dim3_block_neurons_DP->x * 5u * sizeof(T_),
                                                                  tmp_number_neuron_units - 1u, // Subtract bias.
                                                                  tmp_number_neuron_units,
                                                                  batch_size_received,
                                                                  this->batch_size,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                  static_cast<T_>(batch_size_received),
                                                                  this->normalization_momentum_average,
                                                                  tmp_epsilon,
                                                                  tmp_r_correction_maximum,
                                                                  tmp_d_correction_maximum,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_threads,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_threads)

                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Parallel_Activation<T_> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (tmp_number_neuron_units, ptr_layer_it_received);
            }
            // |END| KERNEL LAUNCH |END|
        }
        // Condition to enter into dynamic parallelisme of each sample.
        else
        {
            // KERNEL LAUNCH
            //    1: Launching do-while elements.
            if(ptr_dim3_batch_size_grid_received->x * ptr_dim3_batch_size_block_received->x < batch_size_received)
            {
                kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Summation<T_> <<< *ptr_dim3_batch_size_grid_received,
                                                                                                                                                                                                                                   *ptr_dim3_batch_size_block_received,
                                                                                                                                                                                                                                   ptr_dim3_batch_size_block_received->x * sizeof(T_) >>> (batch_size_received,
                                                                                                                                                                                                                                                                                                                        tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                        tmp_number_connections,
                                                                                                                                                                                                                                                                                                                        this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                        this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_array_previous_layer_outputs_begin,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_last_neuron);
                
                this->Transpose_Layer_Forward__Batch_Normalization(ptr_layer_it_received);
                
                LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average<T_>,
                                                                  ptr_layer_it_received->ptr_dim3_grid_neurons_DP,
                                                                  ptr_layer_it_received->ptr_dim3_block_neurons_DP,
                                                                  ptr_layer_it_received->ptr_dim3_block_neurons_DP->x * 5u * sizeof(T_),
                                                                  tmp_number_neuron_units - 1u, // Subtract bias.
                                                                  tmp_number_neuron_units,
                                                                  batch_size_received,
                                                                  this->batch_size,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                  static_cast<T_>(batch_size_received),
                                                                  this->normalization_momentum_average,
                                                                  tmp_epsilon,
                                                                  tmp_r_correction_maximum,
                                                                  tmp_d_correction_maximum,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_threads,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_threads)
                    
                kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Activations<T_> <<< *ptr_dim3_batch_size_grid_received,
                                                                                                                                                                                                                                  *ptr_dim3_batch_size_block_received,
                                                                                                                                                                                                                                  ptr_dim3_batch_size_block_received->x * sizeof(T_) >>> (batch_size_received,
                                                                                                                                                                                                                                                                                                                      tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron,
                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_last_neuron);
            }
            //    2: Launching size condition.
            else if(ptr_dim3_batch_size_grid_received->x * ptr_dim3_batch_size_block_received->x > batch_size_received)
            {
                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Summation<T_> <<< *ptr_dim3_batch_size_grid_received,
                                                                                                                                                                                                                         *ptr_dim3_batch_size_block_received,
                                                                                                                                                                                                                         ptr_dim3_batch_size_block_received->x * sizeof(T_) >>> (batch_size_received,
                                                                                                                                                                                                                                                                                                             tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                             tmp_number_connections,
                                                                                                                                                                                                                                                                                                             this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                             this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                             tmp_ptr_array_previous_layer_outputs_begin,
                                                                                                                                                                                                                                                                                                             tmp_ptr_layer_it_first_neuron,
                                                                                                                                                                                                                                                                                                             tmp_ptr_layer_it_last_neuron);
                
                this->Transpose_Layer_Forward__Batch_Normalization(ptr_layer_it_received);
                
                LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average<T_>,
                                                                  ptr_layer_it_received->ptr_dim3_grid_neurons_DP,
                                                                  ptr_layer_it_received->ptr_dim3_block_neurons_DP,
                                                                  ptr_layer_it_received->ptr_dim3_block_neurons_DP->x * 5u * sizeof(T_),
                                                                  tmp_number_neuron_units - 1u, // Subtract bias.
                                                                  tmp_number_neuron_units,
                                                                  batch_size_received,
                                                                  this->batch_size,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                  static_cast<T_>(batch_size_received),
                                                                  this->normalization_momentum_average,
                                                                  tmp_epsilon,
                                                                  tmp_r_correction_maximum,
                                                                  tmp_d_correction_maximum,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_threads,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_threads)
                    
                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Activations<T_> <<< *ptr_dim3_batch_size_grid_received,
                                                                                                                                                                                                                          *ptr_dim3_batch_size_block_received,
                                                                                                                                                                                                                          ptr_dim3_batch_size_block_received->x * sizeof(T_) >>> (batch_size_received,
                                                                                                                                                                                                                                                                                                              tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                              tmp_ptr_layer_it_first_neuron,
                                                                                                                                                                                                                                                                                                              tmp_ptr_layer_it_last_neuron);
            }
            //    3: Standard.
            else
            {
                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Summation<T_> <<< *ptr_dim3_batch_size_grid_received,
                                                                                                                                                                                                                          *ptr_dim3_batch_size_block_received,
                                                                                                                                                                                                                          ptr_dim3_batch_size_block_received->x * sizeof(T_) >>> (tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                              tmp_number_connections,
                                                                                                                                                                                                                                                                                                              this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                              this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                              tmp_ptr_array_previous_layer_outputs_begin,
                                                                                                                                                                                                                                                                                                              tmp_ptr_layer_it_first_neuron,
                                                                                                                                                                                                                                                                                                              tmp_ptr_layer_it_last_neuron);
                
                this->Transpose_Layer_Forward__Batch_Normalization(ptr_layer_it_received);
                
                LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Reduce_Average<T_>,
                                                                  ptr_layer_it_received->ptr_dim3_grid_neurons_DP,
                                                                  ptr_layer_it_received->ptr_dim3_block_neurons_DP,
                                                                  ptr_layer_it_received->ptr_dim3_block_neurons_DP->x * 5u * sizeof(T_),
                                                                  tmp_number_neuron_units - 1u, // Subtract bias.
                                                                  tmp_number_neuron_units,
                                                                  batch_size_received,
                                                                  this->batch_size,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                  static_cast<T_>(batch_size_received),
                                                                  this->normalization_momentum_average,
                                                                  tmp_epsilon,
                                                                  tmp_r_correction_maximum,
                                                                  tmp_d_correction_maximum,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_threads,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_threads)

                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Batch__Serialize_Activations<T_> <<< *ptr_dim3_batch_size_grid_received,
                                                                                                                                                                                                                         *ptr_dim3_batch_size_block_received,
                                                                                                                                                                                                                         ptr_dim3_batch_size_block_received->x * sizeof(T_) >>> (tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                             tmp_ptr_layer_it_first_neuron,
                                                                                                                                                                                                                                                                                                             tmp_ptr_layer_it_last_neuron);
            }
            // |END| KERNEL LAUNCH |END|
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
                kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP,
                                                                                                                                                                                                                                   *ptr_layer_it_received->ptr_dim3_block_neurons_DP,
                                                                                                                                                                                                                                   ptr_layer_it_received->ptr_dim3_block_neurons_DP->x * sizeof(T_) >>> (tmp_number_neuron_units - 1u,
                                                                                                                                                                                                                                                                                                                                            tmp_data_index,
                                                                                                                                                                                                                                                                                                                                            tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                            this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                                            this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_array_previous_layer_outputs_begin + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
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
                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP,
                                                                                                                                                                                                                          *ptr_layer_it_received->ptr_dim3_block_neurons_DP,
                                                                                                                                                                                                                          ptr_layer_it_received->ptr_dim3_block_neurons_DP->x * sizeof(T_) >>> (tmp_number_neuron_units - 1u,
                                                                                                                                                                                                                                                                                                                                    tmp_data_index,
                                                                                                                                                                                                                                                                                                                                    tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                    tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                    this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                                    this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_array_previous_layer_outputs_begin + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
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
                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Summation<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons_DP,
                                                                                                                                                                                                                          *ptr_layer_it_received->ptr_dim3_block_neurons_DP,
                                                                                                                                                                                                                          ptr_layer_it_received->ptr_dim3_block_neurons_DP->x * sizeof(T_) >>> (tmp_data_index,
                                                                                                                                                                                                                                                                                                                                tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                                this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_array_previous_layer_outputs_begin + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
            }
        }
        // |END| KERNEL LAUNCH |END|
        
        // KERNEL LAUNCH
        //    1: Launching do-while elements.
        if(ptr_layer_it_received->ptr_dim3_grid_neurons->x * ptr_layer_it_received->ptr_dim3_block_neurons->x < tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Make the average.
            kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Average<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                                                                                                                                                               *ptr_layer_it_received->ptr_dim3_block_neurons,
                                                                                                                                                                                               ptr_layer_it_received->ptr_dim3_block_neurons->x * 5u * sizeof(T_) >>> (tmp_number_neuron_units - 1u,
                                                                                                                                                                                                                                                                                                       tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                       static_cast<T_>(batch_size_received),
                                                                                                                                                                                                                                                                                                       this->normalization_momentum_average,
                                                                                                                                                                                                                                                                                                       tmp_epsilon,
                                                                                                                                                                                                                                                                                                       tmp_r_correction_maximum,
                                                                                                                                                                                                                                                                                                       tmp_d_correction_maximum,
                                                                                                                                                                                                                                                                                                       tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                       tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                       tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                                                                                                                                                                                                                                                       tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                                                                                                                                                                                                                                                                       tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                                                                                                                                                                                                                                                       tmp_ptr_layer_it_first_neuron->ptr_d_correction);
            
            // Loop through each sample.
            for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
            {
                // Parallel each neurons for doing a activation function.
                kernel_while__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                                *ptr_layer_it_received->ptr_dim3_block_neurons,
                                                                                                                                                                                                                                ptr_layer_it_received->ptr_dim3_block_neurons->x * sizeof(T_) >>> (tmp_number_neuron_units - 1u,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_type_activation_function);
            }
        }
        //    2: Launching size condition.
        else if(ptr_layer_it_received->ptr_dim3_grid_neurons->x * ptr_layer_it_received->ptr_dim3_block_neurons->x > tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Make the average.
            kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Average<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                                                                                                                                                      *ptr_layer_it_received->ptr_dim3_block_neurons,
                                                                                                                                                                                      ptr_layer_it_received->ptr_dim3_block_neurons->x * 5u * sizeof(T_) >>> (tmp_number_neuron_units - 1u,
                                                                                                                                                                                                                                                                                               tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                static_cast<T_>(batch_size_received),
                                                                                                                                                                                                                                                                                                this->normalization_momentum_average,
                                                                                                                                                                                                                                                                                                tmp_epsilon,
                                                                                                                                                                                                                                                                                                tmp_r_correction_maximum,
                                                                                                                                                                                                                                                                                                tmp_d_correction_maximum,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_d_correction);
            
            // Loop through each sample.
            for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
            {
                // Parallel each neurons for doing a activation function.
                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                        *ptr_layer_it_received->ptr_dim3_block_neurons,
                                                                                                                                                                                                                        ptr_layer_it_received->ptr_dim3_block_neurons->x * sizeof(T_) >>> (tmp_number_neuron_units - 1u,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function);
            }
        }
        //    3: Standard.
        else
        {
            // Make the average.
            kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Parallel_Average<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                                                                                                                                                      *ptr_layer_it_received->ptr_dim3_block_neurons,
                                                                                                                                                                                      ptr_layer_it_received->ptr_dim3_block_neurons->x * 5u * sizeof(T_) >>> (tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                static_cast<T_>(batch_size_received),
                                                                                                                                                                                                                                                                                                this->normalization_momentum_average,
                                                                                                                                                                                                                                                                                                tmp_epsilon,
                                                                                                                                                                                                                                                                                                tmp_r_correction_maximum,
                                                                                                                                                                                                                                                                                                tmp_d_correction_maximum,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_mean_average,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_variance_average,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_d_correction);
            
            // Loop through each sample.
            for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
            {
                // Parallel each neurons for doing a activation function.
                kernel__Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Serialize_Batch__Parallel_Activation<T_> <<< *ptr_layer_it_received->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                        *ptr_layer_it_received->ptr_dim3_block_neurons,
                                                                                                                                                                                                                        ptr_layer_it_received->ptr_dim3_block_neurons->x * sizeof(T_) >>> (tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_shift,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_d_correction,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function);
            }
        }
        // |END| KERNEL LAUNCH |END|
    }
    // If we don't enter into dynamic parallelisme, we serialize the computation.
    else
    {
        // Synchronize if needed to see the output of the previous layer.
        CUDA__Device_Synchronise(ref_synchronized_received, MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);

        // Loop through each sample.
        for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
        {
            tmp_ptr_array_previous_layer_outputs = tmp_ptr_array_previous_layer_outputs_begin + tmp_data_index * (tmp_number_connections + 1u); // Add bias.

            // Loop through each neurons for doing a reduction of summation.
            for(tmp_ptr_array_parameters = this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_layer_it_last_neuron; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                     tmp_ptr_array_parameters += tmp_number_connections + 1u) // Add bias.
            {
                if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
                {
                    Reduce::Reduce_XX<T_>(tmp_number_connections,
                                                         tmp_number_neuron_units,
                                                         *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_data_index * this->neurons_total_reduce_summation_size,
                                                         tmp_ptr_array_parameters,
                                                         tmp_ptr_array_previous_layer_outputs,
                                                         tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                          tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);
                }
                else { tmp_ptr_neuron_unit_it->ptr_array_values[tmp_data_index * tmp_number_neuron_units] = 0_T; }
            }
        }
        
        // Do we need to synchronise? Based on "Reduce_XX" Function.
        // => Synchronize if needed to see the summation reduced of the layer.
        if(tmp_number_connections >= warpSize * 2u) { CUDA__Check_Error(); }

        // Loop through each sample.
        for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
        {
            // Loop through each neurons for retrieve reduced summation and then do a summation of mean and variance.
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_layer_it_last_neuron; ++tmp_ptr_neuron_unit_it)
            {
                if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
                {
                    tmp_summation = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_data_index * this->neurons_total_reduce_summation_size);
                
                    // mean += summation
                    *tmp_ptr_neuron_unit_it->ptr_array_means += tmp_summation;
                    // variance += pow(summation, 2)
                    *tmp_ptr_neuron_unit_it->ptr_array_variances += tmp_summation * tmp_summation;

                    tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units] = tmp_summation;
                }
            }
        }

        // Make the average.
        for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_layer_it_last_neuron; ++tmp_ptr_neuron_unit_it)
        {
            if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
            {
                // Average batch mean.
                // mean_b = sum(summation, N) / N
                *tmp_ptr_neuron_unit_it->ptr_array_means = tmp_mean = *tmp_ptr_neuron_unit_it->ptr_array_means / static_cast<T_>(batch_size_received);

                // Average exponentialy global mean.
                // mean += momentum * (mean_b - mean)
                tmp_mean_average = *tmp_ptr_neuron_unit_it->ptr_mean_average += this->normalization_momentum_average * (tmp_mean - *tmp_ptr_neuron_unit_it->ptr_mean_average); // Exponential moving average.
            
                // Average batch variance.
                // variance_b = sqrt( ((sum(pow(summation, 2), N) / N) - pow(mean_b, 2) + epsilon )
                *tmp_ptr_neuron_unit_it->ptr_array_variances = tmp_variance = sqrt(*tmp_ptr_neuron_unit_it->ptr_array_variances / static_cast<T_>(batch_size_received) - tmp_mean * tmp_mean + tmp_epsilon);
            
                // Average exponentialy global variance.
                // variance += momentum * (variance_b - variance)   "Risk of NaN if momentum equal 1.0"
                tmp_variance_average = *tmp_ptr_neuron_unit_it->ptr_variance_average += this->normalization_momentum_average * (tmp_variance - *tmp_ptr_neuron_unit_it->ptr_variance_average); // Exponential moving average.

                // r correction.
                // value = variance_b / variance
                tmp_gamma = tmp_variance / tmp_variance_average;
                // low = 1 / r_correction_max
                tmp_r_correction = 1_T / tmp_r_correction_maximum;
                // high = r_correction_max
                // r_correction = clip(value, low, high)
                *tmp_ptr_neuron_unit_it->ptr_d_correction = MyEA::Math::Clip<T_>(tmp_gamma, tmp_r_correction, tmp_r_correction_maximum);

                // d correction.
                // value = (mean_b - mean) / variance
                tmp_d_correction = (*tmp_ptr_neuron_unit_it->ptr_array_means - tmp_mean_average) / tmp_variance_average;
                // low = -d_correction_max
                // high = d_correction_max
                // d_correction = clip(value, low, high)
                *tmp_ptr_neuron_unit_it->ptr_d_correction = MyEA::Math::Clip<T_>(tmp_d_correction, -tmp_d_correction_maximum, tmp_d_correction_maximum);
            }
        }

        // Activation function: Loop through each sample.
        for(tmp_data_index = 0_zu; tmp_data_index != batch_size_received; ++tmp_data_index)
        {
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_layer_it_last_neuron; ++tmp_ptr_neuron_unit_it)
            {
                if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
                {
                    tmp_summation = tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units];
                
                    // Normalize.
                    // value_hat = (summation - mean_b) / variance_b * r_correction + d_correction
                    tmp_ptr_neuron_unit_it->ptr_array_values_hats[tmp_data_index * tmp_number_neuron_units] = tmp_summation = (tmp_summation - *tmp_ptr_neuron_unit_it->ptr_array_means) / *tmp_ptr_neuron_unit_it->ptr_array_variances * *tmp_ptr_neuron_unit_it->ptr_d_correction + *tmp_ptr_neuron_unit_it->ptr_d_correction;
                
                    // Scale and shift.
                    // value_normalize = scale * value_hat + shift
                    tmp_ptr_neuron_unit_it->ptr_array_values_normalizes[tmp_data_index * tmp_number_neuron_units] = tmp_summation = *tmp_ptr_neuron_unit_it->ptr_scale * tmp_summation + *tmp_ptr_neuron_unit_it->ptr_shift;
                
                    // Activation.
                    // value = AF(value_normalize)
                    Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_data_index * tmp_number_neuron_units],
                                           tmp_summation,
                                           *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
                }
            }
        }
    }
}
