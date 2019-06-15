#include <Tools/CUDA_Configuration.cuh>
#include <Tools/CUDA_Zero_1D.cuh>
#include <CUDA/CUDA_Flag_1D.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

#include <curand_kernel.h>

// Public function
__device__ bool CUDA_Neural_Network::Set__Probability_Retained_Unit(size_t const index_layer_received,
                                                                                                T_ const retention_probability_received,
                                                                                                bool const scale_weights_received)
{
    if(index_layer_received >= this->total_layers)
    {
        PRINT_FORMAT("%s: ERROR: Layer received (%d) overflow the number of layers (%d) in the neural network." NEW_LINE,
                                __FUNCTION__,
                                index_layer_received,
                                this->total_layers);

        return(false);
    }
    else if(this->ptr_array_layers == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: The array of layers is a nullptr." NEW_LINE, __FUNCTION__);

        return(false);
    }
        
    return(this->Set__Probability_Retained_Unit(this->ptr_array_layers + index_layer_received,
                                                                    retention_probability_received,
                                                                    scale_weights_received));
}

__device__ void CUDA_Neural_Network::Scale_Weight__Dropout(T_ const scale_factor_received, struct CUDA_Layer const *const ptr_layer_it_received)
{
    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED: this->Scale_Weight__FC__Forward__Dropout(scale_factor_received, ptr_layer_it_received); break;
    }
}

__device__ void CUDA_Neural_Network::Scale_Weight__FC__Forward__Dropout(T_ const scale_factor_received, struct CUDA_Layer const *const ptr_layer_it_received)
{
    struct CUDA_Neuron const *const tmp_ptr_neuron_unit_it(ptr_layer_it_received->ptr_array_neuron_units);
    
    T_ *tmp_ptr_array_parameters(this->ptr_array_parameters + *tmp_ptr_neuron_unit_it->ptr_first_forward_connection_index);
    T_ const *const tmp_ptr_array_parameters_end(tmp_ptr_array_parameters + *ptr_layer_it_received->ptr_number_neurons * *tmp_ptr_neuron_unit_it->ptr_number_forward_connections);
    
    for(; tmp_ptr_array_parameters != tmp_ptr_array_parameters_end; ++tmp_ptr_array_parameters)
    { *tmp_ptr_array_parameters *= scale_factor_received; }
}

// Private function.
__device__ bool CUDA_Neural_Network::Set__Probability_Retained_Unit(struct CUDA_Layer *ptr_layer_received,
                                                                                                        T_ const retention_probability_received,
                                                                                                        bool const scale_weights_received)
{
    struct CUDA_Layer const *tmp_ptr_last_layer;
    struct CUDA_Layer *tmp_ptr_layer_it;
    
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Layer received is a nullptr." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else if(retention_probability_received < 0.0f)
    {
        PRINT_FORMAT("%s: ERROR: Probability for retained a unit (%f) in the layer, underflow the requirement minimum of 0.0." NEW_LINE,
                        __FUNCTION__,
                        retention_probability_received);

        return(false);
    }
    else if(retention_probability_received > 1.0f)
    {
        PRINT_FORMAT("%s: ERROR: Probability for retained a unit (%f) in the layer, overflow the requirement maximum of 1.0." NEW_LINE,
                        __FUNCTION__,
                        retention_probability_received);

        return(false);
    }
        
    if(ptr_layer_received->dropout_values[0u] != retention_probability_received)
    {
        if(scale_weights_received && ptr_layer_received != this->ptr_array_layers) { this->Scale_Weight__Dropout(ptr_layer_received->dropout_values[0u] / retention_probability_received, ptr_layer_received); }

        ptr_layer_received->dropout_values[0u] = retention_probability_received;

        if(retention_probability_received != 1.0f)
        {
            this->use_Dropout = true;

            if(this->Allocate__Neuron__Mask_Dropout_Bernoulli() == false)
            {
                PRINT_FORMAT("%s: ERROR: Can not allocate neurons mask dropout!" NEW_LINE, __FUNCTION__);
                
                return(false);
            }
        }
        else // Check if we use dropout
        {
            bool tmp_use_Dropout(false);
            
            // Loop through each layer to do a check if a layer use dropout.
            for(tmp_ptr_last_layer = this->ptr_last_layer,
                tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
            {
                if(tmp_ptr_layer_it->dropout_values[0u] != 1_T)
                {
                    tmp_use_Dropout = true;

                    break;
                }
            }
            
            this->use_Dropout = tmp_use_Dropout;
            // |END| Loop through each layer to do a check if a layer use dropout. |END|

            if(tmp_use_Dropout == false)
            {
                this->Deallocate__Neuron__Mask_Dropout_Bernoulli();
            }
        }
    }

    return(true);
}

__device__ void CUDA_Neural_Network::Reset__Parameter__AF_Units__Mask_Dropout(bool *ptr_array_neuron_units_mask_dropout_received)
{
    struct CUDA_Neuron *tmp_ptr_neuron_unit_it(this->ptr_array_layers->ptr_array_neuron_units);
    struct CUDA_Neuron const *const tmp_ptr_last_neuron_unit(tmp_ptr_neuron_unit_it + this->total_neuron_units);

    this->ptr_array_af_units_mask_dropout_bernoulli = ptr_array_neuron_units_mask_dropout_received;

    for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                         ++ptr_array_neuron_units_mask_dropout_received)
    { tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli = ptr_array_neuron_units_mask_dropout_received; }
}

//__managed__ size_t tmp_count_dropped = 0u;
//__managed__ size_t tmp_count_total = 0u;

template<typename T>
__global__ void kernel__Dropout_Bernoulli__Neurons(bool *const ptr_array_mask_dropout_received,
                                                                T const probability_retained_unit_received,
                                                                struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(cuRAND_Bernoulli(probability_retained_unit_received, curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)))
    { ptr_array_mask_dropout_received[tmp_thread_global_index] = true; }
    else // Dropout neuron
    { ptr_array_mask_dropout_received[tmp_thread_global_index] = false; }
}
    
template<typename T>
__global__ void kernel__Dropout_Bernoulli__Neurons(size_t const size_received,
                                                                bool *const ptr_array_mask_dropout_received,
                                                                T const probability_retained_unit_received,
                                                                struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    float const tmp_curand_uniform(curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x));

    if(tmp_thread_global_index < size_received)
    {
        if(cuRAND_Bernoulli(probability_retained_unit_received, tmp_curand_uniform))
        { ptr_array_mask_dropout_received[tmp_thread_global_index] = true; }
        else // Dropout neuron
        { ptr_array_mask_dropout_received[tmp_thread_global_index] = false; }
    }
}

template<typename T>
__global__ void kernel_while__Dropout_Bernoulli__Neurons(size_t const size_received,
                                                                        bool *const ptr_array_mask_dropout_received,
                                                                        T const probability_retained_unit_received,
                                                                        struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        if(cuRAND_Bernoulli(probability_retained_unit_received, curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)))
        { ptr_array_mask_dropout_received[tmp_thread_global_index] = true; }
        else // Dropout neuron
        { ptr_array_mask_dropout_received[tmp_thread_global_index] = false; }
            
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Dropout(void)
{
    bool const tmp_use_parameters_dropout(false);

    /* sync code:
        0: Synchronized,
        1: Critical kernel launch,
        2: Optinal kernel launch. */
    size_t tmp_sync_code(0u),
                      tmp_neuron_index;

    struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct CUDA_Layer *tmp_ptr_previous_layer_it(this->ptr_array_layers),
                                            *tmp_ptr_layer_it(tmp_ptr_previous_layer_it);

    struct CUDA_Neuron const *const tmp_ptr_neuron_unit_it(tmp_ptr_layer_it->ptr_array_neuron_units);
    
    bool *tmp_ptr_array_mask_dropout(tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli);
    bool const *const tmp_ptr_array_mask_dropout_end(tmp_ptr_array_mask_dropout + *tmp_ptr_layer_it->ptr_number_neurons - 1u); // Subtract bias.

    // Input layer.
    if(USE_PARALLEL && *tmp_ptr_layer_it->ptr_number_neurons - 1u >= warpSize)
    {
        // Critical synchronization required. To see the previous neurons flag.
        tmp_sync_code = 1u;
        
        LAUNCH_KERNEL_POINTER_1D(Dropout_Bernoulli__Neurons<T_>,
                                                          tmp_ptr_layer_it->ptr_dim3_grid_neurons_cuRAND,
                                                          tmp_ptr_layer_it->ptr_dim3_block_neurons_cuRAND,
                                                          0_zu,
                                                          *tmp_ptr_layer_it->ptr_number_neurons - 1u, // Subtract bias.
                                                          tmp_ptr_array_mask_dropout,
                                                          tmp_ptr_layer_it->dropout_values[0u],
                                                          this->ptr_array_cuRAND_State_MTGP32_neuroyed)
    }
    else
    {
        for(tmp_neuron_index = 0_zu; tmp_ptr_array_mask_dropout != tmp_ptr_array_mask_dropout_end; ++tmp_ptr_array_mask_dropout,
                                                                                                                                                   ++tmp_neuron_index)
        {
            if(cuRAND_Bernoulli(tmp_ptr_layer_it->dropout_values[0u], curand_uniform(this->ptr_array_cuRAND_State_MTGP32_neuroyed)))
            { *tmp_ptr_array_mask_dropout = true; }
            else // Dropout neuron
            { *tmp_ptr_array_mask_dropout = false; }
        }
    }
    // |END| Input layer. |END|
    
    if(this->use_Batch_Renormalization && tmp_use_parameters_dropout)
    {
        for(++tmp_ptr_layer_it; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                                                    ++tmp_ptr_previous_layer_it)
        {
            if(tmp_ptr_layer_it->use_Batch_Renormalization)
            {
                switch(tmp_ptr_previous_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                        this->Dropout__FC_to__Batch_Normalization(tmp_sync_code,
                                                                                                tmp_ptr_layer_it,
                                                                                                tmp_ptr_previous_layer_it);
                            break;
                }
            }
            else
            {
                switch(tmp_ptr_previous_layer_it->type_layer)
                {
                    case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                        this->Dropout__FC_to(tmp_sync_code,
                                                              tmp_use_parameters_dropout,
                                                              tmp_ptr_layer_it,
                                                              tmp_ptr_previous_layer_it);
                            break;
                }
            }
        }
    }
    else
    {
        for(++tmp_ptr_layer_it; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                                                    ++tmp_ptr_previous_layer_it)
        {
            switch(tmp_ptr_previous_layer_it->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                    this->Dropout__FC_to(tmp_sync_code,
                                                          tmp_use_parameters_dropout,
                                                          tmp_ptr_layer_it,
                                                          tmp_ptr_previous_layer_it);
                        break;
            }
        }
    }

    // If the state of the synchronized is not at zero. We synchronize.
    if(tmp_sync_code != 0u) { CUDA__Check_Error(); }
}

__device__ void CUDA_Neural_Network::Dropout__FC_to(size_t &ref_sync_code_received,
                                                                                bool const use_parameters_dropout_received,
                                                                                struct CUDA_Layer *const ptr_layer_it_received,
                                                                                struct CUDA_Layer const *const ptr_previous_layer_it_received)
{
    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            this->Dropout_Bernoulli__FC_to_FC(ref_sync_code_received,
                                                            use_parameters_dropout_received,
                                                            ptr_layer_it_received,
                                                            ptr_previous_layer_it_received);
                break;
    }
}

__device__ void CUDA_Neural_Network::Dropout__FC_to__Batch_Normalization(size_t &ref_sync_code_received,
                                                                                                                 struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                 struct CUDA_Layer const *const ptr_previous_layer_it_received)
{
    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
            this->Dropout_Bernoulli__FC_to_FC__Batch_Renormalization(ref_sync_code_received,
                                                                                              ptr_layer_it_received,
                                                                                              ptr_previous_layer_it_received);
                break;
    }
}

template<typename T>
__global__ void kernel__Dropout_Bernoulli__Neurons(size_t const number_connections_received,
                                                                bool const *const ptr_array_previous_layer_mask_dropout_received,
                                                                bool *const ptr_array_mask_dropout_received,
                                                                T *const ptr_array_mask_dropout_parameters_received,
                                                                T const probability_retained_unit_received,
                                                                struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                                                struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(cuRAND_Bernoulli(probability_retained_unit_received, curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)))
    {
        ptr_array_mask_dropout_received[tmp_thread_global_index] = true;
            
        Flag_1D<T>(number_connections_received,
                            ptr_array_previous_layer_mask_dropout_received,
                            ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                            ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                            ptr_array_dim3_block_connections_received + tmp_thread_global_index);

        ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(1); // Bias.
    }
    else // Dropout neuron
    {
        ptr_array_mask_dropout_received[tmp_thread_global_index] = false;
            
        Zero_1D<T>(number_connections_received,
                            ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                            ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                            ptr_array_dim3_block_connections_received + tmp_thread_global_index);

        ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(0); // Bias.
    }
}
    
template<typename T>
__global__ void kernel__Dropout_Bernoulli__Neurons(size_t const size_received,
                                                                size_t const number_connections_received,
                                                                bool const *const ptr_array_previous_layer_mask_dropout_received,
                                                                bool *const ptr_array_mask_dropout_received,
                                                                T *const ptr_array_mask_dropout_parameters_received,
                                                                T const probability_retained_unit_received,
                                                                struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                                                struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    float const tmp_curand_uniform(curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x));

    if(tmp_thread_global_index < size_received)
    {
        if(cuRAND_Bernoulli(probability_retained_unit_received, tmp_curand_uniform))
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = true;
            
            Flag_1D<T>(number_connections_received,
                            ptr_array_previous_layer_mask_dropout_received,
                            ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                            ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                            ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(1); // Bias.
        }
        else // Dropout neuron
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = false;
            
            Zero_1D<T>(number_connections_received,
                            ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                            ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                            ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(0); // Bias.
        }
    }
}

template<typename T>
__global__ void kernel_while__Dropout_Bernoulli__Neurons(size_t const size_received,
                                                                        size_t const number_connections_received,
                                                                        bool const *const ptr_array_previous_layer_mask_dropout_received,
                                                                        bool *const ptr_array_mask_dropout_received,
                                                                        T *const ptr_array_mask_dropout_parameters_received,
                                                                        T const probability_retained_unit_received,
                                                                        struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                                                        struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                        struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        if(cuRAND_Bernoulli(probability_retained_unit_received, curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)))
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = true;
            
            Flag_1D<T>(number_connections_received,
                              ptr_array_previous_layer_mask_dropout_received,
                              ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                              ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                              ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(1); // Bias.
        }
        else // Dropout neuron
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = false;
            
            Zero_1D<T>(number_connections_received,
                              ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                              ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                              ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(0); // Bias.
        }
            
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Dropout_Bernoulli__FC_to_FC(size_t &ref_sync_code_received,
                                                                                           bool const use_parameters_dropout_received,
                                                                                           struct CUDA_Layer *const ptr_layer_it_received,
                                                                                           struct CUDA_Layer const *const ptr_previous_layer_it_received)
{
    struct CUDA_Neuron const *const tmp_ptr_neuron_unit_it(ptr_layer_it_received->ptr_array_neuron_units);

    bool *tmp_ptr_array_mask_dropout(tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli);
    bool const *const tmp_ptr_array_mask_dropout_end(tmp_ptr_array_mask_dropout + *ptr_layer_it_received->ptr_number_neurons - 1u), // Subtract bias.
                    *tmp_ptr_array_previous_layer_mask_dropout(ptr_previous_layer_it_received->ptr_array_neuron_units->ptr_mask_dropout_bernoulli);

    size_t const tmp_number_connections(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections - 1u); // Subtract bias.
    size_t tmp_neuron_index;

    T_ *tmp_ptr_array_mask_dropout_parameters(this->ptr_array_mask_dropout_parameters + *tmp_ptr_neuron_unit_it->ptr_first_forward_connection_index);
    
    struct dim3 const *tmp_ptr_array_dim3_grid_connections(tmp_ptr_neuron_unit_it->ptr_dim3_grid_connections),
                              *tmp_ptr_array_dim3_block_connections(tmp_ptr_neuron_unit_it->ptr_dim3_block_connections);

    if(USE_PARALLEL && *ptr_layer_it_received->ptr_number_neurons - 1u >= warpSize)
    {
        // Critical synchronization required. To see the previous neurons flag.
        ref_sync_code_received = 1u;

        if(use_parameters_dropout_received)
        {
            LAUNCH_KERNEL_POINTER_1D(Dropout_Bernoulli__Neurons<T_>,
                                                              ptr_layer_it_received->ptr_dim3_grid_neurons_cuRAND,
                                                              ptr_layer_it_received->ptr_dim3_block_neurons_cuRAND,
                                                              0_zu,
                                                              *ptr_layer_it_received->ptr_number_neurons - 1u, // Subtract bias.
                                                              tmp_number_connections,
                                                              tmp_ptr_array_previous_layer_mask_dropout,
                                                              tmp_ptr_array_mask_dropout,
                                                              tmp_ptr_array_mask_dropout_parameters,
                                                              ptr_layer_it_received->dropout_values[0u],
                                                              this->ptr_array_cuRAND_State_MTGP32_neuroyed,
                                                              tmp_ptr_array_dim3_grid_connections,
                                                              tmp_ptr_array_dim3_block_connections)
        }
        else
        {
            LAUNCH_KERNEL_POINTER_1D(Dropout_Bernoulli__Neurons<T_>,
                                                              ptr_layer_it_received->ptr_dim3_grid_neurons_cuRAND,
                                                              ptr_layer_it_received->ptr_dim3_block_neurons_cuRAND,
                                                              0_zu,
                                                              *ptr_layer_it_received->ptr_number_neurons - 1u, // Subtract bias.
                                                              tmp_ptr_array_mask_dropout,
                                                              ptr_layer_it_received->dropout_values[0u],
                                                              this->ptr_array_cuRAND_State_MTGP32_neuroyed)
        }
    }
    else
    {
        // Need a synchronization to see the previous neurons flag.
        if(ref_sync_code_received == 1u)
        {
            // Set the state at zero (Synchronized).
            ref_sync_code_received = 0u;

            CUDA__Check_Error();
        }

        for(tmp_neuron_index = 0_zu; tmp_ptr_array_mask_dropout != tmp_ptr_array_mask_dropout_end; ++tmp_ptr_array_mask_dropout,
                                                                                                                                                   ++tmp_neuron_index,
                                                                                                                                                   ++tmp_ptr_array_dim3_grid_connections,
                                                                                                                                                   ++tmp_ptr_array_dim3_block_connections,
                                                                                                                                                   tmp_ptr_array_mask_dropout_parameters += tmp_number_connections + 1u) // Add bias.
        {
            if(cuRAND_Bernoulli(ptr_layer_it_received->dropout_values[0u], curand_uniform(this->ptr_array_cuRAND_State_MTGP32_neuroyed)))
            {
                *tmp_ptr_array_mask_dropout = true;
                
                if(use_parameters_dropout_received)
                {
                    Flag_1D<T_>(tmp_number_connections,
                                        tmp_ptr_array_previous_layer_mask_dropout,
                                        tmp_ptr_array_mask_dropout_parameters,
                                        tmp_ptr_array_dim3_grid_connections,
                                        tmp_ptr_array_dim3_block_connections);

                    tmp_ptr_array_mask_dropout_parameters[tmp_number_connections] = 1_T; // Bias.
                }
            }
            else // Dropout neuron
            {
                *tmp_ptr_array_mask_dropout = false;
                    
                if(use_parameters_dropout_received)
                {
                    Zero_1D<T_>(tmp_number_connections,
                                        tmp_ptr_array_mask_dropout_parameters,
                                        tmp_ptr_array_dim3_grid_connections,
                                        tmp_ptr_array_dim3_block_connections);

                    tmp_ptr_array_mask_dropout_parameters[tmp_number_connections] = 0_T; // Bias.
                }
            }
        }
        
        // If number of connections is bigger or equal to 32. We need a synchronization to see the connections flag.
        if(use_parameters_dropout_received && tmp_number_connections >= warpSize) { ref_sync_code_received = 2u; }
    }

    //PRINT_FORMAT("tmp_count_dropped: %u / %u, %f%%" NEW_LINE,
    //                        tmp_count_dropped,
    //                        tmp_count_total,
    //                        static_cast<float>(tmp_count_dropped) / static_cast<float>(tmp_count_total) * 100.0f);
}

template<typename T>
__global__ void kernel__Dropout_Bernoulli__Neurons__Batch_Renormalization(size_t const number_connections_received,
                                                                                                                    bool const *const ptr_array_previous_layer_mask_dropout_received,
                                                                                                                    bool *const ptr_array_mask_dropout_received,
                                                                                                                    T const *const ptr_array_parameters_received,
                                                                                                                    T const *const ptr_array_parameters_scale_received,
                                                                                                                    T const *const ptr_array_parameters_shift_received,
                                                                                                                    T *const ptr_array_original_mask_dropout_parameters_received,
                                                                                                                    T *const ptr_array_mask_dropout_parameters_received,
                                                                                                                    T const probability_retained_unit_received,
                                                                                                                    struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                                                                                                    struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                    struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(cuRAND_Bernoulli(probability_retained_unit_received, curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)))
    {
        ptr_array_mask_dropout_received[tmp_thread_global_index] = true;
            
        Flag_1D<T>(number_connections_received,
                          ptr_array_previous_layer_mask_dropout_received,
                          ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                          ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                          ptr_array_dim3_block_connections_received + tmp_thread_global_index);

        ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(1); // Bias.
            
        // TODO: Optimize with a real index.
        ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_scale_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(1); // Batch normalization scale.
        ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_shift_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(1); // Batch normalization shift.
    }
    else // Dropout neuron
    {
        ptr_array_mask_dropout_received[tmp_thread_global_index] = false;
            
        Zero_1D<T>(number_connections_received,
                          ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                          ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                          ptr_array_dim3_block_connections_received + tmp_thread_global_index);

        ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(0); // Bias.
            
        // TODO: Optimize with a real index.
        ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_scale_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(0); // Batch normalization scale.
        ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_shift_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(0); // Batch normalization shift.
    }
}
    
template<typename T>
__global__ void kernel__Dropout_Bernoulli__Neurons__Batch_Renormalization(size_t const size_received,
                                                                                                                    size_t const number_connections_received,
                                                                                                                    bool const *const ptr_array_previous_layer_mask_dropout_received,
                                                                                                                    bool *const ptr_array_mask_dropout_received,
                                                                                                                    T *const ptr_array_parameters_received,
                                                                                                                    T *const ptr_array_parameters_scale_received,
                                                                                                                    T *const ptr_array_parameters_shift_received,
                                                                                                                    T *const ptr_array_original_mask_dropout_parameters_received,
                                                                                                                    T *const ptr_array_mask_dropout_parameters_received,
                                                                                                                    T const probability_retained_unit_received,
                                                                                                                    struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                                                                                                    struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                    struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    float const tmp_curand_uniform(curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x));

    if(tmp_thread_global_index < size_received)
    {
        if(cuRAND_Bernoulli(probability_retained_unit_received, tmp_curand_uniform))
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = true;
            
            Flag_1D<T>(number_connections_received,
                              ptr_array_previous_layer_mask_dropout_received,
                              ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                              ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                              ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(1); // Bias.
            
            // TODO: Optimize with a real index.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_scale_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(1); // Batch normalization scale.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_shift_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(1); // Batch normalization shift.
        }
        else // Dropout neuron
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = false;
            
            Zero_1D<T>(number_connections_received,
                              ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                              ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                              ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(0); // Bias.
            
            // TODO: Optimize with a real index.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_scale_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(0); // Batch normalization scale.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_shift_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(0); // Batch normalization shift.
        }
    }
}

template<typename T>
__global__ void kernel_while__Dropout_Bernoulli__Neurons__Batch_Renormalization(size_t const size_received,
                                                                                                                            size_t const number_connections_received,
                                                                                                                            bool const *const ptr_array_previous_layer_mask_dropout_received,
                                                                                                                            bool *const ptr_array_mask_dropout_received,
                                                                                                                            T *const ptr_array_parameters_received,
                                                                                                                            T *const ptr_array_parameters_scale_received,
                                                                                                                            T *const ptr_array_parameters_shift_received,
                                                                                                                            T *const ptr_array_original_mask_dropout_parameters_received,
                                                                                                                            T *const ptr_array_mask_dropout_parameters_received,
                                                                                                                            T const probability_retained_unit_received,
                                                                                                                            struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                                                                                                            struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                            struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        if(cuRAND_Bernoulli(probability_retained_unit_received, curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)))
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = true;
            
            Flag_1D<T>(number_connections_received,
                               ptr_array_previous_layer_mask_dropout_received,
                               ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                               ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                               ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(1); // Bias.
            
            // TODO: Optimize with a real index.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_scale_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(1); // Batch normalization scale.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_shift_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(1); // Batch normalization shift.
        }
        else // Dropout neuron
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = false;
            
            Zero_1D<T>(number_connections_received,
                               ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                               ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                               ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(0); // Bias.
            
            // TODO: Optimize with a real index.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_scale_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(0); // Batch normalization scale.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_shift_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(0); // Batch normalization shift.
        }
            
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Dropout_Bernoulli__FC_to_FC__Batch_Renormalization(size_t &ref_sync_code_received,
                                                                                                                            struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                            struct CUDA_Layer const *const ptr_previous_layer_it_received)
{
    struct CUDA_Neuron const *const tmp_ptr_neuron_unit_it(ptr_layer_it_received->ptr_array_neuron_units);

    bool *tmp_ptr_array_mask_dropout(tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli);
    bool const *const tmp_ptr_array_mask_dropout_end(tmp_ptr_array_mask_dropout + *ptr_layer_it_received->ptr_number_neurons - 1u), // Subtract bias.
                    *tmp_ptr_array_previous_layer_mask_dropout(ptr_previous_layer_it_received->ptr_array_neuron_units->ptr_mask_dropout_bernoulli);

    size_t const tmp_number_connections(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections - 1u); // Subtract bias.
    size_t tmp_neuron_index;

    T_ *tmp_ptr_array_mask_dropout_parameters(this->ptr_array_mask_dropout_parameters + *tmp_ptr_neuron_unit_it->ptr_first_forward_connection_index),
         *tmp_ptr_array_scales(tmp_ptr_neuron_unit_it->ptr_scale),
         *tmp_ptr_array_shifts(tmp_ptr_neuron_unit_it->ptr_shift);
    
    struct dim3 const *tmp_ptr_array_dim3_grid_connections(tmp_ptr_neuron_unit_it->ptr_dim3_grid_connections),
                              *tmp_ptr_array_dim3_block_connections(tmp_ptr_neuron_unit_it->ptr_dim3_block_connections);

    if(USE_PARALLEL && *ptr_layer_it_received->ptr_number_neurons - 1u >= warpSize)
    {
        // Critical synchronization required. To see the previous neurons flag.
        ref_sync_code_received = 1u;
        
        LAUNCH_KERNEL_POINTER_1D(Dropout_Bernoulli__Neurons__Batch_Renormalization<T_>,
                                                            ptr_layer_it_received->ptr_dim3_grid_neurons_cuRAND,
                                                            ptr_layer_it_received->ptr_dim3_block_neurons_cuRAND,
                                                            0_zu,
                                                            *ptr_layer_it_received->ptr_number_neurons - 1u, // Subtract bias.
                                                            tmp_number_connections,
                                                            tmp_ptr_array_previous_layer_mask_dropout,
                                                            tmp_ptr_array_mask_dropout,
                                                            this->ptr_array_parameters,
                                                            tmp_ptr_array_scales,
                                                            tmp_ptr_array_shifts,
                                                            this->ptr_array_mask_dropout_parameters,
                                                            tmp_ptr_array_mask_dropout_parameters,
                                                            ptr_layer_it_received->dropout_values[0u],
                                                            this->ptr_array_cuRAND_State_MTGP32_neuroyed,
                                                            tmp_ptr_array_dim3_grid_connections,
                                                            tmp_ptr_array_dim3_block_connections)
    }
    else
    {
        // Need a synchronization to see the previous neurons flag.
        if(ref_sync_code_received == 1u)
        {
            // Set the state at zero (Synchronized).
            ref_sync_code_received = 0u;

            CUDA__Check_Error();
        }
        
        for(tmp_neuron_index = 0_zu; tmp_ptr_array_mask_dropout != tmp_ptr_array_mask_dropout_end; ++tmp_ptr_array_mask_dropout,
                                                                                                                                                   ++tmp_neuron_index,
                                                                                                                                                   ++tmp_ptr_array_scales,
                                                                                                                                                   ++tmp_ptr_array_shifts,
                                                                                                                                                   ++tmp_ptr_array_dim3_grid_connections,
                                                                                                                                                   ++tmp_ptr_array_dim3_block_connections,
                                                                                                                                                   tmp_ptr_array_mask_dropout_parameters += tmp_number_connections + 1u) // Add bias.
        {
            if(cuRAND_Bernoulli(ptr_layer_it_received->dropout_values[0u], curand_uniform(this->ptr_array_cuRAND_State_MTGP32_neuroyed)))
            {
                *tmp_ptr_array_mask_dropout = true;
                
                Flag_1D<T_>(tmp_number_connections,
                                    tmp_ptr_array_previous_layer_mask_dropout,
                                    tmp_ptr_array_mask_dropout_parameters,
                                    tmp_ptr_array_dim3_grid_connections,
                                    tmp_ptr_array_dim3_block_connections);

                tmp_ptr_array_mask_dropout_parameters[tmp_number_connections] = 1_T; // Bias.

                // TODO: Optimize with a real index.
                this->ptr_array_mask_dropout_parameters[static_cast<size_t>(tmp_ptr_array_scales - this->ptr_array_parameters)] = 1_T; // Batch normalization scale.
                this->ptr_array_mask_dropout_parameters[static_cast<size_t>(tmp_ptr_array_shifts - this->ptr_array_parameters)] = 1_T; // Batch normalization shift.
            }
            else // Dropout neuron
            {
                *tmp_ptr_array_mask_dropout = false;
                
                Zero_1D<T_>(tmp_number_connections,
                                    tmp_ptr_array_mask_dropout_parameters,
                                    tmp_ptr_array_dim3_grid_connections,
                                    tmp_ptr_array_dim3_block_connections);

                tmp_ptr_array_mask_dropout_parameters[tmp_number_connections] = 0_T; // Bias.

                // TODO: Optimize with a real index.
                this->ptr_array_mask_dropout_parameters[static_cast<size_t>(tmp_ptr_array_scales - this->ptr_array_parameters)] = 0_T; // Batch normalization scale.
                this->ptr_array_mask_dropout_parameters[static_cast<size_t>(tmp_ptr_array_shifts - this->ptr_array_parameters)] = 0_T; // Batch normalization shift.
            }
        }

        // If number of connections is bigger or equal to 32. We need a synchronization to see the connections flag.
        if(tmp_number_connections >= warpSize) { ref_sync_code_received = 2u; }
    }

    //PRINT_FORMAT("tmp_count_dropped: %u / %u, %f%%" NEW_LINE,
    //                        tmp_count_dropped,
    //                        tmp_count_total,
    //                        static_cast<float>(tmp_count_dropped) / static_cast<float>(tmp_count_total) * 100.0f);
}
