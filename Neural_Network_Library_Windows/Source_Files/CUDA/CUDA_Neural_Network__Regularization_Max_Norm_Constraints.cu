#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Reduce.cuh>
#include <CUDA/CUDA_Multiply_1D.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

__global__ void kernel__CNeural_Network__Set__Regularization__Max_Norm_Constraints(T_ const regularization__max_norm_constraints_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Regularization__Max_Norm_Constraints(regularization__max_norm_constraints_received); }

__host__ __device__ bool CUDA_Neural_Network::Set__Regularization__Max_Norm_Constraints(T_ const regularization__max_norm_constraints_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CNeural_Network__Set__Regularization__Max_Norm_Constraints <<< 1u, 1u >>> (regularization__max_norm_constraints_received, this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#else
    if(this->regularization__max_norm_constraints != regularization__max_norm_constraints_received)
    {
        T_ const tmp_regularization__max_norm_constraints(this->regularization__max_norm_constraints);

        this->regularization__max_norm_constraints = regularization__max_norm_constraints_received;

        if(tmp_regularization__max_norm_constraints == 0_T && regularization__max_norm_constraints_received != 0_T)
        {
            if(this->Allocate__Neurons_Reduce_Norms() == false)
            {
                PRINT_FORMAT("%s: ERROR: Can not allocate regularization connections!" NEW_LINE, __FUNCTION__);
        
                return(false);
            }
        }
        else if(tmp_regularization__max_norm_constraints != 0_T && regularization__max_norm_constraints_received == 0_T)
        { this->Deallocate__Neurons_Reduce_Norms(); }
    }
#endif

    return(true);
}

__device__ void CUDA_Neural_Network::Update_Weight_Regularization__Max_Norm_Constraints(void)
{
    struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer - 1);
    struct CUDA_Layer const *tmp_ptr_layer_it(this->ptr_array_layers + 1);
        
    this->Update_Weight_Regularization__Max_Norm_Constraints__Neurons(tmp_ptr_layer_it, tmp_ptr_last_layer);
}

template<typename T>
__device__ inline void Vector__Max_Norm_Constraints_Reduce(bool &ref_synchronized_received,
                                                                                        size_t const number_connections_received,
                                                                                        size_t const increment_step_dim3_received,
                                                                                        T *const ptr_array_reduce_norms_received,
                                                                                        T *const ptr_array_parameters_received,
                                                                                        struct dim3 const *const ptr_array_dim3_grid_norms_received,
                                                                                        struct dim3 const *const ptr_array_dim3_block_norms_received)
{
    Reduce::Reduce_Square<T>(number_connections_received,
                                              increment_step_dim3_received,
                                              ptr_array_reduce_norms_received,
                                              ptr_array_parameters_received,
                                                ptr_array_dim3_grid_norms_received,
                                                ptr_array_dim3_block_norms_received);
    
    // Do we need to synchronise? Based on "Reduce_Square" Function.
    // => Synchronize if needed to see the norms reduced of the neuron.
    if(number_connections_received >= warpSize)
    {
        // We need a synchronisation here.
        ref_synchronized_received = false;
    }
}

template<typename T>
__device__ inline void Vector__Max_Norm_Constraints_Normalize(bool &ref_synchronized_received,
                                                                                            size_t const number_connections_received,
                                                                                            T const regularization__max_norm_constraints_received,
                                                                                            T *const ptr_array_reduce_norms_received,
                                                                                            T *const ptr_array_parameters_received,
                                                                                            struct dim3 const *const ptr_dim3_grid_connections_received,
                                                                                            struct dim3 const *const ptr_dim3_block_connections_received)
{
    T tmp_desired,
       tmp_desired_max_norm;
    
    *ptr_array_reduce_norms_received = sqrt(*ptr_array_reduce_norms_received);

    tmp_desired = MyEA::Math::Clip<T_>(*ptr_array_reduce_norms_received,
                                            T(0),
                                            regularization__max_norm_constraints_received);

    if(tmp_desired != *ptr_array_reduce_norms_received)
    {
        tmp_desired_max_norm = tmp_desired / *ptr_array_reduce_norms_received;
        
        Multiply::Multiply_X_Y_1D(ref_synchronized_received,
                                               number_connections_received,
                                               tmp_desired_max_norm,
                                               ptr_array_parameters_received,
                                               ptr_dim3_grid_connections_received,
                                               ptr_dim3_block_connections_received);
    }
}

template<typename T>
__global__ void kernel__Update_Weight_Regularization__Max_Norm_Constraints__Neurons(size_t const *const ptr_array_neuroyed_number_neurons_in_layer_received,
                                                                                                                                        T const regularization__max_norm_constraints_received,
                                                                                                                                        T **const ptr_array_2D_reduce_norms_received,
                                                                                                                                        T *const ptr_array_weigths_received,
                                                                                                                                        size_t const *const ptr_array_first_index_connection_received,
                                                                                                                                        size_t const *const ptr_array_neuroyed_number_connections_received,
                                                                                                                                        struct dim3 **const ptr_array_2D_dim3_grid_norms_received,
                                                                                                                                        struct dim3 **const ptr_array_2D_dim3_block_norms_received,
                                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                                        struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    // By default the synchronized state is set to true.
    bool tmp_synchronized(true);

    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
                                tmp_number_connections(ptr_array_neuroyed_number_connections_received[tmp_thread_global_index]);
    
    if(tmp_number_connections != 0u) // If is not a bias.
    {
        Vector__Max_Norm_Constraints_Reduce<T>(tmp_synchronized,
                                                                        tmp_number_connections - 1_zu, // Subtract bias.
                                                                        ptr_array_neuroyed_number_neurons_in_layer_received[tmp_thread_global_index],
                                                                        ptr_array_2D_reduce_norms_received[tmp_thread_global_index],
                                                                        ptr_array_weigths_received + ptr_array_first_index_connection_received[tmp_thread_global_index],
                                                                        ptr_array_2D_dim3_grid_norms_received[tmp_thread_global_index],
                                                                        ptr_array_2D_dim3_block_norms_received[tmp_thread_global_index]);
    }

    // Do we need to synchronise? Based on "CUDA__Device_Synchronise" Function.
    // => Synchronisation before using the reduce norms.
    CUDA__Device_Synchronise(tmp_synchronized, MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK);
    
    if(tmp_number_connections != 0u) // If is not a bias.
    {
        Vector__Max_Norm_Constraints_Normalize<T>(tmp_synchronized,
                                                                          tmp_number_connections - 1_zu, // Subtract bias.
                                                                          regularization__max_norm_constraints_received,
                                                                          ptr_array_2D_reduce_norms_received[tmp_thread_global_index],
                                                                          ptr_array_weigths_received + ptr_array_first_index_connection_received[tmp_thread_global_index],
                                                                          ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                                                          ptr_array_dim3_block_connections_received + tmp_thread_global_index);
    }
}
    
template<typename T>
__global__ void kernel__Update_Weight_Regularization__Max_Norm_Constraints__Neurons(size_t const size_received,
                                                                                                                                        size_t const *const ptr_array_neuroyed_number_neurons_in_layer_received,
                                                                                                                                        T const regularization__max_norm_constraints_received,
                                                                                                                                        T **const ptr_array_2D_reduce_norms_received,
                                                                                                                                        T *const ptr_array_weigths_received,
                                                                                                                                        size_t const *const ptr_array_first_index_connection_received,
                                                                                                                                        size_t const *const ptr_array_neuroyed_number_connections_received,
                                                                                                                                        struct dim3 **const ptr_array_2D_dim3_grid_norms_received,
                                                                                                                                        struct dim3 **const ptr_array_2D_dim3_block_norms_received,
                                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                                        struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    // By default the synchronized state is set to true.
    bool tmp_synchronized(true);

    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    size_t tmp_number_connections;
    
    if(tmp_thread_global_index < size_received)
    {
        tmp_number_connections = ptr_array_neuroyed_number_connections_received[tmp_thread_global_index];

        if(tmp_number_connections != 0u) // If is not a bias.
        {
            Vector__Max_Norm_Constraints_Reduce<T>(tmp_synchronized,
                                                                          tmp_number_connections - 1_zu, // Subtract bias.
                                                                          ptr_array_neuroyed_number_neurons_in_layer_received[tmp_thread_global_index],
                                                                          ptr_array_2D_reduce_norms_received[tmp_thread_global_index],
                                                                          ptr_array_weigths_received + ptr_array_first_index_connection_received[tmp_thread_global_index],
                                                                          ptr_array_2D_dim3_grid_norms_received[tmp_thread_global_index],
                                                                          ptr_array_2D_dim3_block_norms_received[tmp_thread_global_index]);
        }
    }

    // Do we need to synchronise? Based on "CUDA__Device_Synchronise" Function.
    // => Synchronisation before using the reduce norms.
    CUDA__Device_Synchronise(tmp_synchronized, MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK);
    
    if(tmp_thread_global_index < size_received && tmp_number_connections != 0u)
    {
        //PRINT_FORMAT("Neuron_unit[%u], nConnections(%u), Norm(%f)" NEW_LINE,
        //                        tmp_thread_global_index,
        //                        tmp_number_connections,
        //                        *(ptr_array_2D_reduce_norms_received[tmp_thread_global_index]));

        Vector__Max_Norm_Constraints_Normalize<T>(tmp_synchronized,
                                                                          tmp_number_connections - 1_zu, // Subtract bias.
                                                                          regularization__max_norm_constraints_received,
                                                                          ptr_array_2D_reduce_norms_received[tmp_thread_global_index],
                                                                          ptr_array_weigths_received + ptr_array_first_index_connection_received[tmp_thread_global_index],
                                                                          ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                                                          ptr_array_dim3_block_connections_received + tmp_thread_global_index);
    }
}
    
template<typename T>
__global__ void kernel_while__Update_Weight_Regularization__Max_Norm_Constraints__Neurons(size_t const size_received,
                                                                                                                                            size_t const *const ptr_array_neuroyed_number_neurons_in_layer_received,
                                                                                                                                            T const regularization__max_norm_constraints_received,
                                                                                                                                            T **const ptr_array_2D_reduce_norms_received,
                                                                                                                                            T *const ptr_array_weigths_received,
                                                                                                                                            size_t const *const ptr_array_first_index_connection_received,
                                                                                                                                            size_t const *const ptr_array_neuroyed_number_connections_received,
                                                                                                                                            struct dim3 **const ptr_array_2D_dim3_grid_norms_received,
                                                                                                                                            struct dim3 **const ptr_array_2D_dim3_block_norms_received,
                                                                                                                                            struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                                            struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    // By default the synchronized state is set to true.
    bool tmp_synchronized(true);

    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
                      tmp_number_connections;
    
    do
    {
        tmp_number_connections = ptr_array_neuroyed_number_connections_received[tmp_thread_global_index];
        
        if(tmp_number_connections != 0u) // If is not a bias.
        {
            Vector__Max_Norm_Constraints_Reduce<T>(tmp_synchronized,
                                                                          tmp_number_connections - 1_zu, // Subtract bias.
                                                                          ptr_array_neuroyed_number_neurons_in_layer_received[tmp_thread_global_index],
                                                                          ptr_array_2D_reduce_norms_received[tmp_thread_global_index],
                                                                          ptr_array_weigths_received + ptr_array_first_index_connection_received[tmp_thread_global_index],
                                                                          ptr_array_2D_dim3_grid_norms_received[tmp_thread_global_index],
                                                                          ptr_array_2D_dim3_block_norms_received[tmp_thread_global_index]);
        }

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
    
    // Reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Do we need to synchronise? Based on "CUDA__Device_Synchronise" Function.
    // => Synchronisation before using the reduce norms.
    CUDA__Device_Synchronise(tmp_synchronized, MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK);

    do
    {
        tmp_number_connections = ptr_array_neuroyed_number_connections_received[tmp_thread_global_index];
        
        if(tmp_number_connections != 0u) // If is not a bias.
        {
            Vector__Max_Norm_Constraints_Normalize<T>(tmp_synchronized,
                                                                              tmp_number_connections - 1_zu, // Subtract bias.
                                                                              regularization__max_norm_constraints_received,
                                                                              ptr_array_2D_reduce_norms_received[tmp_thread_global_index],
                                                                              ptr_array_weigths_received + ptr_array_first_index_connection_received[tmp_thread_global_index],
                                                                              ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                                                              ptr_array_dim3_block_connections_received + tmp_thread_global_index);
        }

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Update_Weight_Regularization__Max_Norm_Constraints__Neurons(struct CUDA_Layer const *const ptr_layer_it_received, struct CUDA_Layer const *const ptr_last_layer_received)
{
    // By default the synchronized state is set to true.
    bool tmp_synchronized(true);

    struct CUDA_Neuron const *const tmp_ptr_last_neuron_unit(ptr_last_layer_received->ptr_last_neuron_unit);
    struct CUDA_Neuron *tmp_ptr_neuron_unit_it(ptr_layer_it_received->ptr_array_neuron_units);

    size_t const tmp_number_neurons_received(static_cast<size_t>(tmp_ptr_last_neuron_unit - tmp_ptr_neuron_unit_it)),
                                *tmp_ptr_array_neuroyed_number_neurons_in_layer(this->ptr_array_neuroyed_number_neurons_in_layer + static_cast<size_t>(tmp_ptr_neuron_unit_it - this->ptr_array_layers->ptr_array_neuron_units));
    
    if(USE_PARALLEL && tmp_number_neurons_received >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        tmp_synchronized = false;
        
        LAUNCH_KERNEL_1D(Update_Weight_Regularization__Max_Norm_Constraints__Neurons<T_>,
                                          this->ptr_array_dim3_grid[6u],
                                          this->ptr_array_dim3_block[6u],
                                          0_zu,
                                          tmp_number_neurons_received,
                                          tmp_ptr_array_neuroyed_number_neurons_in_layer,
                                          this->regularization__max_norm_constraints,
                                          tmp_ptr_neuron_unit_it->ptr_array_reduce_norms,
                                          this->ptr_array_parameters,
                                          tmp_ptr_neuron_unit_it->ptr_first_forward_connection_index,
                                          tmp_ptr_neuron_unit_it->ptr_number_forward_connections,
                                          tmp_ptr_neuron_unit_it->ptr_array_2D_dim3_grid_reduce_norms,
                                          tmp_ptr_neuron_unit_it->ptr_array_2D_dim3_block_reduce_norms,
                                          tmp_ptr_neuron_unit_it->ptr_dim3_grid_connections,
                                          tmp_ptr_neuron_unit_it->ptr_dim3_block_connections)
    }
    else
    {
        // Loop through each neuron of the range received as arguments. by default the whole network with connection to it.
        for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                             ++tmp_ptr_array_neuroyed_number_neurons_in_layer)
        {
            if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections != 0u) // If is not a bias.
            {
                Vector__Max_Norm_Constraints_Reduce<T_>(tmp_synchronized,
                                                                                *tmp_ptr_neuron_unit_it->ptr_number_forward_connections - 1u, // Subtract bias.
                                                                                *tmp_ptr_array_neuroyed_number_neurons_in_layer,
                                                                                *tmp_ptr_neuron_unit_it->ptr_array_reduce_norms,
                                                                                this->ptr_array_parameters + *tmp_ptr_neuron_unit_it->ptr_first_forward_connection_index,
                                                                                *tmp_ptr_neuron_unit_it->ptr_array_2D_dim3_grid_reduce_norms,
                                                                                *tmp_ptr_neuron_unit_it->ptr_array_2D_dim3_block_reduce_norms);
            }
        }
        
        // Synchronize if needed to see the reduced norms of the network.
        CUDA__Device_Synchronise(tmp_synchronized, MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);

        // Loop through each neuron of the range received as arguments. by default the whole network with connection to it.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_received->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
        {
            if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections != 0u) // If is not a bias.
            {
                //PRINT_FORMAT("Neuron_unit[%u], nConnections(%u), Norm(%f)" NEW_LINE,
                //                        static_cast<size_t>(tmp_ptr_neuron_unit_it - ptr_layer_it_received->ptr_array_neuron_units),
                //                        *tmp_ptr_neuron_unit_it->ptr_number_forward_connections,
                //                        *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_norms));

                Vector__Max_Norm_Constraints_Normalize<T_>(tmp_synchronized,
                                                                                    *tmp_ptr_neuron_unit_it->ptr_number_forward_connections - 1u, // Subtract bias.
                                                                                    this->regularization__max_norm_constraints,
                                                                                    *tmp_ptr_neuron_unit_it->ptr_array_reduce_norms,
                                                                                    this->ptr_array_parameters + *tmp_ptr_neuron_unit_it->ptr_first_forward_connection_index,
                                                                                    tmp_ptr_neuron_unit_it->ptr_dim3_grid_connections,
                                                                                    tmp_ptr_neuron_unit_it->ptr_dim3_block_connections);
            }
        }
    }
    
    // Synchronize if needed to see the weights norms of the network.
    CUDA__Device_Synchronise(tmp_synchronized, MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);
}

__host__ __device__ T_ CUDA_Neural_Network::Get__Regularization__Max_Norm_Constraints(void) const { return(this->regularization__max_norm_constraints); }
