#include <Tools/CUDA_Configuration.cuh>
#include <Tools/CUDA_Memory_Copy_1D.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

#include <Neural_Network/Neural_Network.hpp>

#include <chrono>

template<typename T>
__global__ void kernel__CNeural_Network__Copy_Neurons(size_t *const ptr_array_neuron_units_first_connection_index_destination_received,
                                                                                      size_t *const ptr_array_neuron_units_last_connection_index_destination_received,
                                                                                      size_t *const ptr_array_neuron_units_number_connections_destination_received,
                                                                                      size_t const *const ptr_array_neuron_units_first_connection_index_source_received,
                                                                                      size_t const *const ptr_array_neuron_units_last_connection_index_source_received,
                                                                                      T *const ptr_array_neuron_units_steepness_destination_received,
                                                                                      T const *const ptr_array_neuron_units_steepness_source_received,
                                                                                      enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS *const ptr_array_neuron_units_activation_function_destination_received,
                                                                                      enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_neuron_units_activation_function_source_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    size_t tmp_number_connections;

    ptr_array_neuron_units_last_connection_index_destination_received[tmp_thread_global_index] = tmp_number_connections = ptr_array_neuron_units_last_connection_index_source_received[tmp_thread_global_index];
        
    tmp_number_connections -= ptr_array_neuron_units_first_connection_index_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_first_connection_index_source_received[tmp_thread_global_index];
        
    ptr_array_neuron_units_number_connections_destination_received[tmp_thread_global_index] = tmp_number_connections;

    ptr_array_neuron_units_steepness_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_steepness_source_received[tmp_thread_global_index];

    ptr_array_neuron_units_activation_function_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_activation_function_source_received[tmp_thread_global_index];
}

template<typename T>
__global__ void kernel__CNeural_Network__Copy_Neurons(size_t const size_received,
                                                                                      size_t *const ptr_array_neuron_units_first_connection_index_destination_received,
                                                                                      size_t *const ptr_array_neuron_units_last_connection_index_destination_received,
                                                                                      size_t *const ptr_array_neuron_units_number_connections_destination_received,
                                                                                      size_t const *const ptr_array_neuron_units_first_connection_index_source_received,
                                                                                      size_t const *const ptr_array_neuron_units_last_connection_index_source_received,
                                                                                      T *const ptr_array_neuron_units_steepness_destination_received,
                                                                                      T const *const ptr_array_neuron_units_steepness_source_received,
                                                                                      enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS *const ptr_array_neuron_units_activation_function_destination_received,
                                                                                      enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_neuron_units_activation_function_source_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    size_t tmp_number_connections;

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_neuron_units_last_connection_index_destination_received[tmp_thread_global_index] = tmp_number_connections = ptr_array_neuron_units_last_connection_index_source_received[tmp_thread_global_index];
        
        tmp_number_connections -= ptr_array_neuron_units_first_connection_index_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_first_connection_index_source_received[tmp_thread_global_index];
        
        ptr_array_neuron_units_number_connections_destination_received[tmp_thread_global_index] = tmp_number_connections;

        ptr_array_neuron_units_steepness_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_steepness_source_received[tmp_thread_global_index];

        ptr_array_neuron_units_activation_function_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_activation_function_source_received[tmp_thread_global_index];
    }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Copy_Neurons(size_t const size_received,
                                                                                               size_t *const ptr_array_neuron_units_first_connection_index_destination_received,
                                                                                               size_t *const ptr_array_neuron_units_last_connection_index_destination_received,
                                                                                               size_t *const ptr_array_neuron_units_number_connections_destination_received,
                                                                                               size_t const *const ptr_array_neuron_units_first_connection_index_source_received,
                                                                                               size_t const *const ptr_array_neuron_units_last_connection_index_source_received,
                                                                                               T *const ptr_array_neuron_units_steepness_destination_received,
                                                                                               T const *const ptr_array_neuron_units_steepness_source_received,
                                                                                               enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS *const ptr_array_neuron_units_activation_function_destination_received,
                                                                                               enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_neuron_units_activation_function_source_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
                      tmp_number_connections;

    do
    {
        ptr_array_neuron_units_last_connection_index_destination_received[tmp_thread_global_index] = tmp_number_connections = ptr_array_neuron_units_last_connection_index_source_received[tmp_thread_global_index];
        
        tmp_number_connections -= ptr_array_neuron_units_first_connection_index_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_first_connection_index_source_received[tmp_thread_global_index];
        
        ptr_array_neuron_units_number_connections_destination_received[tmp_thread_global_index] = tmp_number_connections;

        ptr_array_neuron_units_steepness_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_steepness_source_received[tmp_thread_global_index];

        ptr_array_neuron_units_activation_function_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_activation_function_source_received[tmp_thread_global_index];

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Copy_Neurons(size_t const *ptr_array_neuron_units_first_connection_index_received,
                                                                                  size_t const *ptr_array_neuron_units_last_connection_index_received,
                                                                                  T_ const *ptr_array_neuron_units_steepness_received,
                                                                                  enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *ptr_array_neuron_units_activation_function_received,
                                                                                  struct CUDA_Neuron *const ptr_array_copy_first_neuron_received,
                                                                                  struct CUDA_Neuron *const ptr_array_copy_last_neuron_received)
{
    size_t const tmp_size(static_cast<size_t>(ptr_array_copy_last_neuron_received - ptr_array_copy_first_neuron_received));

    if(USE_PARALLEL && tmp_size >= warpSize)
    {
        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;
        
        this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(tmp_size,
                                                                                                                                                0u,
                                                                                                                                                tmp_dim3_grid,
                                                                                                                                                tmp_dim3_block);

        LAUNCH_KERNEL_1D(CNeural_Network__Copy_Neurons<T_>,
                                          tmp_dim3_grid,
                                          tmp_dim3_block,
                                          0_zu,
                                          tmp_size,
                                          ptr_array_copy_first_neuron_received->ptr_first_forward_connection_index,
                                          ptr_array_copy_first_neuron_received->ptr_last_forward_connection_index,
                                          ptr_array_copy_first_neuron_received->ptr_number_forward_connections,
                                          ptr_array_neuron_units_first_connection_index_received,
                                          ptr_array_neuron_units_last_connection_index_received,
                                          ptr_array_copy_first_neuron_received->ptr_activation_steepness,
                                          ptr_array_neuron_units_steepness_received,
                                          ptr_array_copy_first_neuron_received->ptr_type_activation_function,
                                          ptr_array_neuron_units_activation_function_received)

        CUDA__Check_Error();
    }
    else
    {
        for(struct CUDA_Neuron *tmp_ptr_neuron_unit_it(ptr_array_copy_first_neuron_received); tmp_ptr_neuron_unit_it != ptr_array_copy_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                                                                                 ++ptr_array_neuron_units_first_connection_index_received,
                                                                                                                                                                                                                                 ++ptr_array_neuron_units_last_connection_index_received,
                                                                                                                                                                                                                                 ++ptr_array_neuron_units_steepness_received,
                                                                                                                                                                                                                                 ++ptr_array_neuron_units_activation_function_received)
        {
            this->Copy__Neuron_Unit(tmp_ptr_neuron_unit_it,
                                         *ptr_array_neuron_units_first_connection_index_received,
                                         *ptr_array_neuron_units_last_connection_index_received,
                                         *ptr_array_neuron_units_steepness_received,
                                         *ptr_array_neuron_units_activation_function_received);
        }
    }

    // Prepare grids and blocks dimensions.
    this->Prepare__Layers__Grids_Blocks_Dimensions();
    this->Prepare__Neurons__Grids_Blocks_Dimensions();

    this->Prepare__Batch_Layers__Grids_Blocks_Dimensions(this->batch_size);
    // |END| Prepare grids and blocks dimensions. |END|
}

__device__ void inline CUDA_Neural_Network::Copy__Neuron_Unit(struct CUDA_Neuron *const ptr_copy_neuron_received,
                                                                                size_t const neuron_first_connection_index_received,
                                                                                size_t const neuron_last_connection_index_received,
                                                                                T_ const neuron_steepness_received,
                                                                                enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const neuron_activation_function_received)
{
    *ptr_copy_neuron_received->ptr_first_forward_connection_index = neuron_first_connection_index_received;
    *ptr_copy_neuron_received->ptr_last_forward_connection_index = neuron_last_connection_index_received;
    *ptr_copy_neuron_received->ptr_number_forward_connections = neuron_last_connection_index_received - neuron_first_connection_index_received;

    *ptr_copy_neuron_received->ptr_activation_steepness = neuron_steepness_received;

    *ptr_copy_neuron_received->ptr_type_activation_function = neuron_activation_function_received;
}

__device__ void CUDA_Neural_Network::Copy__FC_to_FC(struct CUDA_Neuron *ptr_copy_neuron_it_received,
                                                                                        struct CUDA_Neuron const *const ptr_copy_last_neuron_received,
                                                                                        struct CUDA_Neuron *const ptr_copy_first_neuron_received,
                                                                                        size_t const *&ptr_array_neuron_units_first_connection_index_received,
                                                                                        size_t const *&ptr_array_neuron_units_last_connection_index_received,
                                                                                        T_ const *&ptr_array_neuron_units_steepness_received,
                                                                                        enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *&ptr_array_neuron_units_activation_function_received)
{
    for(; ptr_copy_neuron_it_received != ptr_copy_last_neuron_received; ++ptr_copy_neuron_it_received)
    {
        this->Copy__Neuron_Unit(ptr_copy_neuron_it_received,
                                     *ptr_array_neuron_units_first_connection_index_received++,
                                     *ptr_array_neuron_units_last_connection_index_received++,
                                     *ptr_array_neuron_units_steepness_received++,
                                     *ptr_array_neuron_units_activation_function_received++);
    }
}

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Optimizer_Gradient_Descent__Host_To_Device(T const optimizer_time_step_received,
                                                                                                                                                T const warm_restarts_maximum_learning_rate_received,
                                                                                                                                                T const warm_restarts_T_i_received,
                                                                                                                                                T *const ptr_array_previous_delta_parameters_received,
                                                                                                                                                class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    ptr_CNeural_Network_received->optimizer_time_step = optimizer_time_step_received;
    ptr_CNeural_Network_received->warm_restarts_maximum_learning_rate = warm_restarts_maximum_learning_rate_received;
    ptr_CNeural_Network_received->warm_restarts_T_i = warm_restarts_T_i_received;
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                                   ptr_CNeural_Network_received->ptr_array_previous_delta_parameters,
                                                   ptr_array_previous_delta_parameters_received,
                                                   ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                                   ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Neural_Network::Copy__Optimizer_Gradient_Descent__Host_To_Device(T const optimizer_time_step_received,
                                                                                                                T const warm_restarts_maximum_learning_rate_received,
                                                                                                                T const warm_restarts_T_i_received,
                                                                                                                T const *const ptr_array_previous_delta_parameters_received)
{
    T *tmp_ptr_device_array_previous_delta_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_delta_parameters, this->total_parameters * sizeof(T)));
        
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_delta_parameters,
                                                    ptr_array_previous_delta_parameters_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__CNeural_Network__Copy__Optimizer_Gradient_Descent__Host_To_Device<T> <<< 1u, 1u >>> (optimizer_time_step_received,
                                                                                                                                                            warm_restarts_maximum_learning_rate_received,
                                                                                                                                                            warm_restarts_T_i_received,
                                                                                                                                                            tmp_ptr_device_array_previous_delta_parameters,
                                                                                                                                                            this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_delta_parameters)); // T

    return(true);
}
template bool Neural_Network::Copy__Optimizer_Gradient_Descent__Host_To_Device(T_ const,
                                                                                                                              T_ const,
                                                                                                                              T_ const,
                                                                                                                              T_ const *const);

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Optimizer_RPROP_minus__Host_To_Device(T *const ptr_array_previous_steps_received,
                                                                                                                                            T *const ptr_array_previous_derivates_parameters_received,
                                                                                                                                            class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_CNeural_Network_received->ptr_array_previous_steps,
                                             ptr_array_previous_steps_received,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_CNeural_Network_received->ptr_array_previous_derivatives_parameters,
                                             ptr_array_previous_derivates_parameters_received,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Neural_Network::Copy__Optimizer_RPROP_minus__Host_To_Device(T const *const ptr_array_previous_steps_received, T const *const ptr_array_previous_derivates_parameters_received)
{
    T *tmp_ptr_device_array_previous_steps,
        *tmp_ptr_device_array_previous_derivatives_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_steps, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_derivatives_parameters, this->total_parameters * sizeof(T)));
        
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_steps,
                                                    ptr_array_previous_steps_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_derivatives_parameters,
                                                    ptr_array_previous_derivates_parameters_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__CNeural_Network__Copy__Optimizer_RPROP_minus__Host_To_Device<T> <<< 1u, 1u >>> (tmp_ptr_device_array_previous_steps,
                                                                                                                                                tmp_ptr_device_array_previous_derivatives_parameters,
                                                                                                                                                this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_steps)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_derivatives_parameters)); // T

    return(true);
}
template bool Neural_Network::Copy__Optimizer_RPROP_minus__Host_To_Device(T_ const *const, T_ const *const);

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Optimizer_RPROP_plus__Host_To_Device(T const loss_received,
                                                                                                                                        T const previous_loss_received,
                                                                                                                                        T *const ptr_array_previous_steps_received,
                                                                                                                                        T *const ptr_array_previous_derivates_parameters_received,
                                                                                                                                        T *const ptr_array_previous_delta_parameters_received,
                                                                                                                                        class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    ptr_CNeural_Network_received->loss_rprop = loss_received;
    ptr_CNeural_Network_received->previous_loss_rprop = previous_loss_received;
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_CNeural_Network_received->ptr_array_previous_steps,
                                             ptr_array_previous_steps_received,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);

    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_CNeural_Network_received->ptr_array_previous_derivatives_parameters,
                                             ptr_array_previous_derivates_parameters_received,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);

    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_CNeural_Network_received->ptr_array_previous_delta_parameters,
                                             ptr_array_previous_delta_parameters_received,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Neural_Network::Copy__Optimizer_RPROP_plus__Host_To_Device(T const loss_received,
                                                                                                                    T const previous_loss_received,
                                                                                                                    T const *const ptr_array_previous_steps_received,
                                                                                                                    T const *const ptr_array_previous_derivates_parameters_received,
                                                                                                                    T const *const ptr_array_previous_delta_parameters_received)
{
    T *tmp_ptr_device_array_previous_steps,
        *tmp_ptr_device_array_previous_derivatives_parameters,
        *tmp_ptr_device_array_previous_delta_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_steps, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_derivatives_parameters, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_delta_parameters, this->total_parameters * sizeof(T)));
        
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_steps,
                                                    ptr_array_previous_steps_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_derivatives_parameters,
                                                    ptr_array_previous_derivates_parameters_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_delta_parameters,
                                                    ptr_array_previous_delta_parameters_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__CNeural_Network__Copy__Optimizer_RPROP_plus__Host_To_Device<T> <<< 1u, 1u >>> (loss_received,
                                                                                                                                                previous_loss_received,
                                                                                                                                                tmp_ptr_device_array_previous_steps,
                                                                                                                                                tmp_ptr_device_array_previous_derivatives_parameters,
                                                                                                                                                tmp_ptr_device_array_previous_delta_parameters,
                                                                                                                                                this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_steps)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_derivatives_parameters)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_delta_parameters)); // T

    return(true);
}
template bool Neural_Network::Copy__Optimizer_RPROP_plus__Host_To_Device(T_ const,
                                                                                                                                T_ const,
                                                                                                                                T_ const *const,
                                                                                                                                T_ const *const,
                                                                                                                                T_ const *const);

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Optimizer_Adam__Host_To_Device(T const optimizer_time_step_received,
                                                                                                                               T const warm_restarts_maximum_learning_rate_received,
                                                                                                                               T const warm_restarts_T_i_received,
                                                                                                                               T *const ptr_array_previous_biased_first_moment_received,
                                                                                                                               T *const ptr_array_previous_biased_second_moment_received,
                                                                                                                               class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    ptr_CNeural_Network_received->optimizer_time_step = optimizer_time_step_received;
    ptr_CNeural_Network_received->warm_restarts_maximum_learning_rate = warm_restarts_maximum_learning_rate_received;
    ptr_CNeural_Network_received->warm_restarts_T_i = warm_restarts_T_i_received;

    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_CNeural_Network_received->ptr_array_previous_biased_first_moment,
                                             ptr_array_previous_biased_first_moment_received,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);

    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_CNeural_Network_received->ptr_array_previous_biased_second_moment,
                                             ptr_array_previous_biased_second_moment_received,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Neural_Network::Copy__Optimizer_Adam__Host_To_Device(T const optimizer_time_step_received,
                                                                                               T const warm_restarts_maximum_learning_rate_received,
                                                                                               T const warm_restarts_T_i_received,
                                                                                               T const *const ptr_array_previous_biased_first_moment_received,
                                                                                               T const *const ptr_array_previous_biased_second_moment_received)
{
    T *tmp_ptr_device_array_previous_biased_first_moment,
        *tmp_ptr_device_array_previous_biased_second_moment;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_first_moment, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_second_moment, this->total_parameters * sizeof(T)));
        
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_biased_first_moment,
                                                    ptr_array_previous_biased_first_moment_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_biased_second_moment,
                                                    ptr_array_previous_biased_second_moment_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__CNeural_Network__Copy__Optimizer_Adam__Host_To_Device<T> <<< 1u, 1u >>> (optimizer_time_step_received,
                                                                                                                                            warm_restarts_maximum_learning_rate_received,
                                                                                                                                            warm_restarts_T_i_received,
                                                                                                                                            tmp_ptr_device_array_previous_biased_first_moment,
                                                                                                                                            tmp_ptr_device_array_previous_biased_second_moment,
                                                                                                                                            this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_first_moment)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_second_moment)); // T

    return(true);
}
template bool Neural_Network::Copy__Optimizer_Adam__Host_To_Device(T_ const,
                                                                                                            T_ const,
                                                                                                            T_ const,
                                                                                                            T_ const *const,
                                                                                                            T_ const *const);

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Optimizer_AMSGrad__Host_To_Device(T const optimizer_time_step_received,
                                                                                                                                    T const warm_restarts_maximum_learning_rate_received,
                                                                                                                                    T const warm_restarts_T_i_received,
                                                                                                                                    T *const ptr_array_previous_biased_first_moment_received,
                                                                                                                                    T *const ptr_array_previous_biased_second_moment_received,
                                                                                                                                    T *const ptr_array_previous_biased_second_moment_hat_received,
                                                                                                                                    class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    ptr_CNeural_Network_received->optimizer_time_step = optimizer_time_step_received;
    ptr_CNeural_Network_received->warm_restarts_maximum_learning_rate = warm_restarts_maximum_learning_rate_received;
    ptr_CNeural_Network_received->warm_restarts_T_i = warm_restarts_T_i_received;

    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_CNeural_Network_received->ptr_array_previous_biased_first_moment,
                                             ptr_array_previous_biased_first_moment_received,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);

    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_CNeural_Network_received->ptr_array_previous_biased_second_moment,
                                             ptr_array_previous_biased_second_moment_received,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);

    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_CNeural_Network_received->ptr_array_previous_biased_second_moment_hat,
                                             ptr_array_previous_biased_second_moment_hat_received,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Neural_Network::Copy__Optimizer_AMSGrad__Host_To_Device(T const optimizer_time_step_received,
                                                                                                    T const warm_restarts_maximum_learning_rate_received,
                                                                                                    T const warm_restarts_T_i_received,
                                                                                                    T const *const ptr_array_previous_biased_first_moment_received,
                                                                                                    T const *const ptr_array_previous_biased_second_moment_received,
                                                                                                    T const *const ptr_array_previous_biased_second_moment_hat_received)
{
    T *tmp_ptr_device_array_previous_biased_first_moment,
        *tmp_ptr_device_array_previous_biased_second_moment,
        *tmp_ptr_device_array_previous_biased_second_hat_moment;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_first_moment, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_second_moment, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_second_hat_moment, this->total_parameters * sizeof(T)));
        
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_biased_first_moment,
                                                    ptr_array_previous_biased_first_moment_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_biased_second_moment,
                                                    ptr_array_previous_biased_second_moment_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_biased_second_hat_moment,
                                                    ptr_array_previous_biased_second_moment_hat_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__CNeural_Network__Copy__Optimizer_AMSGrad__Host_To_Device<T> <<< 1u, 1u >>> (optimizer_time_step_received,
                                                                                                                                                warm_restarts_maximum_learning_rate_received,
                                                                                                                                                warm_restarts_T_i_received,
                                                                                                                                                tmp_ptr_device_array_previous_biased_first_moment,
                                                                                                                                                tmp_ptr_device_array_previous_biased_second_moment,
                                                                                                                                                tmp_ptr_device_array_previous_biased_second_hat_moment,
                                                                                                                                                this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_first_moment)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_second_moment)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_second_hat_moment)); // T

    return(true);
}
template bool Neural_Network::Copy__Optimizer_AMSGrad__Host_To_Device(T_ const,
                                                                                                                  T_ const,
                                                                                                                  T_ const,
                                                                                                                  T_ const *const,
                                                                                                                  T_ const *const,
                                                                                                                  T_ const *const);

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Batch_Normalization_Neurons__Host_To_Device(T *const ptr_array_neuron_units_scale_received,
                                                                                                                                                  T *const ptr_array_neuron_units_shift_received,
                                                                                                                                                  T *const ptr_array_neuron_units_mean_average_received,
                                                                                                                                                  T *const ptr_array_neuron_units_variance_average_received,
                                                                                                                                                  class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_neuron_units,
                                                 ptr_CNeural_Network_received->ptr_array_normalized_batch_units_scales,
                                                 ptr_array_neuron_units_scale_received,
                                                 ptr_CNeural_Network_received->ptr_array_dim3_grid + 3,
                                                 ptr_CNeural_Network_received->ptr_array_dim3_block + 3);

    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_neuron_units,
                                                 ptr_CNeural_Network_received->ptr_array_normalized_batch_units_shifts,
                                                 ptr_array_neuron_units_shift_received,
                                                 ptr_CNeural_Network_received->ptr_array_dim3_grid + 3,
                                                 ptr_CNeural_Network_received->ptr_array_dim3_block + 3);

    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_neuron_units,
                                                 ptr_CNeural_Network_received->ptr_array_normalized_batch_units_means_averages,
                                                 ptr_array_neuron_units_mean_average_received,
                                                 ptr_CNeural_Network_received->ptr_array_dim3_grid + 3,
                                                 ptr_CNeural_Network_received->ptr_array_dim3_block + 3);

    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_neuron_units,
                                                 ptr_CNeural_Network_received->ptr_array_normalized_batch_units_variances_averages,
                                                 ptr_array_neuron_units_variance_average_received,
                                                 ptr_CNeural_Network_received->ptr_array_dim3_grid + 3,
                                                 ptr_CNeural_Network_received->ptr_array_dim3_block + 3);
}

template<typename T>
bool Neural_Network::Copy__Batch_Normalization_Neurons__Host_To_Device(T const *const ptr_array_neuron_units_scale_received,
                                                                                                                            T const *const ptr_array_neuron_units_shift_received,
                                                                                                                            T const *const ptr_array_neuron_units_mean_average_received,
                                                                                                                            T const *const ptr_array_neuron_units_variance_average_received) const
{
    T *tmp_ptr_device_array_neurons_scale(NULL),
       *tmp_ptr_device_array_neurons_shift(NULL),
       *tmp_ptr_device_array_neurons_mean_average(NULL),
       *tmp_ptr_device_array_neurons_variance_average(NULL);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_scale, this->total_neuron_units * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_shift, this->total_neuron_units * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_mean_average, this->total_neuron_units * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_variance_average, this->total_neuron_units * sizeof(T)));
        
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_neurons_scale,
                                                    ptr_array_neuron_units_scale_received,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_neurons_shift,
                                                    ptr_array_neuron_units_shift_received,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_neurons_mean_average,
                                                    ptr_array_neuron_units_mean_average_received,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_neurons_variance_average,
                                                    ptr_array_neuron_units_variance_average_received,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__CNeural_Network__Copy__Batch_Normalization_Neurons__Host_To_Device<T> <<< 1u, 1u >>> (tmp_ptr_device_array_neurons_scale,
                                                                                                                                                              tmp_ptr_device_array_neurons_shift,
                                                                                                                                                              tmp_ptr_device_array_neurons_mean_average,
                                                                                                                                                              tmp_ptr_device_array_neurons_variance_average,
                                                                                                                                                              this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_scale)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_shift)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_mean_average)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_variance_average)); // T

    return(true);
}
template bool Neural_Network::Copy__Batch_Normalization_Neurons__Host_To_Device(T_ const *const,
                                                                                                                                        T_ const *const,
                                                                                                                                        T_ const *const,
                                                                                                                                        T_ const *const) const;

__Lch_Bds__(MAXIMUM_THREADS_PER_BLOCK, 1)
__global__ void kernel__CNeural_Network__Copy__Host_To_Device(size_t const *ptr_array_number_neurons_by_layer_received,
                                                                                                   size_t const *ptr_array_neuron_units_first_connection_index_received,
                                                                                                   size_t const *ptr_array_neuron_units_last_connection_index_received,
                                                                                                   size_t const *ptr_array_neuron_units_bias_index_received,
                                                                                                   size_t const number_loss_received,
                                                                                                   size_t const number_bit_fail_received,
                                                                                                   T_ const loss_values_received,
                                                                                                   T_ const *ptr_array_accuracy_value_received,
                                                                                                   T_ const *ptr_array_dropout_value_by_layer_received,
                                                                                                   T_ const *ptr_array_neuron_units_activation_steepness_received,
                                                                                                   T_ const *ptr_array_parameters_received,
                                                                                                   enum MyEA::Common::ENUM_TYPE_LAYER const *ptr_array_type_layer_received,
                                                                                                   enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION const *ptr_array_type_activation_received,
                                                                                                   enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT const *ptr_array_type_dropout_by_layer_received,
                                                                                                   enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const *ptr_array_tpye_normalization_by_layer_received,
                                                                                                   enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *ptr_array_neuron_units_type_activation_function_received,
                                                                                                   class Neural_Network const *const ptr_Neural_Network_received,
                                                                                                   class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{

    size_t const *tmp_ptr_array_number_neurons(ptr_array_number_neurons_by_layer_received),
                                *tmp_ptr_array_neuron_units_first_connection_index(ptr_array_neuron_units_first_connection_index_received),
                                *tmp_ptr_array_neuron_units_last_connection_index(ptr_array_neuron_units_last_connection_index_received);

    enum MyEA::Common::ENUM_TYPE_LAYER const *tmp_ptr_array_type_layer(ptr_array_type_layer_received);
    enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION const *tmp_ptr_array_type_activation(ptr_array_type_activation_received);
    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const *tmp_ptr_array_neuron_units_activation_function(ptr_array_neuron_units_type_activation_function_received);
    enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT const *tmp_ptr_array_type_dropout_by_layer_received(ptr_array_type_dropout_by_layer_received);
    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const *tmp_ptr_array_type_normalization_by_layer_received(ptr_array_tpye_normalization_by_layer_received);

    T_ const *tmp_ptr_array_dropout_value_by_layers(ptr_array_dropout_value_by_layer_received),
                  *tmp_ptr_array_neuron_units_activation_steepness(ptr_array_neuron_units_activation_steepness_received);
        
    // General parameters.
    ptr_CNeural_Network_received->type_network = ptr_Neural_Network_received->type_network;
    ptr_CNeural_Network_received->connection_rate = ptr_Neural_Network_received->connection_rate;
    ptr_CNeural_Network_received->number_recurrent_depth = ptr_Neural_Network_received->number_recurrent_depth;
    ptr_CNeural_Network_received->number_time_delays = ptr_Neural_Network_received->number_time_delays;
    // |END| General parameters. |END|

    // Gradient descent parameters.
    ptr_CNeural_Network_received->learning_rate = ptr_Neural_Network_received->learning_rate;
    ptr_CNeural_Network_received->learning_momentum = ptr_Neural_Network_received->learning_momentum;
    ptr_CNeural_Network_received->use_Nesterov = ptr_Neural_Network_received->use_Nesterov;
    // |END| Gradient descent parameters. |END|
        
    // Quickprop parameters.
    ptr_CNeural_Network_received->quickprop_decay = ptr_Neural_Network_received->quickprop_decay;
    ptr_CNeural_Network_received->quickprop_mu = ptr_Neural_Network_received->quickprop_mu;
    // |END| Quickprop parameters. |END|

    // Resillent propagation parameters.
    ptr_CNeural_Network_received->rprop_increase_factor = ptr_Neural_Network_received->rprop_increase_factor;
    ptr_CNeural_Network_received->rprop_decrease_factor = ptr_Neural_Network_received->rprop_decrease_factor;
    ptr_CNeural_Network_received->rprop_delta_min = ptr_Neural_Network_received->rprop_delta_min;
    ptr_CNeural_Network_received->rprop_delta_max = ptr_Neural_Network_received->rprop_delta_max;
    ptr_CNeural_Network_received->rprop_delta_zero = ptr_Neural_Network_received->rprop_delta_zero;
    ptr_CNeural_Network_received->loss_rprop = ptr_Neural_Network_received->loss_rprop;
    ptr_CNeural_Network_received->previous_loss_rprop = ptr_Neural_Network_received->previous_loss_rprop;
    // |END| Resillent propagation parameters. |END|
        
    // SARProp parameters.
    ptr_CNeural_Network_received->sarprop_weight_decay_shift = ptr_Neural_Network_received->sarprop_weight_decay_shift;
    ptr_CNeural_Network_received->sarprop_step_error_threshold_factor = ptr_Neural_Network_received->sarprop_step_error_threshold_factor;
    ptr_CNeural_Network_received->sarprop_step_error_shift = ptr_Neural_Network_received->sarprop_step_error_shift;
    ptr_CNeural_Network_received->sarprop_temperature = ptr_Neural_Network_received->sarprop_temperature;
    ptr_CNeural_Network_received->sarprop_epoch = ptr_Neural_Network_received->sarprop_epoch;
    // |END| SARProp parameters. |END|
        
    // Adam parameters.
     ptr_CNeural_Network_received->adam_learning_rate = ptr_Neural_Network_received->adam_learning_rate;
     ptr_CNeural_Network_received->adam_beta1 = ptr_Neural_Network_received->adam_beta1;
     ptr_CNeural_Network_received->adam_beta2 = ptr_Neural_Network_received->adam_beta2;
     ptr_CNeural_Network_received->adam_epsilon = ptr_Neural_Network_received->adam_epsilon;
     ptr_CNeural_Network_received->use_adam_bias_correction = ptr_Neural_Network_received->use_adam_bias_correction;
     ptr_CNeural_Network_received->adam_gamma = ptr_Neural_Network_received->adam_gamma;
    // |END| Adam parameters. |END|

    // Loss parameters.
    *ptr_CNeural_Network_received->ptr_array_number_loss = number_loss_received;
    *ptr_CNeural_Network_received->ptr_array_number_bit_fail = number_bit_fail_received;
    *ptr_CNeural_Network_received->ptr_array_loss_values = loss_values_received;
    ptr_CNeural_Network_received->loss_training = ptr_Neural_Network_received->loss_training;
    ptr_CNeural_Network_received->loss_validating = ptr_Neural_Network_received->loss_validating;
    ptr_CNeural_Network_received->loss_testing = ptr_Neural_Network_received->loss_testing;
    // |END| Loss parameters. |END|
        
    // Accuracy parameters.
    *ptr_CNeural_Network_received->ptr_array_accuracy_values[0u] = ptr_array_accuracy_value_received[0u];
    *ptr_CNeural_Network_received->ptr_array_accuracy_values[1u] = ptr_array_accuracy_value_received[1u];
    *ptr_CNeural_Network_received->ptr_array_accuracy_values[2u] = ptr_array_accuracy_value_received[2u];
    *ptr_CNeural_Network_received->ptr_array_accuracy_values[3u] = ptr_array_accuracy_value_received[3u];
    *ptr_CNeural_Network_received->ptr_array_accuracy_values[4u] = ptr_array_accuracy_value_received[4u];
    ptr_CNeural_Network_received->number_accuracy_trial = ptr_Neural_Network_received->number_accuracy_trial;
    ptr_CNeural_Network_received->accuracy_variance = ptr_Neural_Network_received->accuracy_variance;
    ptr_CNeural_Network_received->accuracy_training = ptr_Neural_Network_received->accuracy_training;
    ptr_CNeural_Network_received->accuracy_validating = ptr_Neural_Network_received->accuracy_validating;
    ptr_CNeural_Network_received->accuracy_testing = ptr_Neural_Network_received->accuracy_testing;
    // |END| Accuracy parameters. |END|

    // Dimension.
    ptr_CNeural_Network_received->number_inputs = ptr_Neural_Network_received->number_inputs;
    ptr_CNeural_Network_received->number_outputs = ptr_Neural_Network_received->number_outputs;
    ptr_CNeural_Network_received->total_neuron_units = ptr_Neural_Network_received->total_neuron_units;
    ptr_CNeural_Network_received->total_block_units = ptr_Neural_Network_received->total_block_units;
    ptr_CNeural_Network_received->total_cell_units = ptr_Neural_Network_received->total_cell_units;
    ptr_CNeural_Network_received->total_parameters = ptr_CNeural_Network_received->total_weights = ptr_Neural_Network_received->total_weights;
    
    // Prepare grids and blocks dimensions.
    ptr_CNeural_Network_received->Prepare__Global__Grids_Blocks_Dimensions();
    // |END| Prepare grids and blocks dimensions. |END|

    struct CUDA_Layer const *const tmp_ptr_last_layer(ptr_CNeural_Network_received->ptr_last_layer);
    struct CUDA_Layer *tmp_ptr_layer_it(ptr_CNeural_Network_received->ptr_array_layers);

    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                    ++tmp_ptr_array_type_layer,
                                                                    ++tmp_ptr_array_type_activation,
                                                                    ++tmp_ptr_array_number_neurons)
    {
        tmp_ptr_layer_it->type_layer = *tmp_ptr_array_type_layer;
        tmp_ptr_layer_it->type_activation = *tmp_ptr_array_type_activation;
        
        // Neuron_unit.
        *tmp_ptr_layer_it->ptr_number_neurons = *tmp_ptr_array_number_neurons;

        tmp_ptr_layer_it->ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_array_neuron_units + *tmp_ptr_array_number_neurons;
        // |END| Neuron_unit. |END|

        // LSTM block.
        //tmp_ptr_layer_it->ptr_last_block_unit = tmp_ptr_layer_it->ptr_array_block_units + static_cast<size_t>(tmp_ptr_original_layer_it->ptr_last_block_unit - tmp_ptr_original_layer_it->ptr_array_block_units);
        // |END| LSTM block. |END|
        
        // LSTM cell.
        //tmp_ptr_layer_it->ptr_last_cell_unit = tmp_ptr_layer_it->ptr_array_cell_units + static_cast<size_t>(tmp_ptr_original_layer_it->ptr_last_cell_unit - tmp_ptr_original_layer_it->ptr_array_cell_units);
        // |END| LSTM cell. |END|
    }
    // |END| Dimension. |END|
    
    // Allocate reduce batch.
    if(ptr_CNeural_Network_received->Allocate_Reduce_Threads() == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Allocate_Reduce_Threads\"." NEW_LINE, __FUNCTION__);

        return;
    }
    // |END| Allocate reduce batch. |END|

    // Allocate reduce cost.
    if(ptr_CNeural_Network_received->Allocate_Reduce_Cost() == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Allocate_Reduce_Cost\"." NEW_LINE, __FUNCTION__);

        return;
    }
    // |END| Allocate reduce cost. |END|

    // Allocate neurons.
    if(ptr_CNeural_Network_received->Allocate__Neuron_Units() == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Allocate__Neuron_Units\"." NEW_LINE, __FUNCTION__);

        return;
    }
    // |END| Allocate neurons. |END|
    
    // Allocate connections.
    if(ptr_CNeural_Network_received->Allocate__Parameter() == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Allocate__Parameter\"." NEW_LINE, __FUNCTION__);

        return;
    }
    // |END| Allocate connections. |END|

    // Copy connections.
    struct CUDA_Layer *const tmp_ptr_first_layer(ptr_CNeural_Network_received->ptr_array_layers);

    ptr_CNeural_Network_received->Copy_Neurons(tmp_ptr_array_neuron_units_first_connection_index,
                                                                         tmp_ptr_array_neuron_units_last_connection_index,
                                                                         tmp_ptr_array_neuron_units_activation_steepness,
                                                                         tmp_ptr_array_neuron_units_activation_function,
                                                                         tmp_ptr_first_layer->ptr_array_neuron_units,
                                                                         tmp_ptr_first_layer->ptr_array_neuron_units + ptr_CNeural_Network_received->total_neuron_units);
    // |END| Copy dimension. |END|    
    
    // Allocate neurons reduce summation.
    if(ptr_CNeural_Network_received->Allocate__Neurons_Reduce_Summation() == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Allocate__Neurons_Reduce_Summation\"." NEW_LINE, __FUNCTION__);

        return;
    }
    // |END| Allocate neurons reduce summation. |END|
    
    // Dropout.
    for(tmp_ptr_layer_it = tmp_ptr_first_layer; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                                                                         ++tmp_ptr_array_dropout_value_by_layers,
                                                                                                                         ++tmp_ptr_array_type_dropout_by_layer_received)
    {
        ptr_CNeural_Network_received->Set__Probability_Retained_Unit(tmp_ptr_layer_it,
                                                                                                   *tmp_ptr_array_type_dropout_by_layer_received == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI ? *tmp_ptr_array_dropout_value_by_layers : 1_T,
                                                                                                   false);
    }
    // |END| Dropout. |END|

    // Batch renormalization.
    for(++tmp_ptr_array_type_normalization_by_layer_received, // Skip input layer.
        tmp_ptr_layer_it = tmp_ptr_first_layer + 1; tmp_ptr_layer_it != tmp_ptr_last_layer - 1; ++tmp_ptr_layer_it,
                                                                                                                                    ++tmp_ptr_array_type_normalization_by_layer_received)
    { ptr_CNeural_Network_received->Set__Batch_Renormalization(tmp_ptr_layer_it, *tmp_ptr_array_type_normalization_by_layer_received == MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION); }
    // |END| Batch renormalization. |END|
    
    // Assign connections.
    Memory::Memory_Copy_1D<T_>(ptr_CNeural_Network_received->total_parameters,
                                                     ptr_CNeural_Network_received->ptr_array_parameters,
                                                     ptr_array_parameters_received,
                                                     ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                                     ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
    // |END| Assign connections. |END|

    // Allocate transposed weights.
    // TODO: Allocate only at training.
    if(ptr_CNeural_Network_received->Allocate_Weights_Transposed() == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Allocate_Weights_Transposed\"." NEW_LINE, __FUNCTION__);

        return;
    }
    // |END| Allocate transposed weights. |END|

    // Allocate derivative parameters.
    // TODO: Allocate only at training.
    ptr_CNeural_Network_received->device__Clear_Train_Arrays();
    // |END| Allocate derivative parameters. |END|

    // Allocate neurons reduce error.
    // TODO: Allocate only at training.
    if(ptr_CNeural_Network_received->Allocate__Neurons_Reduce_Error() == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Allocate__Neurons_Reduce_Error\"." NEW_LINE, __FUNCTION__);

        return;
    }
    // |END| Allocate neurons reduce error. |END|
        
    // Warm restarts parameters.
    ptr_CNeural_Network_received->use_Warm_Restarts = ptr_Neural_Network_received->use_Warm_Restarts;
    ptr_CNeural_Network_received->warm_restarts_decay_learning_rate = ptr_Neural_Network_received->warm_restarts_decay_learning_rate;
    ptr_CNeural_Network_received->warm_restarts_maximum_learning_rate = ptr_CNeural_Network_received->warm_restarts_initial_maximum_learning_rate = ptr_Neural_Network_received->warm_restarts_initial_maximum_learning_rate;
    ptr_CNeural_Network_received->warm_restarts_minimum_learning_rate = ptr_Neural_Network_received->warm_restarts_minimum_learning_rate;
    ptr_CNeural_Network_received->warm_restarts_T_i = ptr_CNeural_Network_received->warm_restarts_initial_T_i = ptr_Neural_Network_received->warm_restarts_initial_T_i;
    ptr_CNeural_Network_received->warm_restarts_multiplier = ptr_Neural_Network_received->warm_restarts_multiplier;
    // |END| Warm restarts parameters. |END|

    // Training parameters.
    ptr_CNeural_Network_received->Set__Optimizer_Function(ptr_Neural_Network_received->type_optimizer_function);
    ptr_CNeural_Network_received->Set__Loss_Function(ptr_Neural_Network_received->type_loss_function);
    ptr_CNeural_Network_received->bit_fail_limit = ptr_Neural_Network_received->bit_fail_limit;
    ptr_CNeural_Network_received->optimizer_time_step = ptr_Neural_Network_received->optimizer_time_step;
    ptr_CNeural_Network_received->epoch_time_step = ptr_Neural_Network_received->epoch_time_step;
    // |END| Training parameters. |END|

    // Regularization parameters.
    ptr_CNeural_Network_received->Set__Regularization__Max_Norm_Constraints(ptr_Neural_Network_received->regularization__max_norm_constraints);
    ptr_CNeural_Network_received->Set__Regularization__L1(ptr_Neural_Network_received->regularization__l1);
    ptr_CNeural_Network_received->Set__Regularization__L2(ptr_Neural_Network_received->regularization__l2);
    ptr_CNeural_Network_received->Set__Regularization__Weight_Decay(ptr_Neural_Network_received->regularization__weight_decay);
    // |END| Regularization parameters. |END|

    // Regularization parameters.
    ptr_CNeural_Network_received->Set__Normalization_Momentum_Average(ptr_Neural_Network_received->normalization_momentum_average);
    ptr_CNeural_Network_received->Set__Normalization_Epsilon(ptr_Neural_Network_received->normalization_epsilon);
    ptr_CNeural_Network_received->Set__Batch_Renormalization_r_Correction_Maximum(ptr_Neural_Network_received->batch_renormalization_r_correction_maximum);
    ptr_CNeural_Network_received->Set__Batch_Renormalization_d_Correction_Maximum(ptr_Neural_Network_received->batch_renormalization_d_correction_maximum);
    // |END| Regularization parameters. |END|

    // TODO: Transpose only on allocation of \"Allocate_Weights_Transposed\".
    ptr_CNeural_Network_received->Transpose_Weights();
}

__host__ bool CUDA_Neural_Network::Copy__Host_To_Device(class Neural_Network const *const ptr_host_Neural_Network_received, size_t const maximum_allowable_memory_received)
{
    if(ptr_host_Neural_Network_received == NULL)
    {
        PRINT_FORMAT("%s: %s: ERROR: Host pointer source is a nullptr." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__);

        return(false);
    }

    if(this->Allocate__Structure(ptr_host_Neural_Network_received->total_layers, maximum_allowable_memory_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Structure(%zu)\" function." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 maximum_allowable_memory_received);

        return(false);
    }

    size_t *tmp_ptr_device_array_number_neurons_by_layer,
              *tmp_ptr_device_array_neurons_first_connection_index,
              *tmp_ptr_device_array_neurons_last_connection_index,
              *tmp_ptr_device_array_neurons_bias_index;
        
    enum MyEA::Common::ENUM_TYPE_LAYER *tmp_ptr_device_array_type_layer;
    enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION *tmp_ptr_device_array_type_activation;
    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION *tmp_ptr_device_array_type_normalization_by_layer;
    enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT *tmp_ptr_device_array_type_dropout_by_layer;
    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS *tmp_ptr_device_array_neurons_type_activation_function;

    T_ *tmp_ptr_device_array_accuracy_values,
         *tmp_ptr_device_array_value_dropout_by_layer,
         *tmp_ptr_device_array_neurons_activation_steepness,
         *tmp_ptr_device_array_parameters;
        
    class Neural_Network *tmp_ptr_device_original_Neural_Network;

    // Allocate layers variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_number_neurons_by_layer, ptr_host_Neural_Network_received->total_layers * sizeof(size_t)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_value_dropout_by_layer, ptr_host_Neural_Network_received->total_layers * sizeof(T_)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_type_layer, ptr_host_Neural_Network_received->total_layers * sizeof(enum MyEA::Common::ENUM_TYPE_LAYER)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_type_activation, ptr_host_Neural_Network_received->total_layers * sizeof(enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_type_dropout_by_layer, ptr_host_Neural_Network_received->total_layers * sizeof(enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_type_normalization_by_layer, ptr_host_Neural_Network_received->total_layers * sizeof(enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION)));
    // |END| Allocate layers variable. |END|

    // Allocate neurons variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_first_connection_index, ptr_host_Neural_Network_received->total_neuron_units * sizeof(size_t)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_last_connection_index, ptr_host_Neural_Network_received->total_neuron_units * sizeof(size_t)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_bias_index, ptr_host_Neural_Network_received->total_neuron_units * sizeof(size_t)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_activation_steepness, ptr_host_Neural_Network_received->total_neuron_units * sizeof(T_)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_type_activation_function, ptr_host_Neural_Network_received->total_neuron_units * sizeof(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS)));
    // |END| Allocate neurons variable. |END|

    // Allocate connections.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_parameters, ptr_host_Neural_Network_received->total_parameters * sizeof(T_)));
    // |END| Allocate connections. |END|
        
    // Allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Neural_Network)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_accuracy_values, 5_zu * sizeof(T_)));
    // |END| Allocate structure neural network global variable. |END|
        
    struct Neuron_unit const *tmp_ptr_host_neuron_it(ptr_host_Neural_Network_received->ptr_array_layers->ptr_array_neuron_units);

    struct Layer const *tmp_ptr_host_layer_it(ptr_host_Neural_Network_received->ptr_array_layers);

    for(size_t tmp_index_neuron,
                            tmp_number_neurons_in_layer,
                            tmp_index_neuron_so_far(0u),
                            tmp_index_layer(0u); tmp_index_layer != ptr_host_Neural_Network_received->total_layers; ++tmp_index_layer,
                                                                                                                                                                ++tmp_ptr_host_layer_it)
    {
        // Assign layers variable.
        tmp_number_neurons_in_layer = static_cast<size_t>(tmp_ptr_host_layer_it->ptr_last_neuron_unit - tmp_ptr_host_layer_it->ptr_array_neuron_units);
        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_number_neurons_by_layer[tmp_index_layer],
                                                        &tmp_number_neurons_in_layer,
                                                        sizeof(size_t),
                                                        cudaMemcpyHostToDevice));
        
        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_value_dropout_by_layer[tmp_index_layer],
                                                        &tmp_ptr_host_layer_it->dropout_values[0u],
                                                        sizeof(T_),
                                                        cudaMemcpyHostToDevice));

        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_type_layer[tmp_index_layer],
                                                        &tmp_ptr_host_layer_it->type_layer,
                                                        sizeof(enum MyEA::Common::ENUM_TYPE_LAYER),
                                                        cudaMemcpyHostToDevice));
            
        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_type_activation[tmp_index_layer],
                                                        &tmp_ptr_host_layer_it->type_activation,
                                                        sizeof(enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION),
                                                        cudaMemcpyHostToDevice));

        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_type_dropout_by_layer[tmp_index_layer],
                                                        &tmp_ptr_host_layer_it->dropout_values[0u],
                                                        sizeof(enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT),
                                                        cudaMemcpyHostToDevice));

        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_type_normalization_by_layer[tmp_index_layer],
                                                        &tmp_ptr_host_layer_it->type_normalization,
                                                        sizeof(enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION),
                                                        cudaMemcpyHostToDevice));
        // |END| Assign layers variable. |END|

        // Assign neurons variable.
        for(tmp_index_neuron = 0u; tmp_index_neuron != tmp_number_neurons_in_layer; ++tmp_index_neuron,
                                                                                                                              ++tmp_index_neuron_so_far,
                                                                                                                              ++tmp_ptr_host_neuron_it)
        {
            CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_neurons_first_connection_index[tmp_index_neuron_so_far],
                                                            tmp_ptr_host_neuron_it->ptr_first_forward_connection_index,
                                                            sizeof(size_t),
                                                            cudaMemcpyHostToDevice));
            CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_neurons_last_connection_index[tmp_index_neuron_so_far],
                                                            tmp_ptr_host_neuron_it->ptr_last_forward_connection_index,
                                                            sizeof(size_t),
                                                            cudaMemcpyHostToDevice));
                
            CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_neurons_activation_steepness[tmp_index_neuron_so_far],
                                                            tmp_ptr_host_neuron_it->ptr_activation_steepness,
                                                            sizeof(T_),
                                                            cudaMemcpyHostToDevice));

            CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_neurons_type_activation_function[tmp_index_neuron_so_far],
                                                            tmp_ptr_host_neuron_it->ptr_type_activation_function,
                                                            sizeof(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS),
                                                            cudaMemcpyHostToDevice));
        }
        // |END| Assign neurons variable. |END|
    }
        
    // Assign connections.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_parameters,
                                                    ptr_host_Neural_Network_received->ptr_array_parameters,
                                                    ptr_host_Neural_Network_received->total_parameters * sizeof(T_),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign connections. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                            ptr_host_Neural_Network_received,
                                                            sizeof(class Neural_Network),
                                                            cudaMemcpyHostToDevice));

    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_accuracy_values,
                                                             ptr_host_Neural_Network_received->ptr_array_accuracy_values,
                                                             5_zu * sizeof(T_),
                                                             cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__CNeural_Network__Copy__Host_To_Device <<< 1u, 1u >>> (tmp_ptr_device_array_number_neurons_by_layer, // size_t
                                                                                                        tmp_ptr_device_array_neurons_first_connection_index, // size_t
                                                                                                        tmp_ptr_device_array_neurons_last_connection_index, // size_t
                                                                                                        tmp_ptr_device_array_neurons_bias_index, // size_t
                                                                                                        *ptr_host_Neural_Network_received->ptr_array_number_loss,
                                                                                                        *ptr_host_Neural_Network_received->ptr_array_number_bit_fail,
                                                                                                        *ptr_host_Neural_Network_received->ptr_array_loss_values,
                                                                                                        tmp_ptr_device_array_accuracy_values,
                                                                                                        tmp_ptr_device_array_value_dropout_by_layer, // T_
                                                                                                        tmp_ptr_device_array_neurons_activation_steepness, // T_
                                                                                                        tmp_ptr_device_array_parameters, // T_
                                                                                                        tmp_ptr_device_array_type_layer, // enum
                                                                                                        tmp_ptr_device_array_type_activation, // enum
                                                                                                        tmp_ptr_device_array_type_dropout_by_layer, // T_
                                                                                                        tmp_ptr_device_array_type_normalization_by_layer, // enum
                                                                                                        tmp_ptr_device_array_neurons_type_activation_function, // enum
                                                                                                        tmp_ptr_device_original_Neural_Network, // struct
                                                                                                        this); // class
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    // Delete layers variable.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_number_neurons_by_layer)); // size_t
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_value_dropout_by_layer)); // T_
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_type_layer)); // enum
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_type_activation)); // enum
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_type_dropout_by_layer)); // T_
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_type_normalization_by_layer)); // bool
    // |END| Delete layers variable. |END|

    // Delete neurons variable.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_first_connection_index)); // size_t
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_last_connection_index)); // size_t
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_bias_index)); // size_t
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_activation_steepness)); // T_
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_type_activation_function)); // enum
    // |END| Delete neurons variable. |END|
    
    // Delete connections.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_parameters)); // T_
    // |END| Delete connections. |END|
    
    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_accuracy_values));
    // |END| Delete neural network. |END|

    if(ptr_host_Neural_Network_received->Use__Normalization() && ptr_host_Neural_Network_received->Copy__Batch_Normalization_Neurons__Host_To_Device(ptr_host_Neural_Network_received->ptr_array_normalized_batch_units_scales,
                                                                                                                                                                                                                                      ptr_host_Neural_Network_received->ptr_array_normalized_batch_units_shifts,
                                                                                                                                                                                                                                      ptr_host_Neural_Network_received->ptr_array_normalized_batch_units_means_averages,
                                                                                                                                                                                                                                      ptr_host_Neural_Network_received->ptr_array_normalized_batch_units_variances_averages) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Copy__Batch_Normalization_Neurons__Host_To_Device()\" function." NEW_LINE,
                                    MyEA::String::Get__Time().c_str(),
                                    __FUNCTION__);

        this->Deallocate();

        return(false);
    }

    if(this->Initialize_cuRAND(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count())) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_cuRAND()\" function." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__);
        
        this->Deallocate();

        return(false);
    }

    return(true);
}

bool Neural_Network::Copy__Optimizer_Paramaters__Device_To_Host(void)
{
    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD: return(this->Copy__Optimizer_Gradient_Descent__Device_To_Host());
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus: return(this->Copy__Optimizer_RPROP_minus__Device_To_Host());
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus: return(this->Copy__Optimizer_RPROP_plus__Device_To_Host());
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: return(this->Copy__Optimizer_Adam__Device_To_Host());
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: return(this->Copy__Optimizer_AMSGrad__Device_To_Host());
        default:
            PRINT_FORMAT("%s: ERROR: Can not copy parameters of the optimizer (%u | %s)." NEW_LINE,
                                    __FUNCTION__,
                                    this->type_optimizer_function,
                                    MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
                return(false);
    }
}

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Optimizer_Gradient_Descent__Device_To_Host(T *const ptr_optimizer_time_step_received,
                                                                                                                                                T *const ptr_warm_restarts_maximum_learning_rate_received,
                                                                                                                                                T *const ptr_warm_T_i_received,
                                                                                                                                                T *const ptr_array_previous_delta_parameters_received,
                                                                                                                                                class CUDA_Neural_Network const *const ptr_CNeural_Network_received)
{
    *ptr_optimizer_time_step_received = ptr_CNeural_Network_received->optimizer_time_step;
    *ptr_warm_restarts_maximum_learning_rate_received = ptr_CNeural_Network_received->warm_restarts_maximum_learning_rate;
    *ptr_warm_T_i_received = ptr_CNeural_Network_received->warm_restarts_T_i;
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_array_previous_delta_parameters_received,
                                             ptr_CNeural_Network_received->ptr_array_previous_delta_parameters,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Neural_Network::Copy__Optimizer_Gradient_Descent__Device_To_Host(T &ref_optimizer_time_step_received,
                                                                                                                T &ref_warm_maximum_learning_rate_received,
                                                                                                                T &ref_warm_T_i_received,
                                                                                                                T *const ptr_array_previous_delta_parameters_received) const
{
    T *tmp_ptr_device_optimizer_time_step,
        *tmp_ptr_device_warm_maximum_learning_rate,
        *tmp_ptr_device_warm_T_i,
        *tmp_ptr_device_array_previous_delta_weights_received;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_optimizer_time_step, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_warm_maximum_learning_rate, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_warm_T_i, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_delta_weights_received, this->total_parameters * sizeof(T)));

    kernel__CNeural_Network__Copy__Optimizer_Gradient_Descent__Device_To_Host<T> <<< 1u, 1u >>> (tmp_ptr_device_optimizer_time_step,
                                                                                                                                                            tmp_ptr_device_warm_maximum_learning_rate,
                                                                                                                                                            tmp_ptr_device_warm_T_i,
                                                                                                                                                            tmp_ptr_device_array_previous_delta_weights_received,
                                                                                                                                                            this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&ref_optimizer_time_step_received,
                                                    tmp_ptr_device_optimizer_time_step,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_warm_maximum_learning_rate_received,
                                                    tmp_ptr_device_warm_maximum_learning_rate,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_warm_T_i_received,
                                                    tmp_ptr_device_warm_T_i,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_delta_parameters_received,
                                                    tmp_ptr_device_array_previous_delta_weights_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_optimizer_time_step)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_warm_maximum_learning_rate)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_warm_T_i)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_delta_weights_received)); // T

    return(true);
}

bool Neural_Network::Copy__Optimizer_Gradient_Descent__Device_To_Host(void)
{
    return(this->Copy__Optimizer_Gradient_Descent__Device_To_Host<T_>(this->optimizer_time_step,
                                                                                                             this->warm_restarts_maximum_learning_rate,
                                                                                                             this->warm_restarts_T_i,
                                                                                                             this->ptr_array_previous_delta_parameters));
}

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Optimizer_RPROP_minus__Device_To_Host(T *const ptr_array_previous_steps_received,
                                                                                                                                            T *const ptr_array_previous_derivates_parameters_received,
                                                                                                                                            class CUDA_Neural_Network const *const ptr_CNeural_Network_received)
{
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_array_previous_steps_received,
                                             ptr_CNeural_Network_received->ptr_array_previous_steps,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_array_previous_derivates_parameters_received,
                                             ptr_CNeural_Network_received->ptr_array_previous_derivatives_parameters,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Neural_Network::Copy__Optimizer_RPROP_minus__Device_To_Host(T *const ptr_array_previous_steps_received, T *const ptr_array_previous_derivates_parameters_received) const
{
    T *tmp_ptr_device_array_previous_steps,
        *tmp_ptr_device_array_previous_derivatives_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_steps, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_derivatives_parameters, this->total_parameters * sizeof(T)));

    kernel__CNeural_Network__Copy__Optimizer_RPROP_minus__Device_To_Host<T> <<< 1u, 1u >>> (tmp_ptr_device_array_previous_steps,
                                                                                                                                                tmp_ptr_device_array_previous_derivatives_parameters,
                                                                                                                                                this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_steps_received,
                                                    tmp_ptr_device_array_previous_steps,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_derivates_parameters_received,
                                                    tmp_ptr_device_array_previous_derivatives_parameters,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_steps)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_derivatives_parameters)); // T

    return(true);
}

bool Neural_Network::Copy__Optimizer_RPROP_minus__Device_To_Host(void)
{ return(this->Copy__Optimizer_RPROP_minus__Device_To_Host<T_>(this->ptr_array_previous_steps, this->ptr_array_previous_derivatives_parameters)); }

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Optimizer_RPROP_plus__Device_To_Host(T *ptr_loss_received,
                                                                                                                                        T *ptr_previous_loss_received,
                                                                                                                                        T *const ptr_array_previous_steps_received,
                                                                                                                                        T *const ptr_array_previous_derivates_parameters_received,
                                                                                                                                        T *const ptr_array_previous_delta_parameters_received,
                                                                                                                                        class CUDA_Neural_Network const *const ptr_CNeural_Network_received)
{
    *ptr_loss_received = ptr_CNeural_Network_received->loss_rprop;
    *ptr_previous_loss_received = ptr_CNeural_Network_received->previous_loss_rprop;
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_array_previous_steps_received,
                                             ptr_CNeural_Network_received->ptr_array_previous_steps,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_array_previous_derivates_parameters_received,
                                             ptr_CNeural_Network_received->ptr_array_previous_derivatives_parameters,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_array_previous_delta_parameters_received,
                                             ptr_CNeural_Network_received->ptr_array_previous_delta_parameters,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Neural_Network::Copy__Optimizer_RPROP_plus__Device_To_Host(T &ref_loss_received,
                                                                                                                    T &ref_previous_loss_received,
                                                                                                                    T *const ptr_array_previous_steps_received,
                                                                                                                    T *const ptr_array_previous_derivates_parameters_received,
                                                                                                                    T *const ptr_array_previous_delta_parameters_received) const
{
    T *tmp_ptr_device_loss,
        *tmp_ptr_device_previous_loss,
        *tmp_ptr_device_array_previous_steps,
        *tmp_ptr_device_array_previous_derivatives_parameters,
        *tmp_ptr_device_array_previous_delta_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_loss, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_previous_loss, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_steps, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_derivatives_parameters, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_delta_parameters, this->total_parameters * sizeof(T)));

    kernel__CNeural_Network__Copy__Optimizer_RPROP_plus__Device_To_Host<T> <<< 1u, 1u >>> (tmp_ptr_device_loss,
                                                                                                                                                tmp_ptr_device_previous_loss,
                                                                                                                                                tmp_ptr_device_array_previous_steps,
                                                                                                                                                tmp_ptr_device_array_previous_derivatives_parameters,
                                                                                                                                                tmp_ptr_device_array_previous_delta_parameters,
                                                                                                                                                this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&ref_loss_received,
                                                    tmp_ptr_device_loss,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_previous_loss_received,
                                                    tmp_ptr_device_previous_loss,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_steps_received,
                                                    tmp_ptr_device_array_previous_steps,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_derivates_parameters_received,
                                                    tmp_ptr_device_array_previous_derivatives_parameters,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_delta_parameters_received,
                                                    tmp_ptr_device_array_previous_delta_parameters,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_loss)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_previous_loss)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_steps)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_derivatives_parameters)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_delta_parameters)); // T

    return(true);
}

bool Neural_Network::Copy__Optimizer_RPROP_plus__Device_To_Host(void)
{
    return(this->Copy__Optimizer_RPROP_plus__Device_To_Host<T_>(this->loss_rprop,
                                                                                            this->previous_loss_rprop,
                                                                                            this->ptr_array_previous_steps,
                                                                                            this->ptr_array_previous_derivatives_parameters,
                                                                                            this->ptr_array_previous_delta_parameters));
}

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Optimizer_Adam__Device_To_Host(T *const ptr_optimizer_time_step_received,
                                                                                                                                T *const ptr_warm_restarts_maximum_learning_rate_received,
                                                                                                                                T *const ptr_warm_T_i_received,
                                                                                                                                T *const ptr_array_previous_biased_first_moment_received,
                                                                                                                                T *const ptr_array_previous_biased_second_moment_received,
                                                                                                                                class CUDA_Neural_Network const *const ptr_CNeural_Network_received)
{
    *ptr_optimizer_time_step_received = ptr_CNeural_Network_received->optimizer_time_step;
    *ptr_warm_restarts_maximum_learning_rate_received = ptr_CNeural_Network_received->warm_restarts_maximum_learning_rate;
    *ptr_warm_T_i_received = ptr_CNeural_Network_received->warm_restarts_T_i;

    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_array_previous_biased_first_moment_received,
                                             ptr_CNeural_Network_received->ptr_array_previous_biased_first_moment,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_array_previous_biased_second_moment_received,
                                             ptr_CNeural_Network_received->ptr_array_previous_biased_second_moment,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Neural_Network::Copy__Optimizer_Adam__Device_To_Host(T &ref_optimizer_time_step_received,
                                                                                                        T &ref_warm_maximum_learning_rate_received,
                                                                                                        T &ref_warm_T_i_received,
                                                                                                        T *const ptr_array_previous_biased_first_moment_received,
                                                                                                        T *const ptr_array_previous_biased_second_moment_received) const
{
    T *tmp_ptr_device_optimizer_time_step,
        *tmp_ptr_device_warm_maximum_learning_rate,
        *tmp_ptr_device_warm_T_i,
        *tmp_ptr_device_array_previous_biased_first_moment,
        *tmp_ptr_device_array_previous_biased_second_moment;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_optimizer_time_step, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_warm_maximum_learning_rate, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_warm_T_i, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_first_moment, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_second_moment, this->total_parameters * sizeof(T)));

    kernel__CNeural_Network__Copy__Optimizer_Adam__Device_To_Host<T> <<< 1u, 1u >>> (tmp_ptr_device_optimizer_time_step,
                                                                                                                                            tmp_ptr_device_warm_maximum_learning_rate,
                                                                                                                                            tmp_ptr_device_warm_T_i,
                                                                                                                                            tmp_ptr_device_array_previous_biased_first_moment,
                                                                                                                                            tmp_ptr_device_array_previous_biased_second_moment,
                                                                                                                                            this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&ref_optimizer_time_step_received,
                                                    tmp_ptr_device_optimizer_time_step,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_warm_maximum_learning_rate_received,
                                                    tmp_ptr_device_warm_maximum_learning_rate,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_warm_T_i_received,
                                                    tmp_ptr_device_warm_T_i,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_biased_first_moment_received,
                                                    tmp_ptr_device_array_previous_biased_first_moment,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_biased_second_moment_received,
                                                    tmp_ptr_device_array_previous_biased_second_moment,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_optimizer_time_step)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_warm_maximum_learning_rate)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_warm_T_i)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_first_moment)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_second_moment)); // T

    return(true);
}

bool Neural_Network::Copy__Optimizer_Adam__Device_To_Host(void)
{
    return(this->Copy__Optimizer_Adam__Device_To_Host<T_>(this->optimizer_time_step,
                                                                                            this->warm_restarts_maximum_learning_rate,
                                                                                            this->warm_restarts_T_i,
                                                                                            this->ptr_array_previous_biased_first_moment,
                                                                                            this->ptr_array_previous_biased_second_moment));
}

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Optimizer_AMSGrad__Device_To_Host(T *const ptr_optimizer_time_step_received,
                                                                                                                                    T *const ptr_warm_restarts_maximum_learning_rate_received,
                                                                                                                                    T *const ptr_warm_T_i_received,
                                                                                                                                    T *const ptr_array_previous_biased_first_moment_received,
                                                                                                                                    T *const ptr_array_previous_biased_second_moment_received,
                                                                                                                                    T *const ptr_array_previous_biased_second_moment_hat_received,
                                                                                                                                    class CUDA_Neural_Network const *const ptr_CNeural_Network_received)
{
    *ptr_optimizer_time_step_received = ptr_CNeural_Network_received->optimizer_time_step;
    *ptr_warm_restarts_maximum_learning_rate_received = ptr_CNeural_Network_received->warm_restarts_maximum_learning_rate;
    *ptr_warm_T_i_received = ptr_CNeural_Network_received->warm_restarts_T_i;
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_array_previous_biased_first_moment_received,
                                             ptr_CNeural_Network_received->ptr_array_previous_biased_first_moment,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_array_previous_biased_second_moment_received,
                                             ptr_CNeural_Network_received->ptr_array_previous_biased_second_moment,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                             ptr_array_previous_biased_second_moment_hat_received,
                                             ptr_CNeural_Network_received->ptr_array_previous_biased_second_moment_hat,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Neural_Network::Copy__Optimizer_AMSGrad__Device_To_Host(T &ref_optimizer_time_step_received,
                                                                                                              T &ref_warm_maximum_learning_rate_received,
                                                                                                              T &ref_warm_T_i_received,
                                                                                                              T *const ptr_array_previous_biased_first_moment_received,
                                                                                                              T *const ptr_array_previous_biased_second_moment_received,
                                                                                                              T *const ptr_array_previous_biased_second_moment_hat_received) const
{
    T *tmp_ptr_device_optimizer_time_step,
        *tmp_ptr_device_warm_maximum_learning_rate,
        *tmp_ptr_device_warm_T_i,
        *tmp_ptr_device_array_previous_biased_first_moment,
        *tmp_ptr_device_array_previous_biased_second_moment,
        *tmp_ptr_device_array_previous_biased_second_moment_hat;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_optimizer_time_step, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_warm_maximum_learning_rate, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_warm_T_i, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_first_moment, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_second_moment, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_second_moment_hat, this->total_parameters * sizeof(T)));

    kernel__CNeural_Network__Copy__Optimizer_AMSGrad__Device_To_Host<T> <<< 1u, 1u >>> (tmp_ptr_device_optimizer_time_step,
                                                                                                                                                tmp_ptr_device_warm_maximum_learning_rate,
                                                                                                                                                tmp_ptr_device_warm_T_i,
                                                                                                                                                tmp_ptr_device_array_previous_biased_first_moment,
                                                                                                                                                tmp_ptr_device_array_previous_biased_second_moment,
                                                                                                                                                tmp_ptr_device_array_previous_biased_second_moment_hat,
                                                                                                                                                this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&ref_optimizer_time_step_received,
                                                    tmp_ptr_device_optimizer_time_step,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_warm_maximum_learning_rate_received,
                                                    tmp_ptr_device_warm_maximum_learning_rate,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_warm_T_i_received,
                                                    tmp_ptr_device_warm_T_i,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_biased_first_moment_received,
                                                    tmp_ptr_device_array_previous_biased_first_moment,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_biased_second_moment_received,
                                                    tmp_ptr_device_array_previous_biased_second_moment,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_biased_second_moment_hat_received,
                                                    tmp_ptr_device_array_previous_biased_second_moment_hat,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_optimizer_time_step)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_warm_maximum_learning_rate)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_warm_T_i)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_first_moment)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_second_moment)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_second_moment_hat)); // T

    return(true);
}

bool Neural_Network::Copy__Optimizer_AMSGrad__Device_To_Host(void)
{
    return(this->Copy__Optimizer_AMSGrad__Device_To_Host<T_>(this->optimizer_time_step,
                                                                                                 this->warm_restarts_maximum_learning_rate,
                                                                                                 this->warm_restarts_T_i,
                                                                                                 this->ptr_array_previous_biased_first_moment,
                                                                                                 this->ptr_array_previous_biased_second_moment,
                                                                                                 this->ptr_array_previous_biased_second_moment_hat));
}

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Batch_Normalization_Neurons__Device_To_Host(T *const ptr_array_neuron_units_scale_received,
                                                                                                                                                T *const ptr_array_neuron_units_shift_received,
                                                                                                                                                T *const ptr_array_neuron_units_mean_average_received,
                                                                                                                                                T *const ptr_array_neuron_units_variance_average_received,
                                                                                                                                                class CUDA_Neural_Network const *const ptr_CNeural_Network_received)
{
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_neuron_units,
                                             ptr_array_neuron_units_scale_received,
                                             ptr_CNeural_Network_received->ptr_array_normalized_batch_units_scales,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 3,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 3);
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_neuron_units,
                                             ptr_array_neuron_units_shift_received,
                                             ptr_CNeural_Network_received->ptr_array_normalized_batch_units_shifts,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 3,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 3);
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_neuron_units,
                                             ptr_array_neuron_units_mean_average_received,
                                             ptr_CNeural_Network_received->ptr_array_normalized_batch_units_means_averages,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 3,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 3);
    
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_neuron_units,
                                             ptr_array_neuron_units_variance_average_received,
                                             ptr_CNeural_Network_received->ptr_array_normalized_batch_units_variances_averages,
                                             ptr_CNeural_Network_received->ptr_array_dim3_grid + 3,
                                             ptr_CNeural_Network_received->ptr_array_dim3_block + 3);
}

template<typename T>
bool Neural_Network::Copy__Batch_Normalization_Neurons__Device_To_Host(T *const ptr_array_neuron_units_scale_received,
                                                                                                                            T *const ptr_array_neuron_units_shift_received,
                                                                                                                            T *const ptr_array_neuron_units_mean_average_received,
                                                                                                                            T *const ptr_array_neuron_units_variance_average_received) const
{
    T *tmp_ptr_device_array_neurons_scale,
        *tmp_ptr_device_array_neurons_shift,
        *tmp_ptr_device_array_neurons_mean_average,
        *tmp_ptr_device_array_neurons_variance_average;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_scale, this->total_neuron_units * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_shift, this->total_neuron_units * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_mean_average, this->total_neuron_units * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_variance_average, this->total_neuron_units * sizeof(T)));

    kernel__CNeural_Network__Copy__Batch_Normalization_Neurons__Device_To_Host<T> <<< 1u, 1u >>> (tmp_ptr_device_array_neurons_scale,
                                                                                                                                                tmp_ptr_device_array_neurons_shift,
                                                                                                                                                tmp_ptr_device_array_neurons_mean_average,
                                                                                                                                                tmp_ptr_device_array_neurons_variance_average,
                                                                                                                                                this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
        
    CUDA__Safe_Call(cudaMemcpy(ptr_array_neuron_units_scale_received,
                                                    tmp_ptr_device_array_neurons_scale,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_neuron_units_shift_received,
                                                    tmp_ptr_device_array_neurons_shift,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_neuron_units_mean_average_received,
                                                    tmp_ptr_device_array_neurons_mean_average,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_neuron_units_variance_average_received,
                                                    tmp_ptr_device_array_neurons_variance_average,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_scale)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_shift)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_mean_average)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_variance_average)); // T

    return(true);
}

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Parameters__Host_To_Device(T *const ptr_array_parameters_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                                   ptr_CNeural_Network_received->ptr_array_parameters,
                                                   ptr_array_parameters_received,
                                                   ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                                   ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
}

void Neural_Network::Copy__Parameters__Host_To_Device(void)
{
    T_ *tmp_ptr_device_array_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_parameters, this->total_parameters * sizeof(T_)));

    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_parameters,
                                                    this->ptr_array_parameters,
                                                    this->total_parameters * sizeof(T_),
                                                    cudaMemcpyHostToDevice));

    kernel__CNeural_Network__Copy__Parameters__Host_To_Device<T_> <<< 1u, 1u >>> (tmp_ptr_device_array_parameters, this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_parameters)); // T_
}

bool Neural_Network::Copy__Batch_Normalization_Neurons__Device_To_Host(void)
{
    return(this->Copy__Batch_Normalization_Neurons__Device_To_Host<T_>(this->ptr_array_normalized_batch_units_scales,
                                                                                                               this->ptr_array_normalized_batch_units_shifts,
                                                                                                               this->ptr_array_normalized_batch_units_means_averages,
                                                                                                               this->ptr_array_normalized_batch_units_variances_averages));
}

template<typename T>
__global__ void kernel__CNeural_Network__Copy__Parameters__Device_To_Host(T *const ptr_array_parameters_received, class CUDA_Neural_Network const *const ptr_CNeural_Network_received)
{
    Memory::Memory_Copy_1D<T>(ptr_CNeural_Network_received->total_parameters,
                                                   ptr_array_parameters_received,
                                                   ptr_CNeural_Network_received->ptr_array_parameters,
                                                   ptr_CNeural_Network_received->ptr_array_dim3_grid + 1,
                                                   ptr_CNeural_Network_received->ptr_array_dim3_block + 1);
}

bool Neural_Network::Copy__Parameters__Device_To_Host(void)
{
    T_ *tmp_ptr_device_array_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_parameters, this->total_parameters * sizeof(T_)));

    kernel__CNeural_Network__Copy__Parameters__Device_To_Host<T_> <<< 1u, 1u >>> (tmp_ptr_device_array_parameters, this->ptr_device_Neural_Network);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(this->ptr_array_parameters,
                                                    tmp_ptr_device_array_parameters,
                                                    this->total_parameters * sizeof(T_),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_parameters)); // T_
    
    this->is_update_from_device = true;

    return(true);
}

__global__ void kernel__CNeural_Network__Copy_Device_To_Host(size_t *const ptr_array_number_neurons_by_layer_received,
                                                                                                 size_t *const ptr_array_neuron_units_first_connection_index_received,
                                                                                                 size_t *const ptr_array_neuron_units_last_connection_index_received,
                                                                                                 enum MyEA::Common::ENUM_TYPE_LAYER *const ptr_array_type_layer_received,
                                                                                                 enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION *const ptr_array_type_activation_received,
                                                                                                 enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS *const ptr_array_neuron_units_type_activation_function_received,
                                                                                                 T_ *const ptr_array_weigth_received,
                                                                                                 T_ *const ptr_array_neuron_sum_received,
                                                                                                 T_ *const ptr_array_neuron_value_received,
                                                                                                 T_ *const ptr_array_neuron_units_activation_steepness_received,
                                                                                                 class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    PRINT_FORMAT("%s: [FUNCTION DEPRECATED] TODO: Fix \"kernel__CNeural_Network__Copy_Device_To_Host\" algorithm." NEW_LINE, __FUNCTION__);
}

bool Neural_Network::Copy_Device_To_Host(bool const refresh_from_genetic_algorithm_received)
{
    if(refresh_from_genetic_algorithm_received)
    {
        T_ *tmp_ptr_device_array_parameters;

        CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_parameters, this->total_parameters * sizeof(T_)));

        kernel__CNeural_Network__Copy__Parameters__Device_To_Host<T_> <<< 1u, 1u >>> (tmp_ptr_device_array_parameters, this->ptr_device_Neural_Network);
            
    #if defined(COMPILE_DEBUG)
        CUDA__Check_Error();
    #endif

        CUDA__Safe_Call(cudaMemcpy(this->ptr_array_parameters, tmp_ptr_device_array_parameters, this->total_parameters * sizeof(T_), cudaMemcpyDeviceToHost));

        CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_parameters)); // T_
    }
    else
    {
        PRINT_FORMAT("%s: [FUNCTION DEPRECATED] Need to Fix \"Copy_Device_To_Host\" algorithm." NEW_LINE, __FUNCTION__);
    }

    return(true);
}

__global__ void kernel__CNeural_Network__Copy_Warm_Restarts_Parameters(class Neural_Network const *const ptr_Neural_Network_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Copy_Warm_Restarts_Parameters(ptr_Neural_Network_received); }

__host__ __device__ void CUDA_Neural_Network::Copy_Warm_Restarts_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
#if defined(__CUDA_ARCH__) == false
    class Neural_Network *tmp_ptr_device_original_Neural_Network;

    // Allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Neural_Network)));
    // |END| Allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    ptr_Neural_Network_received,
                                                    sizeof(class Neural_Network),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__CNeural_Network__Copy_Warm_Restarts_Parameters <<< 1u, 1u >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    this->use_Warm_Restarts = ptr_Neural_Network_received->use_Warm_Restarts;
    this->warm_restarts_decay_learning_rate = ptr_Neural_Network_received->warm_restarts_decay_learning_rate;
    this->warm_restarts_initial_maximum_learning_rate = ptr_Neural_Network_received->warm_restarts_initial_maximum_learning_rate;
    this->warm_restarts_maximum_learning_rate = ptr_Neural_Network_received->warm_restarts_maximum_learning_rate;
    this->warm_restarts_minimum_learning_rate = ptr_Neural_Network_received->warm_restarts_minimum_learning_rate;
    this->warm_restarts_initial_T_i = ptr_Neural_Network_received->warm_restarts_initial_T_i;
    this->warm_restarts_T_i = ptr_Neural_Network_received->warm_restarts_T_i;
    this->warm_restarts_multiplier = ptr_Neural_Network_received->warm_restarts_multiplier;
#endif
}
    
__global__ void kernel__CNeural_Network__Copy_Optimizer_Parameters(class Neural_Network const *const ptr_Neural_Network_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Copy__Optimizer_Parameters(ptr_Neural_Network_received); }

__host__ __device__ void CUDA_Neural_Network::Copy__Optimizer_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
#if defined(__CUDA_ARCH__) == false
    class Neural_Network *tmp_ptr_device_original_Neural_Network;

    // Allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Neural_Network)));
    // |END| Allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    ptr_Neural_Network_received,
                                                    sizeof(class Neural_Network),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__CNeural_Network__Copy_Optimizer_Parameters <<< 1u, 1u >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD: this->Copy__Gradient_Descent_Parameters(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus: this->Copy_RPROP_minus_Parameters(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus: this->Copy_RPROP_plus_Parameters(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP: this->Copy_SARProp_Parameters(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP: this->Copy_QuickProp_Parameters(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: this->Copy_Adam_Parameters(ptr_Neural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: this->Copy_NosAdam_Parameters(ptr_Neural_Network_received); break;
        default:
            PRINT_FORMAT("%s: ERROR: Can not copy parameters of the optimizer (%u)." NEW_LINE,
                        __FUNCTION__,
                        this->type_optimizer_function);
                break;
    }

    this->Copy_Warm_Restarts_Parameters(ptr_Neural_Network_received);

    this->optimizer_time_step = 0_T;
    this->epoch_time_step = 1_T;
#endif
}
    
__global__ void kernel__CNeural_Network__Copy__Gradient_Descent_Parameters(class Neural_Network const *const ptr_Neural_Network_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Copy__Gradient_Descent_Parameters(ptr_Neural_Network_received); }

__host__ __device__ void CUDA_Neural_Network::Copy__Gradient_Descent_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
#if defined(__CUDA_ARCH__) == false
    class Neural_Network *tmp_ptr_device_original_Neural_Network;

    // Allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Neural_Network)));
    // |END| Allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    ptr_Neural_Network_received,
                                                    sizeof(class Neural_Network),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__CNeural_Network__Copy__Gradient_Descent_Parameters <<< 1u, 1u >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // Gradient descent parameters.
    T_ const tmp_learning_momentum(this->learning_momentum);

    this->learning_rate = ptr_Neural_Network_received->learning_rate;
    this->learning_momentum = ptr_Neural_Network_received->learning_momentum;
    this->use_Nesterov = ptr_Neural_Network_received->use_Nesterov;
        
    if(tmp_learning_momentum == 0_T)
    { this->Allocate__Parameter__Gradient_Descent(); }
    else if(this->learning_momentum == 0_T)
    { this->Deallocate__Parameter__Gradient_Descent(); }
    // |END| Gradient descent parameters. |END|
#endif
}

__global__ void kernel__CNeural_Network__Copy_QuickProp_Parameters(class Neural_Network const *const ptr_Neural_Network_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Copy_QuickProp_Parameters(ptr_Neural_Network_received); }

__host__ __device__ void CUDA_Neural_Network::Copy_QuickProp_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
#if defined(__CUDA_ARCH__) == false
    class Neural_Network *tmp_ptr_device_original_Neural_Network;

    // Allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Neural_Network)));
    // |END| Allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    ptr_Neural_Network_received,
                                                    sizeof(class Neural_Network),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__CNeural_Network__Copy_QuickProp_Parameters <<< 1u, 1u >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // Quickprop parameters.
    this->quickprop_decay = ptr_Neural_Network_received->quickprop_decay;
    this->quickprop_mu = ptr_Neural_Network_received->quickprop_mu;
    // |END| Quickprop parameters. |END|
#endif
}

__global__ void kernel__CNeural_Network__Copy_RPROP_minus_Parameters(class Neural_Network const *const ptr_Neural_Network_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Copy_RPROP_minus_Parameters(ptr_Neural_Network_received); }

__host__ __device__ void CUDA_Neural_Network::Copy_RPROP_minus_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
#if defined(__CUDA_ARCH__) == false
    class Neural_Network *tmp_ptr_device_original_Neural_Network;

    // Allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Neural_Network)));
    // |END| Allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    ptr_Neural_Network_received,
                                                    sizeof(class Neural_Network),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__CNeural_Network__Copy_RPROP_minus_Parameters <<< 1u, 1u >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // Resillent propagation minus parameters.
    this->rprop_increase_factor = ptr_Neural_Network_received->rprop_increase_factor;
    this->rprop_decrease_factor = ptr_Neural_Network_received->rprop_decrease_factor;
    this->rprop_delta_min = ptr_Neural_Network_received->rprop_delta_min;
    this->rprop_delta_max = ptr_Neural_Network_received->rprop_delta_max;
    this->rprop_delta_zero = ptr_Neural_Network_received->rprop_delta_zero;
    // |END| Resillent propagation minus parameters. |END|
#endif
}

__global__ void kernel__CNeural_Network__Copy_RPROP_plus_Parameters(class Neural_Network const *const ptr_Neural_Network_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Copy_RPROP_plus_Parameters(ptr_Neural_Network_received); }

__host__ __device__ void CUDA_Neural_Network::Copy_RPROP_plus_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
#if defined(__CUDA_ARCH__) == false
    class Neural_Network *tmp_ptr_device_original_Neural_Network;

    // Allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Neural_Network)));
    // |END| Allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    ptr_Neural_Network_received,
                                                    sizeof(class Neural_Network),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__CNeural_Network__Copy_RPROP_plus_Parameters <<< 1u, 1u >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // Resillent propagation plus parameters.
    this->Copy_RPROP_minus_Parameters(ptr_Neural_Network_received);

    this->loss_rprop = ptr_Neural_Network_received->loss_rprop;
    this->previous_loss_rprop = ptr_Neural_Network_received->previous_loss_rprop;
    // |END| Resillent propagation plus parameters. |END|
#endif
}

__global__ void kernel__CNeural_Network__Copy_SARProp_Parameters(class Neural_Network const *const ptr_Neural_Network_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Copy_SARProp_Parameters(ptr_Neural_Network_received); }

__host__ __device__ void CUDA_Neural_Network::Copy_SARProp_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
#if defined(__CUDA_ARCH__) == false
    class Neural_Network *tmp_ptr_device_original_Neural_Network;

    // Allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Neural_Network)));
    // |END| Allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    ptr_Neural_Network_received,
                                                    sizeof(class Neural_Network),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__CNeural_Network__Copy_SARProp_Parameters <<< 1u, 1u >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // SARProp parameters.
    this->sarprop_weight_decay_shift = ptr_Neural_Network_received->sarprop_weight_decay_shift;
    this->sarprop_step_error_threshold_factor = ptr_Neural_Network_received->sarprop_step_error_threshold_factor;
    this->sarprop_step_error_shift = ptr_Neural_Network_received->sarprop_step_error_shift;
    this->sarprop_temperature = ptr_Neural_Network_received->sarprop_temperature;
    this->sarprop_epoch = ptr_Neural_Network_received->sarprop_epoch;
    // |END| SARProp parameters. |END|
#endif
}

__global__ void kernel__CNeural_Network__Copy_Adam_Parameters(class Neural_Network const *const ptr_Neural_Network_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Copy_Adam_Parameters(ptr_Neural_Network_received); }

__host__ __device__ void CUDA_Neural_Network::Copy_Adam_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
#if defined(__CUDA_ARCH__) == false
    class Neural_Network *tmp_ptr_device_original_Neural_Network;

    // Allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Neural_Network)));
    // |END| Allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    ptr_Neural_Network_received,
                                                    sizeof(class Neural_Network),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__CNeural_Network__Copy_Adam_Parameters <<< 1u, 1u >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // Adam parameters.
     this->adam_learning_rate = ptr_Neural_Network_received->adam_learning_rate;
     this->adam_beta1 = ptr_Neural_Network_received->adam_beta1;
     this->adam_beta2 = ptr_Neural_Network_received->adam_beta2;
     this->adam_epsilon = ptr_Neural_Network_received->adam_epsilon;
     this->use_adam_bias_correction = ptr_Neural_Network_received->use_adam_bias_correction;
    // |END| Adam parameters. |END|
#endif
}

__global__ void kernel__CNeural_Network__Copy_NosAdam_Parameters(class Neural_Network const *const ptr_Neural_Network_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Copy_NosAdam_Parameters(ptr_Neural_Network_received); }

__host__ __device__ void CUDA_Neural_Network::Copy_NosAdam_Parameters(class Neural_Network const *const ptr_Neural_Network_received)
{
#if defined(__CUDA_ARCH__) == false
    class Neural_Network *tmp_ptr_device_original_Neural_Network;

    // Allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Neural_Network)));
    // |END| Allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    ptr_Neural_Network_received,
                                                    sizeof(class Neural_Network),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__CNeural_Network__Copy_NosAdam_Parameters <<< 1u, 1u >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // Adam parameters.
     this->adam_learning_rate = ptr_Neural_Network_received->adam_learning_rate;
     this->adam_beta1 = ptr_Neural_Network_received->adam_beta1;
     this->adam_beta2 = ptr_Neural_Network_received->adam_beta2;
     this->adam_epsilon = ptr_Neural_Network_received->adam_epsilon;
     this->use_adam_bias_correction = ptr_Neural_Network_received->use_adam_bias_correction;
     this->adam_gamma = ptr_Neural_Network_received->adam_gamma;
    // |END| Adam parameters. |END|
#endif
}

__global__ void kernel__CNeural_Network__Copy_Dropout(T_ const *const ptr_array_probability_retained_unit_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->device__Copy_Dropout(ptr_array_probability_retained_unit_received); }

__host__ void CUDA_Neural_Network::Copy__Dropout(class Neural_Network const *const ptr_Neural_Network_received)
{
    T_ *tmp_ptr_device_array_probability_retained_unit_by_layer;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_probability_retained_unit_by_layer, ptr_Neural_Network_received->total_layers * sizeof(T_)));

    for(size_t tmp_index_layer(0u); tmp_index_layer != ptr_Neural_Network_received->total_layers; ++tmp_index_layer)
    {
        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_probability_retained_unit_by_layer[tmp_index_layer],
                                                        &(ptr_Neural_Network_received->ptr_array_layers + tmp_index_layer)->dropout_values[0u],
                                                        sizeof(T_),
                                                        cudaMemcpyHostToDevice));
    }

    kernel__CNeural_Network__Copy_Dropout <<< 1u, 1u >>> (tmp_ptr_device_array_probability_retained_unit_by_layer, this);

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_probability_retained_unit_by_layer));
}

__device__ void CUDA_Neural_Network::device__Copy_Dropout(T_ const *ptr_array_probability_retained_unit_received)
{
    struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer - 1); // Subtract output layer.
    struct CUDA_Layer *tmp_ptr_layer_it(this->ptr_array_layers);

    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                    ++ptr_array_probability_retained_unit_received)
    { this->Set__Probability_Retained_Unit(tmp_ptr_layer_it, *ptr_array_probability_retained_unit_received); }
}

__global__ void kernel__CNeural_Network__Copy_Normalization(enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const *const ptr_array_normalization_by_layers_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->device__Copy_Normalization(ptr_array_normalization_by_layers_received); }

__host__ void CUDA_Neural_Network::Copy__Normalization(class Neural_Network const *const ptr_Neural_Network_received)
{
    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION *tmp_ptr_device_array_normalization_by_layers(NULL);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_normalization_by_layers, ptr_Neural_Network_received->total_layers * sizeof(enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION)));

    for(size_t tmp_index_layer(0u); tmp_index_layer != ptr_Neural_Network_received->total_layers; ++tmp_index_layer)
    {
        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_normalization_by_layers[tmp_index_layer],
                                                        &(ptr_Neural_Network_received->ptr_array_layers + tmp_index_layer)->type_normalization,
                                                        sizeof(bool),
                                                        cudaMemcpyHostToDevice));
    }

    kernel__CNeural_Network__Copy_Normalization <<< 1u, 1u >>> (tmp_ptr_device_array_normalization_by_layers, this);

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_normalization_by_layers));

    if(ptr_Neural_Network_received->Use__Normalization())
    {
        if(ptr_Neural_Network_received->Copy__Batch_Normalization_Neurons__Host_To_Device(ptr_Neural_Network_received->ptr_array_normalized_batch_units_scales,
                                                                                                                                         ptr_Neural_Network_received->ptr_array_normalized_batch_units_shifts,
                                                                                                                                         ptr_Neural_Network_received->ptr_array_normalized_batch_units_means_averages,
                                                                                                                                         ptr_Neural_Network_received->ptr_array_normalized_batch_units_variances_averages) == false)
        {
            PRINT_FORMAT("%s: ERROR: From \"Copy__Batch_Normalization_Neurons__Host_To_Device\"." NEW_LINE,
                                     __FUNCTION__);
        }
    }
}

__device__ void CUDA_Neural_Network::device__Copy_Normalization(enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const *ptr_array_normalization_by_layers_received)
{
    struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer - 1); // Subtract output layer.
    struct CUDA_Layer *tmp_ptr_layer_it(this->ptr_array_layers);
        
    // Hidden layer.
    for(++ptr_array_normalization_by_layers_received,
        ++tmp_ptr_layer_it; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                                                ++ptr_array_normalization_by_layers_received)
    { this->Set__Batch_Renormalization(tmp_ptr_layer_it, *ptr_array_normalization_by_layers_received == MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION); }
    // |END| Hidden layer. |END|
}
    
