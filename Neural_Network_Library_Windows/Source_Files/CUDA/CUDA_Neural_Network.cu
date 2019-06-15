#include <Tools/CUDA_Reallocate.cuh>
#include <Tools/CUDA_Fill_1D.cuh>
#include <Tools/CUDA_Zero_1D.cuh>

#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Dataset_Manager.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>
#include <CUDA/CUDA_Activation_Functions.cuh>
#include <CUDA/CUDA_Reduce.cuh>
#include <CUDA/CUDA_Transpose.cuh>
#include <CUDA/CUDA_Activation_Functions.cuh>

#include <curand_kernel.h>

#include <Neural_Network/Neural_Network.hpp>

__device__ void Activation_Real(T_ &ref_value_received,
                                              T_ const summation_received,
                                              enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS const type_activation_function_received)
{
    switch(type_activation_function_received)
    {
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_SIGMOID: ref_value_received = Activation_Function_SIGMOID_real_t<T_>(summation_received); break;
        case MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_LEAKY_RELU: ref_value_received = Activation_Function_LRELU_real_t<T_>(summation_received); break;
        default:
            PRINT_FORMAT("%s: ERROR: Activation function (%u) not implemented yet!" NEW_LINE,
                                     __FUNCTION__,
                                     type_activation_function_received);
                break;
    }
}

__device__ __host__ CUDA_Neural_Network::CUDA_Neural_Network(void) { }

__global__ void kernel__CNeural_Network__Add_CUDA_Device(int const index_device_received,
                                                                                            struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received,
                                                                                            class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Add_CUDA_Device(index_device_received, ptr_struct_cudaDeviceProp_received); }
    
__device__ bool CUDA_Neural_Network::Add_CUDA_Device(int const index_device_received, struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received)
{
    if(this->_ptr_Class_Device_Information_Array == nullptr)
    { this->_ptr_Class_Device_Information_Array = new class CUDA_Device_Information_Array; }

    return(this->_ptr_Class_Device_Information_Array->Push_Back(index_device_received, ptr_struct_cudaDeviceProp_received));
}

__host__ bool CUDA_Neural_Network::Initialize_CUDA_Device(void)
{
    int tmp_index_device(0),
        tmp_number_CUDA_devices;
        
    struct cudaDeviceProp tmp_struct_cudaDeviceProp,
                                     *tmp_ptr_device_struct_cudaDeviceProp(NULL);

    CUDA__Safe_Call(cudaGetDeviceCount(&tmp_number_CUDA_devices));
        
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_struct_cudaDeviceProp, sizeof(struct cudaDeviceProp)));

    for(; tmp_index_device != tmp_number_CUDA_devices; ++tmp_index_device)
    {
        CUDA__Safe_Call(cudaGetDeviceProperties(&tmp_struct_cudaDeviceProp, tmp_index_device));

        CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_struct_cudaDeviceProp,
                                                        &tmp_struct_cudaDeviceProp,
                                                        sizeof(struct cudaDeviceProp),
                                                        cudaMemcpyKind::cudaMemcpyHostToDevice));

        kernel__CNeural_Network__Add_CUDA_Device <<< 1u, 1u >>> (tmp_index_device,
                                                                                                      tmp_ptr_device_struct_cudaDeviceProp,
                                                                                                      this);
            
    #if defined(COMPILE_DEBUG)
        CUDA__Check_Error();
    #endif
    }

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_struct_cudaDeviceProp));

    return(true);
}

__device__ __host__ CUDA_Neural_Network::~CUDA_Neural_Network(void)
{ this->Deallocate(); }

__device__ class CUDA_Device_Information_Array *CUDA_Neural_Network::Get__Class_Device_Information_Array(void) const { return(this->_ptr_Class_Device_Information_Array); }

// Public function.
__device__ bool CUDA_Neural_Network::Set__Batch_Renormalization(size_t const index_layer_received, bool const Set__received)
{
    if(index_layer_received >= this->total_layers)
    {
        PRINT_FORMAT("%s: ERROR: Layer received (%u) as argument overflow the number of layers (%u) in the neural network." NEW_LINE,
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

    return(this->Set__Batch_Renormalization(this->ptr_array_layers + index_layer_received, Set__received));
}

// Private function.
__device__ bool CUDA_Neural_Network::Set__Batch_Renormalization(struct CUDA_Layer *const ptr_layer_received, bool const Set__received)
{
    struct CUDA_Layer const *tmp_ptr_last_layer;
    struct CUDA_Layer *tmp_ptr_layer_it;
    
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Layer received as argument is a nullptr." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        PRINT_FORMAT("%s: ERROR: Layer received as argument is the input layer." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        PRINT_FORMAT("%s: ERROR: Layer received as argument is the output layer." NEW_LINE, __FUNCTION__);

        return(false);
    }

    if(ptr_layer_received->use_Batch_Renormalization != Set__received)
    {
        ptr_layer_received->use_Batch_Renormalization = Set__received;

        if(Set__received)
        {
            if(this->use_Batch_Renormalization == false)
            {
                if(this->Allocate__Batch_Normalization() == false)
                {
                    PRINT_FORMAT("%s: ERROR: From \"Allocate__Batch_Normalization\"." NEW_LINE, __FUNCTION__);

                    return(false);
                }
                else if(Allocate__Neurons_Reduce_Batch_Normalization() == false)
                {
                    PRINT_FORMAT("%s: ERROR: From \"Allocate__Neurons_Reduce_Batch_Normalization\"." NEW_LINE, __FUNCTION__);

                    return(false);
                }

                this->use_Batch_Renormalization = true;
            }
        }
        else // Check if we use batch renormalization
        {
            // TODO: Replace the checkup by a counter.
            bool tmp_use_Batch_Renormalization(false);
        
            // Loop through each layer to do a check if a layer use batch renormalization.
            for(tmp_ptr_last_layer = this->ptr_last_layer,
                tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
            {
                if(tmp_ptr_layer_it->use_Batch_Renormalization)
                {
                    tmp_use_Batch_Renormalization = true;

                    break;
                }
            }
            
            this->use_Batch_Renormalization = tmp_use_Batch_Renormalization;
            // |END| Loop through each layer to do a check if a layer use batch renormalization. |END|

            if(tmp_use_Batch_Renormalization == false)
            {
                this->Deallocate_Batch_Reduce();
                this->Deallocate__Normalized_Unit__Batch_Normalization();
                this->Remove_Batch_Normalization();
            }
        }
    }

    return(true);
}

__device__ void CUDA_Neural_Network::Transpose_Layer_Forward__Batch_Normalization(struct CUDA_Layer *const ptr_layer_it_received)
{
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    Transpose::Transpose<T_>(this->batch_size * *ptr_layer_it_received->ptr_number_neurons,
                                            this->batch_size,
                                            *ptr_layer_it_received->ptr_number_neurons,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                            ptr_layer_it_received->ptr_dim3_grid_batch_neurons,
                                            ptr_layer_it_received->ptr_dim3_block_batch_neurons);

    Transpose::Transpose<T_>(this->batch_size * *ptr_layer_it_received->ptr_number_neurons,
                                            this->batch_size,
                                            *ptr_layer_it_received->ptr_number_neurons,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                            ptr_layer_it_received->ptr_dim3_grid_batch_neurons,
                                            ptr_layer_it_received->ptr_dim3_block_batch_neurons);
}

__device__ void CUDA_Neural_Network::Transpose_Layer_Backward__Batch_Normalization(struct CUDA_Layer *const ptr_layer_it_received)
{
    struct CUDA_Neuron *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    Transpose::Transpose<T_>(this->batch_size * *ptr_layer_it_received->ptr_number_neurons,
                                            this->batch_size,
                                            *ptr_layer_it_received->ptr_number_neurons,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                            ptr_layer_it_received->ptr_dim3_grid_batch_neurons,
                                            ptr_layer_it_received->ptr_dim3_block_batch_neurons);

    Transpose::Transpose<T_>(this->batch_size * *ptr_layer_it_received->ptr_number_neurons,
                                            this->batch_size,
                                            *ptr_layer_it_received->ptr_number_neurons,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                            ptr_layer_it_received->ptr_dim3_grid_batch_neurons,
                                            ptr_layer_it_received->ptr_dim3_block_batch_neurons);
}

__device__ void CUDA_Neural_Network::Transpose_Weights(void)
{
    // By default the synchronized state is set to true.
    bool tmp_synchronized(true);

    struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer),
                                                    *tmp_ptr_layer_it(this->ptr_array_layers + 1);
    
    T_ const *tmp_ptr_array_parameters(this->ptr_array_parameters);
    T_ *tmp_ptr_array_weights_transposed(this->ptr_array_transposed_weights);

    size_t tmp_number_weights_in_layer,
                      tmp_number_neurons_in_layer,
                      tmp_number_connections_to_each_neurons;

    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                   tmp_ptr_array_parameters += tmp_number_weights_in_layer,
                                                                   tmp_ptr_array_weights_transposed += tmp_number_weights_in_layer)
    {
        tmp_number_neurons_in_layer = *tmp_ptr_layer_it->ptr_number_neurons - 1_zu; // Subtract bias.
        
        tmp_number_connections_to_each_neurons = *tmp_ptr_layer_it->ptr_array_neuron_units->ptr_number_forward_connections;

        tmp_number_weights_in_layer = tmp_number_neurons_in_layer * tmp_number_connections_to_each_neurons;

        Transpose::Transpose<T_>(tmp_number_weights_in_layer,
                                                 tmp_number_neurons_in_layer,
                                                 tmp_number_connections_to_each_neurons,
                                                 tmp_ptr_array_weights_transposed,
                                                 tmp_ptr_array_parameters,
                                                 tmp_ptr_layer_it->ptr_dim3_grid_weights,
                                                 tmp_ptr_layer_it->ptr_dim3_block_weights);

        //PRINT_FORMAT("%s: Transposed" NEW_LINE, __FUNCTION__);

        // Do we need to synchronise? Based on "Transpose" Function.
        // => Synchronisation before using the transposed weights of the layer.
        if(tmp_number_weights_in_layer >= warpSize) { tmp_synchronized = false; }
    }

    CUDA__Device_Synchronise(tmp_synchronized, MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);
}

__device__ void CUDA_Neural_Network::Prepare__Global__Grids_Blocks_Dimensions(void)
{
    this->Prepare__Layers__Grids_Blocks_Dimensions();
    this->Prepare__Neurons__Grids_Blocks_Dimensions();
    this->Prepare__Parameters__Grids_Blocks_Dimensions();

    this->Prepare__Threads__Grids_Blocks_Dimensions(this->number_threads);
    this->Prepare__Batch__Grids_Blocks_Dimensions(this->batch_size);
}

__device__ bool CUDA_Neural_Network::Prepare__Layers__Grids_Blocks_Dimensions(void)
{
    size_t tmp_number_neurons_in_layer,
                      tmp_number_connections_to_each_neurons;

    struct CUDA_Layer const *tmp_ptr_last_layer(this->ptr_last_layer);
    struct CUDA_Layer *tmp_ptr_layer_it(this->ptr_array_layers);
    
    class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        if((tmp_number_neurons_in_layer = *tmp_ptr_layer_it->ptr_number_neurons) != 0u)
        {
            --tmp_number_neurons_in_layer; // Subtract bias.
            
            tmp_ptr_CUDA_Device->Grid_Block_1Dimensions(tmp_number_neurons_in_layer,
                                                                                     0u,
                                                                                     *tmp_ptr_layer_it->ptr_dim3_grid_neurons,
                                                                                     *tmp_ptr_layer_it->ptr_dim3_block_neurons);
            
            tmp_ptr_CUDA_Device->Grid_Block_Dynamic_Parallelisme(tmp_number_neurons_in_layer,
                                                                                                   0u,
                                                                                                   *tmp_ptr_layer_it->ptr_dim3_grid_neurons_DP,
                                                                                                   *tmp_ptr_layer_it->ptr_dim3_block_neurons_DP);
            
            tmp_ptr_CUDA_Device->Grid_Block_cuRAND_1Dimensions(tmp_number_neurons_in_layer,
                                                                                                   0u,
                                                                                                   *tmp_ptr_layer_it->ptr_dim3_grid_neurons_cuRAND,
                                                                                                   *tmp_ptr_layer_it->ptr_dim3_block_neurons_cuRAND);
            
            tmp_number_connections_to_each_neurons = *tmp_ptr_layer_it->ptr_array_neuron_units->ptr_number_forward_connections;

            // If layer have some weights.
            if(tmp_number_neurons_in_layer * tmp_number_connections_to_each_neurons != 0u)
            {
                tmp_ptr_CUDA_Device->Grid_Block_Transpose_2Dimensions(tmp_number_neurons_in_layer,
                                                                                                          tmp_number_connections_to_each_neurons,
                                                                                                          0u,
                                                                                                          *tmp_ptr_layer_it->ptr_dim3_grid_weights,
                                                                                                          *tmp_ptr_layer_it->ptr_dim3_block_weights);
            }
        }
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Prepare__Neurons__Grids_Blocks_Dimensions(void)
{
    class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

    if(this->total_neuron_units != 0_zu)
    {
        struct CUDA_Neuron *tmp_ptr_neuron_unit_it((this->ptr_array_layers + 1)->ptr_array_neuron_units);
        struct CUDA_Neuron const *tmp_ptr_last_neuron_unit((this->ptr_last_layer - 1)->ptr_last_neuron_unit - 1); // Subtract bias.
        
        // Grid | Block: [3]: Total neurons.
        tmp_ptr_CUDA_Device->Grid_Block_1Dimensions(this->total_neuron_units,
                                                                                 0u,
                                                                                 this->ptr_array_dim3_grid[3u],
                                                                                 this->ptr_array_dim3_block[3u]);
        
        // Grid | Block: [6]: Max norm constraints.
        tmp_ptr_CUDA_Device->Grid_Block_Dynamic_Parallelisme(this->total_neuron_units - this->number_inputs - 1u,
                                                                                               0u,
                                                                                               this->ptr_array_dim3_grid[6u],
                                                                                               this->ptr_array_dim3_block[6u]);
        
        if(this->total_neuron_units_allocated != 0u)
        {
            for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
            {
                if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections != 0u) // If is not a bias.
                {
                    tmp_ptr_CUDA_Device->Grid_Block_1Dimensions(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections,
                                                                                             0u,
                                                                                             *tmp_ptr_neuron_unit_it->ptr_dim3_grid_connections,
                                                                                             *tmp_ptr_neuron_unit_it->ptr_dim3_block_connections);
                }
            }
        }
    }

    return(true);
}

__device__ void CUDA_Neural_Network::Prepare__Parameters__Grids_Blocks_Dimensions(void)
{
    class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
    
    // Grid | Block: [1]: Total parameters.
    tmp_ptr_CUDA_Device->Grid_Block_1Dimensions(this->total_parameters,
                                                                             0u,
                                                                             this->ptr_array_dim3_grid[1u],
                                                                             this->ptr_array_dim3_block[1u]);
    
    // Grid | Block: [2]: Total weights.
    tmp_ptr_CUDA_Device->Grid_Block_1Dimensions(this->total_weights,
                                                                             0u,
                                                                             this->ptr_array_dim3_grid[2u],
                                                                             this->ptr_array_dim3_block[2u]);
    
    // Grid | Block: [9]: Total weights cuRAND MTGP32.
    tmp_ptr_CUDA_Device->Grid_Block_cuRAND_1Dimensions(this->total_weights,
                                                                                            0u,
                                                                                            this->ptr_array_dim3_grid[8u],
                                                                                            this->ptr_array_dim3_block[8u]);
}

__device__ void CUDA_Neural_Network::Prepare__Threads__Grids_Blocks_Dimensions(size_t const number_threads_received)
{
    class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
    
    // Grid | Block: [0]: Total threads
    tmp_ptr_CUDA_Device->Grid_Block_1Dimensions(number_threads_received,
                                                                             0u,
                                                                             this->ptr_array_dim3_grid[0u],
                                                                             this->ptr_array_dim3_block[0u]);
    
    // Grid | Block: [7]: Total threads DP
    tmp_ptr_CUDA_Device->Grid_Block_Dynamic_Parallelisme(number_threads_received,
                                                                                           0u,
                                                                                           this->ptr_array_dim3_grid[7u],
                                                                                           this->ptr_array_dim3_block[7u]);
    
    this->Prepare__Threads_Parameters__Grids_Blocks_Dimensions(number_threads_received);
}

__device__ void CUDA_Neural_Network::Prepare__Threads_Parameters__Grids_Blocks_Dimensions(size_t const number_threads_received)
{
    // Grid | Block: [3]: (threads - 1) * total parameters
    if(number_threads_received > 1u)
    {
        this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions((number_threads_received - 1u) * this->total_parameters,
                                                                                                                                               0u,
                                                                                                                                               this->ptr_array_dim3_grid[4u],
                                                                                                                                               this->ptr_array_dim3_block[4u]);
    }
    else
    {
        this->ptr_array_dim3_grid[4u].x = 1u;
        this->ptr_array_dim3_grid[4u].y = 1u;
        this->ptr_array_dim3_grid[4u].z = 1u;

        this->ptr_array_dim3_block[4u].x = 1u;
        this->ptr_array_dim3_block[4u].y = 1u;
        this->ptr_array_dim3_block[4u].z = 1u;
    }
}

__device__ void CUDA_Neural_Network::Prepare__Batch__Grids_Blocks_Dimensions(size_t const batch_size_received)
{
    this->Prepare__Batch_Neurons__Grids_Blocks_Dimensions(batch_size_received);
    this->Prepare__Batch_Layers__Grids_Blocks_Dimensions(batch_size_received);
}

__device__ void CUDA_Neural_Network::Prepare__Batch_Layers__Grids_Blocks_Dimensions(size_t const batch_size_received)
{
    size_t tmp_number_neurons_in_layer;

    struct CUDA_Layer const *tmp_ptr_last_layer(this->ptr_last_layer);
    struct CUDA_Layer *tmp_ptr_layer_it(this->ptr_array_layers);
    
    class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        if((tmp_number_neurons_in_layer = *tmp_ptr_layer_it->ptr_number_neurons) != 0u)
        {
            tmp_ptr_CUDA_Device->Grid_Block_Transpose_2Dimensions(tmp_number_neurons_in_layer,
                                                                                                      batch_size_received,
                                                                                                      0u,
                                                                                                      *tmp_ptr_layer_it->ptr_dim3_grid_batch_neurons,
                                                                                                      *tmp_ptr_layer_it->ptr_dim3_block_batch_neurons);
        }
    }
}

__device__ void CUDA_Neural_Network::Prepare__Batch_Neurons__Grids_Blocks_Dimensions(size_t const batch_size_received)
{
    if(this->total_neuron_units != 0_zu)
    {
        // Grid | Block: [5]: batch * total neurons
        this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(batch_size_received * this->total_neuron_units,
                                                                                                                                               0u,
                                                                                                                                               this->ptr_array_dim3_grid[5u],
                                                                                                                                               this->ptr_array_dim3_block[5u]);
    }
}

__global__ void kernel__CNeural_Network__Set__Normalization_Momentum_Average(T_ const momentum_average_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Normalization_Momentum_Average(momentum_average_received); }

__host__ __device__ bool CUDA_Neural_Network::Set__Normalization_Momentum_Average(T_ const momentum_average_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CNeural_Network__Set__Normalization_Momentum_Average <<< 1u, 1u >>> (momentum_average_received, this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#else
    if(this->normalization_momentum_average == momentum_average_received) { return(true); }

    this->normalization_momentum_average = momentum_average_received;
#endif

    return(true);
}
    
__global__ void kernel__CNeural_Network__Set__Normalization_Epsilon(T_ const epsilon_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Normalization_Epsilon(epsilon_received); }

__host__ __device__ bool CUDA_Neural_Network::Set__Normalization_Epsilon(T_ const epsilon_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CNeural_Network__Set__Normalization_Epsilon <<< 1u, 1u >>> (epsilon_received, this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#else
    if(this->normalization_epsilon == epsilon_received) { return(true); }

    this->normalization_epsilon = epsilon_received;
#endif

    return(true);
}
    
__global__ void kernel__CNeural_Network__Set__Batch_Renormalization_r_Correction_Maximum(T_ const r_correction_maximum_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Batch_Renormalization_r_Correction_Maximum(r_correction_maximum_received); }

__host__ __device__ bool CUDA_Neural_Network::Set__Batch_Renormalization_r_Correction_Maximum(T_ const r_correction_maximum_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CNeural_Network__Set__Batch_Renormalization_r_Correction_Maximum <<< 1u, 1u >>> (r_correction_maximum_received, this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#else
    if(this->batch_renormalization_r_correction_maximum == r_correction_maximum_received) { return(true); }

    this->batch_renormalization_r_correction_maximum = r_correction_maximum_received;
#endif

    return(true);
}
    
__global__ void kernel__CNeural_Network__Set__Batch_Renormalization_d_Correction_Maximum(T_ const d_correction_maximum_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Batch_Renormalization_d_Correction_Maximum(d_correction_maximum_received); }

__host__ __device__ bool CUDA_Neural_Network::Set__Batch_Renormalization_d_Correction_Maximum(T_ const d_correction_maximum_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CNeural_Network__Set__Batch_Renormalization_d_Correction_Maximum <<< 1u, 1u >>> (d_correction_maximum_received, this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#else
    if(this->batch_renormalization_d_correction_maximum == d_correction_maximum_received) { return(true); }

    this->batch_renormalization_d_correction_maximum = d_correction_maximum_received;
#endif

    return(true);
}

__global__ void kernel__CNeural_Network__Set__Regularization__Weight_Decay(T_ const regularization__weight_decay_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Regularization__Weight_Decay(regularization__weight_decay_received); }

__host__ __device__ bool CUDA_Neural_Network::Set__Regularization__Weight_Decay(T_ const regularization__weight_decay_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CNeural_Network__Set__Regularization__Weight_Decay <<< 1u, 1u >>> (regularization__weight_decay_received, this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#else
    if(this->regularization__weight_decay != regularization__weight_decay_received)
    {
        bool const tmp_use_regularization(this->Use__Regularization_Parameter()),
                        tmp_not_initialized_regularization(this->ptr_array_mask_regularized_parameters == nullptr);

        this->regularization__weight_decay = regularization__weight_decay_received;

        if(tmp_use_regularization == false && regularization__weight_decay_received != 0_T)
        {
            if(this->Allocate__Parameter__Regularization() == false)
            {
                PRINT_FORMAT("%s: ERROR: Can not allocate regularization connections!" NEW_LINE, __FUNCTION__);
        
                return(false);
            }

            if(tmp_not_initialized_regularization) { this->Indexing_Regularization_Parameters(); }
        }

        if(this->Use__Regularization_Parameter() == false)
        { this->Deallocate__Parameter__Regularization(); }
    }
#endif

    return(true);
}

__device__ bool CUDA_Neural_Network::Use__Regularization_Parameter(void) const
{
    if(this->regularization__l1 != 0_T
        ||
        this->regularization__l2 != 0_T
        ||
        this->regularization__weight_decay != 0_T)
    { return(true); }
    
    return(false);
}

__device__ void CUDA_Neural_Network::Indexing_Regularization_Parameters(void)
{
    struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer),
                                                    *tmp_ptr_layer_it(this->ptr_array_layers + 1);
    
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED: this->Indexing_Regularization__Weights__FC__Forward(tmp_ptr_layer_it); break;
            //case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: this->Indexing_Regularization__Weights__LSTM(tmp_ptr_layer_it); break;
        }
    }
        
    // Mask all others parameters that is not a weight.
    T_ const *tmp_ptr_last_mask_regularization(this->ptr_array_mask_regularized_parameters + this->total_parameters_allocated);
    T_ *tmp_ptr_mask_regularization_it(this->ptr_array_mask_regularized_parameters + this->total_weights_allocated);
    
    if(this->total_parameters_allocated - this->total_weights_allocated >= warpSize)
    {
        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;
        
        this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(this->total_parameters_allocated - this->total_weights_allocated,
                                                                                                                                                0u,
                                                                                                                                                tmp_dim3_grid,
                                                                                                                                                tmp_dim3_block);

        Zero_1D<T_>(this->total_parameters_allocated - this->total_weights_allocated,
                            tmp_ptr_mask_regularization_it,
                            &tmp_dim3_grid,
                            &tmp_dim3_block);
    }
    else
    {
        for(; tmp_ptr_mask_regularization_it != tmp_ptr_last_mask_regularization; ++tmp_ptr_mask_regularization_it)
        { *tmp_ptr_mask_regularization_it = 0_T; }
    }
    // |END| Mask all others parameters that is not a weight. |END|
}

template<typename T>
__global__ void kernel__CNeural_Network__Indexing_Regularization__Weights__FC(T *const ptr_array_mask_rergularization_parameters_received,
                                                                                                                            size_t const number_connections_received,
                                                                                                                            size_t const *const ptr_array_first_connection_index_received,
                                                                                                                            size_t const *const ptr_array_last_connection_index_received,
                                                                                                                            struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                            struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    Memory::Fill_1D<T_>(number_connections_received,
                                    ptr_array_mask_rergularization_parameters_received + ptr_array_first_connection_index_received[tmp_thread_global_index],
                                    1_T,
                                    ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                    ptr_array_dim3_block_connections_received + tmp_thread_global_index);

    ptr_array_mask_rergularization_parameters_received[ptr_array_last_connection_index_received[tmp_thread_global_index] - 1u] = 0_T; // Bias.
}

template<typename T>
__global__ void kernel__CNeural_Network__Indexing_Regularization__Weights__FC(size_t const size_received,
                                                                                                                            T *const ptr_array_mask_rergularization_parameters_received,
                                                                                                                            size_t const number_connections_received,
                                                                                                                            size_t const *const ptr_array_first_connection_index_received,
                                                                                                                            size_t const *const ptr_array_last_connection_index_received,
                                                                                                                            struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                            struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    if(tmp_thread_global_index < size_received)
    {
        Memory::Fill_1D<T_>(number_connections_received,
                                                             ptr_array_mask_rergularization_parameters_received + ptr_array_first_connection_index_received[tmp_thread_global_index],
                                                             1_T,
                                                             ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                                             ptr_array_dim3_block_connections_received + tmp_thread_global_index);

        ptr_array_mask_rergularization_parameters_received[ptr_array_last_connection_index_received[tmp_thread_global_index] - 1u] = 0_T; // Bias.
    }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Indexing_Regularization__Weights__FC(size_t const size_received,
                                                                                                                                       T *const ptr_array_mask_rergularization_parameters_received,
                                                                                                                                       size_t const number_connections_received,
                                                                                                                                       size_t const *const ptr_array_first_connection_index_received,
                                                                                                                                       size_t const *const ptr_array_last_connection_index_received,
                                                                                                                                       struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                                       struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    do
    {
        Memory::Fill_1D<T_>(number_connections_received,
                                                             ptr_array_mask_rergularization_parameters_received + ptr_array_first_connection_index_received[tmp_thread_global_index],
                                                             1_T,
                                                             ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                                             ptr_array_dim3_block_connections_received + tmp_thread_global_index);

        ptr_array_mask_rergularization_parameters_received[ptr_array_last_connection_index_received[tmp_thread_global_index] - 1u] = 0_T; // Bias.

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Indexing_Regularization__Weights__FC__Forward(struct CUDA_Layer const *const ptr_layer_it_received)
{
    struct CUDA_Neuron const *const tmp_ptr_layer_it_first_neuron(ptr_layer_it_received->ptr_array_neuron_units);

    if(*ptr_layer_it_received->ptr_number_neurons - 1u >= warpSize)
    {
        LAUNCH_KERNEL_POINTER_1D(CNeural_Network__Indexing_Regularization__Weights__FC<T_>,
                                                          ptr_layer_it_received->ptr_dim3_grid_neurons_DP,
                                                          ptr_layer_it_received->ptr_dim3_block_neurons_DP,
                                                          0_zu,
                                                          *ptr_layer_it_received->ptr_number_neurons - 1u,
                                                          this->ptr_array_mask_regularized_parameters,
                                                          *tmp_ptr_layer_it_first_neuron->ptr_number_forward_connections - 1u, // Subtract bias.
                                                          tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                          tmp_ptr_layer_it_first_neuron->ptr_last_forward_connection_index,
                                                          tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                          tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections)
    }
    else
    {
        size_t const *tmp_ptr_array_first_connection_index(tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index),
                                    *const tmp_ptr_array_first_connection_index_last(tmp_ptr_array_first_connection_index + *ptr_layer_it_received->ptr_number_neurons - 1u), // Subtract bias.
                                    *tmp_ptr_array_last_connection_index(tmp_ptr_layer_it_first_neuron->ptr_last_forward_connection_index),
                                    tmp_number_connections(*tmp_ptr_layer_it_first_neuron->ptr_number_forward_connections - 1u); // Subtract bias.

        struct dim3 const *tmp_ptr_array_dim3_grid_connections(tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections),
                                  *tmp_ptr_array_dim3_block_connections(tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);

        for(; tmp_ptr_array_first_connection_index != tmp_ptr_array_first_connection_index_last; ++tmp_ptr_array_first_connection_index,
                                                                                                                                        ++tmp_ptr_array_last_connection_index,
                                                                                                                                        ++tmp_ptr_array_dim3_grid_connections,
                                                                                                                                        ++tmp_ptr_array_dim3_block_connections)
        {
            Memory::Fill_1D<T_>(tmp_number_connections,
                                                                 this->ptr_array_mask_regularized_parameters + *tmp_ptr_array_first_connection_index,
                                                                 1_T,
                                                                 tmp_ptr_array_dim3_grid_connections,
                                                                 tmp_ptr_array_dim3_block_connections);

            this->ptr_array_mask_regularized_parameters[*tmp_ptr_array_last_connection_index - 1u] = 0_T; // Bias.
        }
    }
}

__device__ bool CUDA_Neural_Network::Multi_Class_Classification(void) const
{ return(this->number_outputs > 1u); }

__device__ void CUDA_Neural_Network::Remove_Batch_Normalization(void)
{
    if(this->ptr_array_parameters != nullptr)
    {
        size_t const tmp_new_size(this->total_parameters_allocated - 2u * this->total_neuron_units_allocated);
        
        if(this->Reallocate__Parameter(tmp_new_size) == false)
        {
            PRINT_FORMAT("%s: ERROR: From \"Reallocate__Parameter\"." NEW_LINE, __FUNCTION__);

            return;
        }
    }
}

template<typename T>
__global__ void kernel__CNeural_Network__Reset__Parameters_Neurons_Batch_Normalization(T *const ptr_array_parameters_scale_it_received,
                                                                                                                                                              T *const ptr_array_parameters_shift_it_received,
                                                                                                                                                              struct CUDA_Neuron *const ptr_array_neuron_units_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    ptr_array_neuron_units_received[tmp_thread_global_index].ptr_scale = ptr_array_parameters_scale_it_received + tmp_thread_global_index;
    ptr_array_neuron_units_received[tmp_thread_global_index].ptr_shift = ptr_array_parameters_shift_it_received + tmp_thread_global_index;
}

template<typename T>
__global__ void kernel__CNeural_Network__Reset__Parameters_Neurons_Batch_Normalization(size_t const size_received,
                                                                                                                                                              T *const ptr_array_parameters_scale_it_received,
                                                                                                                                                              T *const ptr_array_parameters_shift_it_received,
                                                                                                                                                              struct CUDA_Neuron *const ptr_array_neuron_units_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_scale = ptr_array_parameters_scale_it_received + tmp_thread_global_index;
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_shift = ptr_array_parameters_shift_it_received + tmp_thread_global_index;
    }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Reset__Parameters_Neurons_Batch_Normalization(size_t const size_received,
                                                                                                                                                                      T *const ptr_array_parameters_scale_it_received,
                                                                                                                                                                      T *const ptr_array_parameters_shift_it_received,
                                                                                                                                                                      struct CUDA_Neuron *const ptr_array_neuron_units_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_scale = ptr_array_parameters_scale_it_received + tmp_thread_global_index;
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_shift = ptr_array_parameters_shift_it_received + tmp_thread_global_index;

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Reset__Parameter__Normalized_Unit(void)
{
    T_ *tmp_ptr_array_parameters_scale_it(this->ptr_array_parameters + this->total_weights_allocated),
        *tmp_ptr_array_parameters_shift_it(this->ptr_array_parameters + this->total_weights_allocated + this->total_neuron_units_allocated);

    struct CUDA_Neuron *tmp_ptr_neuron_unit_it(this->ptr_array_layers->ptr_array_neuron_units);
    struct CUDA_Neuron const *const tmp_ptr_last_neuron_unit(tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated);
    
    if(USE_PARALLEL && this->total_neuron_units_allocated >= warpSize)
    {
        LAUNCH_KERNEL_1D(CNeural_Network__Reset__Parameters_Neurons_Batch_Normalization<T_>,
                                          this->ptr_array_dim3_grid[3u],
                                          this->ptr_array_dim3_block[3u],
                                          0_zu,
                                          this->total_neuron_units_allocated,
                                          tmp_ptr_array_parameters_scale_it,
                                          tmp_ptr_array_parameters_shift_it,
                                          tmp_ptr_neuron_unit_it)

        CUDA__Check_Error();
    }
    else
    {
        for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                ++tmp_ptr_array_parameters_scale_it,
                                                                                ++tmp_ptr_array_parameters_shift_it)
        {
            tmp_ptr_neuron_unit_it->ptr_scale = tmp_ptr_array_parameters_scale_it;
            tmp_ptr_neuron_unit_it->ptr_shift = tmp_ptr_array_parameters_shift_it;
        }
    }
}

template<typename T>
__global__ void kernel__CNeural_Network__Reset__Derivatives_Parameters_Neurons_Batch_Normalization(T *const ptr_array_derivatives_parameters_scale_it_received,
                                                                                                                                                              T *const ptr_array_derivatives_parameters_shift_it_received,
                                                                                                                                                              struct CUDA_Neuron *const ptr_array_neuron_units_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    ptr_array_neuron_units_received[tmp_thread_global_index].ptr_array_derivatives_scales = ptr_array_derivatives_parameters_scale_it_received + tmp_thread_global_index;
    ptr_array_neuron_units_received[tmp_thread_global_index].ptr_array_derivatives_shifts = ptr_array_derivatives_parameters_shift_it_received + tmp_thread_global_index;
}

template<typename T>
__global__ void kernel__CNeural_Network__Reset__Derivatives_Parameters_Neurons_Batch_Normalization(size_t const size_received,
                                                                                                                                                              T *const ptr_array_derivatives_parameters_scale_it_received,
                                                                                                                                                              T *const ptr_array_derivatives_parameters_shift_it_received,
                                                                                                                                                              struct CUDA_Neuron *const ptr_array_neuron_units_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_array_derivatives_scales = ptr_array_derivatives_parameters_scale_it_received + tmp_thread_global_index;
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_array_derivatives_shifts = ptr_array_derivatives_parameters_shift_it_received + tmp_thread_global_index;
    }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Reset__Derivatives_Parameters_Neurons_Batch_Normalization(size_t const size_received,
                                                                                                                                                                      T *const ptr_array_derivatives_parameters_scale_it_received,
                                                                                                                                                                      T *const ptr_array_derivatives_parameters_shift_it_received,
                                                                                                                                                                      struct CUDA_Neuron *const ptr_array_neuron_units_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_array_derivatives_scales = ptr_array_derivatives_parameters_scale_it_received + tmp_thread_global_index;
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_array_derivatives_shifts = ptr_array_derivatives_parameters_shift_it_received + tmp_thread_global_index;

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

__device__ void CUDA_Neural_Network::Reset__Derivative_Parameter__Normalized_Unit(void)
{
    T_ *tmp_ptr_array_derivatives_parameters_scale_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated),
         *tmp_ptr_array_derivatives_parameters_shift_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated + this->total_neuron_units_allocated);

    struct CUDA_Neuron *tmp_ptr_neuron_unit_it(this->ptr_array_layers->ptr_array_neuron_units);
    struct CUDA_Neuron const *const tmp_ptr_last_neuron_unit(tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated);
    
    if(USE_PARALLEL && this->total_neuron_units_allocated >= warpSize)
    {
        LAUNCH_KERNEL_1D(CNeural_Network__Reset__Derivatives_Parameters_Neurons_Batch_Normalization<T_>,
                                          this->ptr_array_dim3_grid[3u],
                                          this->ptr_array_dim3_block[3u],
                                          0_zu,
                                          this->total_neuron_units_allocated,
                                          tmp_ptr_array_derivatives_parameters_scale_it,
                                          tmp_ptr_array_derivatives_parameters_shift_it,
                                          tmp_ptr_neuron_unit_it)

        CUDA__Check_Error();
    }
    else
    {
        for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                             ++tmp_ptr_array_derivatives_parameters_scale_it,
                                                                             ++tmp_ptr_array_derivatives_parameters_shift_it)
        {
            tmp_ptr_neuron_unit_it->ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it;
            tmp_ptr_neuron_unit_it->ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it;
        }
    }
}

__global__ void kernel__CNeural_Network__Get__Limit_Device_Runtime_Pending_Launch_Count(size_t *const ptr_limit_device_runtime_pending_launch_count_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ *ptr_limit_device_runtime_pending_launch_count_received = ptr_CNeural_Network_received->Get__Limit_Device_Runtime_Pending_Launch_Count(); }

__device__ size_t CUDA_Neural_Network::Get__Limit_Device_Runtime_Pending_Launch_Count(void)
{ return(this->limit_device_runtime_pending_launch_count); }

__host__ void CUDA_Neural_Network::Set__Limit_Device_Runtime_Pending_Launch_Count(size_t limit_device_runtime_pending_launch_count_received)
{
    if(limit_device_runtime_pending_launch_count_received == 0u)
    {
        size_t *tmp_ptr_limit_device_runtime_pending_launch_count;

        CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_limit_device_runtime_pending_launch_count, sizeof(size_t)));

        kernel__CNeural_Network__Get__Limit_Device_Runtime_Pending_Launch_Count <<< 1u, 1u >>> (tmp_ptr_limit_device_runtime_pending_launch_count, this);
            
    #if defined(COMPILE_DEBUG)
        CUDA__Check_Error();
    #endif

        CUDA__Safe_Call(cudaMemcpy(&limit_device_runtime_pending_launch_count_received,
                                                        tmp_ptr_limit_device_runtime_pending_launch_count,
                                                        sizeof(size_t),
                                                        cudaMemcpyKind::cudaMemcpyDeviceToHost));
    }

    CUDA__Safe_Call(cudaDeviceSetLimit(cudaLimit::cudaLimitDevRuntimePendingLaunchCount, limit_device_runtime_pending_launch_count_received));
}
    
__global__ void kernel__CNeural_Network__Set__Available_Memory(size_t const available_memory_mbs_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Set__Maximum_Allowable_Memory(available_memory_mbs_received); }

__host__ __device__ void CUDA_Neural_Network::Set__Maximum_Allowable_Memory(size_t const available_memory_mbs_received)
{
#if defined(__CUDA_ARCH__)
    this->maximum_allowable_memory_bytes = available_memory_mbs_received;
#else
    kernel__CNeural_Network__Set__Available_Memory <<< 1u, 1u >>> (available_memory_mbs_received, this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#endif
}

__device__ void CUDA_Neural_Network::Merge_Derivatives_Parameters(void)
{
    Reduce::Reduce_Array(this->number_threads,
                                       this->total_parameters_allocated,
                                       1_zu,
                                       this->ptr_array_derivatives_parameters,
                                       this->ptr_array_dim3_grid_reduce_threads_DP,
                                       this->ptr_array_dim3_block_reduce_threads_DP,
                                       this->ptr_array_dim3_grid + 1,
                                       this->ptr_array_dim3_block + 1);

    Zero_1D<T_>((this->number_threads - 1_zu) * this->total_parameters_allocated,
                         this->ptr_array_derivatives_parameters + this->total_parameters_allocated,
                         this->ptr_array_dim3_grid + 4,
                         this->ptr_array_dim3_block + 4);
}

__Lch_Bds__(MAXIMUM_THREADS_PER_BLOCK, 1)
__global__ void kernel__CNeural_Network__Update_Threads_Size(size_t const number_threads_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    if(ptr_CNeural_Network_received->Update__Thread_Size(number_threads_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Update__Thread_Size\"" NEW_LINE,
                                __FUNCTION__);
    }
}

__device__ void Compute_Minimum_Threads_Block_Requirements(size_t const number_threads_needed_per_example_received,
                                                                                                 size_t const number_grids_launch_needed_per_example_received,
                                                                                                 size_t const minimum_threads_trigger_received,
                                                                                                 size_t &ref_minimum_threads_per_example_received,
                                                                                                 size_t &ref_maximum_grids_launch_per_example_received)
{
    // If number of threads need per example is more that trigger.
    if(number_threads_needed_per_example_received > minimum_threads_trigger_received
        &&
    // Minimum number of threads per example is bigger than the argument.
        ref_minimum_threads_per_example_received > number_threads_needed_per_example_received)
    // Then assign minimum number of threads per example at the argument.
    { ref_minimum_threads_per_example_received = number_threads_needed_per_example_received; }

    // Maximum number of grids launch per example is smaller than the argument.
    if(ref_maximum_grids_launch_per_example_received < number_grids_launch_needed_per_example_received)
    // Then assign maximum number of grids launch per example at the argument.
    { ref_maximum_grids_launch_per_example_received = number_grids_launch_needed_per_example_received; }
}

__host__ __device__ bool CUDA_Neural_Network::Update__Thread_Size(size_t number_threads_received)
{
#if defined(__CUDA_ARCH__)
    if(number_threads_received <= this->cache_number_threads) { return(true); }
    
    size_t const tmp_number_concurrent_kernel(this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Get__Number_Concurrent_Kernel());
    size_t tmp_number_threads,
        /* Minimum threads required per example for processing propagation through
           the neural network in the forward and backward passes.
           For example: The number of neurons in a layer that is parallelizable. */
              tmp_minimum_threads_per_example(std::numeric_limits<size_t>::max()),
              tmp_maximum_grids_launch_per_example(-std::numeric_limits<size_t>::max());
    
    struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer),
                                          *tmp_ptr_layer_it(this->ptr_array_layers);

    struct CUDA_Neuron const *tmp_ptr_neuron_unit_it;

    class CUDA_Device_Information *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

    switch(this->type_network)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_FEEDFORWARD:
            // Loop through each layer.
            for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
            {
                // Store pointer of the first neuron of the dense layer.
                tmp_ptr_neuron_unit_it = tmp_ptr_layer_it->ptr_array_neuron_units;

                // If use neurons parallelisme (Subtract bias.)
                if(*tmp_ptr_layer_it->ptr_number_neurons - 1u >= tmp_ptr_CUDA_Device->Get__Warp_Size())
                {
                    // If use connections parallelisme. (Reduce, FMAC...)
                    if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections >= tmp_ptr_CUDA_Device->Get__Warp_Size())
                    {
                        Compute_Minimum_Threads_Block_Requirements(tmp_ptr_layer_it->ptr_dim3_grid_neurons->x * tmp_ptr_layer_it->ptr_dim3_block_neurons->x,
                                                                                                  tmp_ptr_layer_it->ptr_dim3_grid_neurons->x * tmp_ptr_layer_it->ptr_dim3_block_neurons->x,
                                                                                                  tmp_ptr_CUDA_Device->Get__Warp_Size(),
                                                                                                  tmp_minimum_threads_per_example,
                                                                                                  tmp_maximum_grids_launch_per_example);
                    }
                    else
                    {
                        Compute_Minimum_Threads_Block_Requirements(tmp_ptr_layer_it->ptr_dim3_grid_neurons->x * tmp_ptr_layer_it->ptr_dim3_block_neurons->x,
                                                                                                  1_zu,
                                                                                                  tmp_ptr_CUDA_Device->Get__Warp_Size(),
                                                                                                  tmp_minimum_threads_per_example,
                                                                                                  tmp_maximum_grids_launch_per_example);
                    }
                }
                // If use connections parallelisme. (Reduce, FMAC...)
                else if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections >= tmp_ptr_CUDA_Device->Get__Warp_Size())
                {
                    Compute_Minimum_Threads_Block_Requirements(1_zu,
                                                                                              tmp_ptr_layer_it->ptr_dim3_grid_neurons->x * tmp_ptr_layer_it->ptr_dim3_block_neurons->x,
                                                                                              tmp_ptr_CUDA_Device->Get__Warp_Size(),
                                                                                              tmp_minimum_threads_per_example,
                                                                                              tmp_maximum_grids_launch_per_example);
                }
            }

            if(tmp_minimum_threads_per_example == std::numeric_limits<size_t>::max())
            { tmp_minimum_threads_per_example = 1_zu; }

            if(tmp_maximum_grids_launch_per_example == -std::numeric_limits<size_t>::max())
            { tmp_maximum_grids_launch_per_example = 1_zu; }
                break;
        default:
            PRINT_FORMAT("%s: ERROR: ... with %d as the type network." NEW_LINE,
                                     __FUNCTION__,
                                     this->type_network);
                return(false);
    }
    
    // Divide the total threads by the number of threads needed per exemple.
    tmp_number_threads = static_cast<size_t>(ceil(static_cast<double>(tmp_ptr_CUDA_Device->Get__Maximum_Threads()) / static_cast<double>(tmp_minimum_threads_per_example)));

    // Don't overflow the number of exemple received as argument.
    if(tmp_number_threads > number_threads_received)
    {
        PRINT_FORMAT("%s: WARNING: Can not compute with the optimal number of threads (%zu). Number of threads reduce to %zu. Need more data to compute or a larger neural network!" NEW_LINE,
                                 __FUNCTION__,
                                 tmp_number_threads,
                                 number_threads_received);

        tmp_number_threads = number_threads_received;
    }
    
    size_t tmp_batch_size_allocate(number_threads_received),
                      tmp_number_threads_allocate(number_threads_received);
    
    this->Allouable__Batch_Size(number_threads_received,
                                             tmp_number_threads,
                                             tmp_batch_size_allocate,
                                             tmp_number_threads_allocate);

    if(this->Reallocate__Thread(tmp_number_threads_allocate) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Reallocate__Thread\"." NEW_LINE,
                                 __FUNCTION__);

        return(false);
    }

    this->number_threads = tmp_number_threads_allocate;
    this->cache_number_threads = number_threads_received;
    
    // Asign the new fixed pool size
    this->limit_device_runtime_pending_launch_count = tmp_number_threads_allocate * tmp_maximum_grids_launch_per_example + 1u;

    // number of threads <= batch size.
    if(this->Update__Batch_Size(tmp_number_threads_allocate) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Update__Batch_Size\"" NEW_LINE,
                                 __FUNCTION__);

        return(false);
    }
    
    PRINT_FORMAT("%s: Number of threads desired: %zu" NEW_LINE, __FUNCTION__, number_threads_received);
    PRINT_FORMAT("%s: Number of threads optimal: %zu" NEW_LINE, __FUNCTION__, tmp_number_threads_allocate);
    PRINT_FORMAT("%s: Batch size: %zu" NEW_LINE, __FUNCTION__, this->batch_size);
    PRINT_FORMAT("%s: Minimum number of threads required, per example: %zu" NEW_LINE, __FUNCTION__, tmp_minimum_threads_per_example);
    PRINT_FORMAT("%s: Maximum grid launch required, per example: %zu" NEW_LINE, __FUNCTION__, tmp_maximum_grids_launch_per_example);
    PRINT_FORMAT("%s: Limit device runtime pending launch count (fixed pool size): %zu" NEW_LINE, __FUNCTION__, this->limit_device_runtime_pending_launch_count);
    PRINT_FORMAT("%s: Maximum allowable memory: %zu bytes" NEW_LINE, __FUNCTION__, this->maximum_allowable_memory_bytes);
    PRINT_FORMAT("%s: Total size neural network: %zu bytes" NEW_LINE, __FUNCTION__, this->Get__Sizeof());
#else
    kernel__CNeural_Network__Update_Threads_Size <<< 1u, 1u >>> (number_threads_received, this);
    
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#endif

    return(true);
}

__device__ bool CUDA_Neural_Network::Allouable__Batch_Size(size_t const batch_size_received,
                                                                                    size_t const maximum_threads_received,
                                                                                    size_t &ref_batch_size_allouable_received,
                                                                                    size_t &ref_number_threads_allouable_received)
{
    //this->Update_Available_Memory();

    if(this->number_threads <= 1_zu)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate batch. Calculate the required threads before running this function!" NEW_LINE, __FUNCTION__);

        ref_number_threads_allouable_received = 0_zu;
        ref_batch_size_allouable_received = 0_zu;

        return(false);
    }
    
    // Size of a thread.
    size_t const  tmp_size_thread(this->Get__Threads_Sizeof(1_zu)),
    // Size of a batch.
                       tmp_size_batch(this->Get__Batch_Sizeof(1_zu)),
    // Size of a neural network with no batch.
                       tmp_size_neural_network(this->Get__Sizeof(1_zu, 1_zu) - (tmp_size_thread + tmp_size_batch)),
    // Available memory substraction size of the neural network without batch.
                       tmp_available_memory_mbs(this->maximum_allowable_memory_bytes - tmp_size_neural_network);
    
    PRINT_FORMAT("%s: Maximum allowable memory: %zu bytes" NEW_LINE, __FUNCTION__, this->maximum_allowable_memory_bytes);
    PRINT_FORMAT("%s: Size neural network: %zu bytes" NEW_LINE, __FUNCTION__, tmp_size_neural_network);
    PRINT_FORMAT("%s: Size for one thread: %zu bytes" NEW_LINE, __FUNCTION__, tmp_size_thread);
    PRINT_FORMAT("%s: Size for a batch of size one: %zu bytes" NEW_LINE, __FUNCTION__, tmp_size_batch);
    PRINT_FORMAT("%s: Total size neural network: %zu bytes" NEW_LINE, __FUNCTION__, this->Get__Sizeof());

    // If can not allocate at least one thread, return false.
    if(static_cast<size_t>(tmp_available_memory_mbs / (tmp_size_thread + tmp_size_batch)) == 0_zu)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate threads. More memory need to be available!" NEW_LINE, __FUNCTION__);

        ref_number_threads_allouable_received = 0_zu;
        ref_batch_size_allouable_received = 0_zu;

        return(false);
    }

    size_t tmp_batch_size_allocate(batch_size_received),
              tmp_threads_allocate(1);

    // Do... while allocatables threads meet the maximum threads allocatables.
    do
    {
        // Maximum batch size equal available memory minus allocates threads, then divide by one batch size.
        size_t const tmp_maximum_batch_size_allocatable(static_cast<size_t>((tmp_available_memory_mbs - tmp_threads_allocate * tmp_size_thread) / tmp_size_batch));

        // If threads allocates is greater than batch size.
        if(tmp_threads_allocate > tmp_maximum_batch_size_allocatable)
        {
            PRINT_FORMAT("%s: WARNING: Can not allocate the optimal number of threads (%zu). Number of threads reduce to %zu. More memory need to be available!" NEW_LINE,
                                     __FUNCTION__,
                                     tmp_threads_allocate,
                                     tmp_threads_allocate - 1_zu);

            // Batch size equal available memory minus past allocates threads, then divide by one batch size.
            tmp_batch_size_allocate = static_cast<size_t>((tmp_available_memory_mbs - (tmp_threads_allocate - 1_zu) * tmp_size_thread) / tmp_size_batch);

            break;
        }
        // If batch size is greater than maximum batch size allocatables.
        else if(tmp_batch_size_allocate > tmp_maximum_batch_size_allocatable)
        {
            PRINT_FORMAT("%s: WARNING: Can not allocate the optimal batch size (%zu). Batch size reduce to %zu. More memory need to be available!" NEW_LINE,
                                     __FUNCTION__,
                                     tmp_batch_size_allocate,
                                     tmp_maximum_batch_size_allocatable);

            // Batch size equal maximum batch size allocatables.
            tmp_batch_size_allocate = tmp_maximum_batch_size_allocatable;

            break;
        }
    } while(tmp_threads_allocate++ < maximum_threads_received);
    
    ref_number_threads_allouable_received = tmp_threads_allocate - 1_zu;
    ref_batch_size_allouable_received = tmp_batch_size_allocate;

    return(true);
}

__global__ void kernel__CNeural_Network__Update_Batch_Size(size_t const batch_size_received, class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    if(ptr_CNeural_Network_received->Update__Batch_Size(batch_size_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Update__Batch_Size\"" NEW_LINE,
                                 __FUNCTION__);
    }
}

__host__ __device__ bool CUDA_Neural_Network::Update__Batch_Size(size_t batch_size_received)
{
#if defined(__CUDA_ARCH__)
    if(batch_size_received <= this->cache_batch_size) { return(true); }
    else if(this->number_threads <= 1u) { return(false); }
    
    size_t tmp_batch_size_allocate(batch_size_received),
                      tmp_number_threads_allocate(batch_size_received);
    
    this->Allouable__Batch_Size(batch_size_received,
                                             this->number_threads,
                                             tmp_batch_size_allocate,
                                             tmp_number_threads_allocate);

    // Reallocate batch size with the new batch size meet.
    if(this->Reallocate__Batch(tmp_batch_size_allocate) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Reallocate__Batch\"." NEW_LINE,
                                __FUNCTION__);

        return(false);
    }

    this->batch_size = tmp_batch_size_allocate;
    this->cache_batch_size = batch_size_received;

    PRINT_FORMAT("%s: Batch size: %u" NEW_LINE, __FUNCTION__, this->batch_size);
#else
    kernel__CNeural_Network__Update_Batch_Size <<< 1u, 1u >>> (batch_size_received, this);
    
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#endif

    return(true);
}

// TODO: Optimize algorithm 'Initialize_Candidate_Weights'
__device__ void CUDA_Neural_Network::Initialize_Candidate_Weights(size_t const first_connection_received,
                                                                                            size_t const last_connection_received,
                                                                                            float const scale_factor_received)
{
    PRINT_FORMAT("%s: [FUNCTION DEPRECATED] TODO: Fix \"Initialize_Candidate_Weights\" algorithm." NEW_LINE, __FUNCTION__);

    /*
    size_t tmp_index_bias_weight(static_cast<size_t>(first_connection_received + (this->ptr_array_layers->ptr_last_neuron_unit - this->ptr_array_layers->ptr_array_neuron_units) - 1));
    
    T_ tmp_prev_step(0_T);

    if(this->type_optimizer_function == MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus) { tmp_prev_step = this->rprop_delta_zero; }
    else { tmp_prev_step = 0_T; }
        
    for(size_t i(first_connection_received); i != last_connection_received; ++i)
    {
        this->ptr_array_parameters[i] = static_cast<T_>(curand_uniform(&this->ptr_array_cuRAND_State_MTGP32_weighted[0u])) * (scale_factor_received - -scale_factor_received) + -scale_factor_received;

        if(i != tmp_index_bias_weight) { this->ptr_array_parameters[i] = MyEA::Math::Absolute(this->ptr_array_parameters[i]); }

        this->ptr_array_derivatives_parameters[i] = 0_T;
        this->ptr_array_previous_steps[i] = tmp_prev_step;
        this->ptr_array_previous_derivatives_parameters[i] = 0_T;
    }
    */
}

template<typename T>
__global__ void kernel__CNeural_Network__Randomize_Weights(T const minimum_weight_received,
                                                                                               T const maximum_weight_received,
                                                                                               T *const ptr_array_weights_received,
                                                                                               struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
{ ptr_array_weights_received[blockIdx.x * blockDim.x + threadIdx.x] = static_cast<T>(curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)) * (maximum_weight_received - minimum_weight_received) + minimum_weight_received; }

template<typename T>
__global__ void kernel__CNeural_Network__Randomize_Weights(size_t const size_received,
                                                                                               T const minimum_weight_received,
                                                                                               T const maximum_weight_received,
                                                                                               T *const ptr_array_weights_received,
                                                                                               struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const tmp_curand_uniform(static_cast<T>(curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)));

    if(tmp_thread_global_index < size_received)
    { ptr_array_weights_received[tmp_thread_global_index] = tmp_curand_uniform * (maximum_weight_received - minimum_weight_received) + minimum_weight_received; }
}

template<typename T>
__global__ void kernel_while__CNeural_Network__Randomize_Weights(size_t const size_received,
                                                                                                        T const minimum_weight_received,
                                                                                                        T const maximum_weight_received,
                                                                                                        T *const ptr_array_weights_received,
                                                                                                        struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        ptr_array_weights_received[tmp_thread_global_index] = static_cast<T>(curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)) * (maximum_weight_received - minimum_weight_received) + minimum_weight_received;

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__global__ void kernel__CNeural_Network__Launch_Randomize_Weights(T_ const minimum_weight_received,
                                                                                                           T_ const maximum_weight_received,
                                                                                                           class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Launch_Randomize_Weights(minimum_weight_received, maximum_weight_received); }

__host__ void CUDA_Neural_Network::Launch_Randomize_Weights(T_ const minimum_weight_received, T_ const maximum_weight_received)
{
#if defined(__CUDA_ARCH__)
    LAUNCH_KERNEL_1D(CNeural_Network__Randomize_Weights,
                                        this->ptr_array_dim3_grid[8u],
                                        this->ptr_array_dim3_block[8u],
                                        0_zu,
                                        this->total_weights,
                                        minimum_weight_received,
                                        maximum_weight_received,
                                        this->ptr_array_parameters,
                                        this->ptr_array_cuRAND_State_MTGP32_weighted)

    CUDA__Check_Error();
#else
    kernel__CNeural_Network__Launch_Randomize_Weights <<< 1u, 1u >>> (minimum_weight_received,
                                                                                                                 maximum_weight_received,
                                                                                                                 this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#endif
}

__global__ void kernel__CNeural_Network__Get__Batch_Sizeof(size_t *const size_t_received,
                                                                                          size_t const batch_size_received,
                                                                                          class CUDA_Neural_Network const *const ptr_CNeural_Network_received)
{ *size_t_received = ptr_CNeural_Network_received->Get__Batch_Sizeof(batch_size_received); }

__host__ __device__ size_t CUDA_Neural_Network::Get__Batch_Sizeof(size_t batch_size_received) const
{
#if defined(__CUDA_ARCH__)
    size_t tmp_total_size_t(0_zu);

    if(batch_size_received == 0u) { batch_size_received = this->batch_size; }
    
    // Neurons.
    if(this->ptr_array_neuron_units_summations != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_neuron_units_values != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_values_hats != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_values_normalizes != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_means != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_variances != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_neuron_units_transposed_mean != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_neuron_units_transposed_variance != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_derivatives_means != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_derivatives_variances != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_neuron_units_errors != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->total_neuron_units_allocated * sizeof(T_); }
    
    if(this->ptr_array_2D_neurons_reduce_summation != nullptr && *this->ptr_array_2D_neurons_reduce_summation != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->neurons_total_reduce_summation_size * sizeof(T_); }
    if(this->ptr_array_2D_neurons_reduce_error != nullptr && *this->ptr_array_2D_neurons_reduce_error != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->neurons_total_reduce_error_size * sizeof(T_); }
    if(this->ptr_array_2D_neurons_reduce_batch_mean != nullptr && *this->ptr_array_2D_neurons_reduce_batch_mean != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->neurons_total_reduce_batch_size * sizeof(T_); }
    if(this->ptr_array_2D_neurons_reduce_batch_variance != nullptr && *this->ptr_array_2D_neurons_reduce_batch_variance != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->neurons_total_reduce_batch_size * sizeof(T_); }
    if(this->ptr_array_2D_neurons_reduce_norms != nullptr && *this->ptr_array_2D_neurons_reduce_norms != nullptr) { tmp_total_size_t += batch_size_received * this->number_recurrent_depth * this->neurons_total_reduce_norms_size * sizeof(T_); }

    return(tmp_total_size_t);
#else
    size_t tmp_size_t(0),
              *tmp_ptr_device_size_t(NULL);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_size_t, sizeof(size_t)));

    kernel__CNeural_Network__Get__Batch_Sizeof <<< 1u, 1u >>> (tmp_ptr_device_size_t,
                                                                                                batch_size_received,
                                                                                                this);
    
    CUDA__Safe_Call(cudaMemcpy(&tmp_size_t,
                                                    tmp_ptr_device_size_t,
                                                    sizeof(size_t),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_size_t));

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(tmp_size_t);
#endif
}

__global__ void kernel__CNeural_Network__Get__Threads_Sizeof(size_t *const size_t_received,
                                                                                             size_t const number_threads_received,
                                                                                             class CUDA_Neural_Network const *const ptr_CNeural_Network_received)
{ *size_t_received = ptr_CNeural_Network_received->Get__Threads_Sizeof(number_threads_received); }

__host__ __device__ size_t CUDA_Neural_Network::Get__Threads_Sizeof(size_t number_threads_received) const
{
#if defined(__CUDA_ARCH__)
    size_t tmp_total_size_t(0_zu);

    if(number_threads_received == 0u) { number_threads_received = this->number_threads; }
    
    // Cost.
    if(this->ptr_array_number_loss != nullptr) { tmp_total_size_t += number_threads_received * sizeof(size_t); }
    if(this->ptr_array_number_bit_fail != nullptr) { tmp_total_size_t += number_threads_received * sizeof(size_t); }
    if(this->ptr_array_loss_values != nullptr) { tmp_total_size_t += number_threads_received * sizeof(T_); }
    if(this->ptr_array_accuracy_values[0u] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(T_); }
    if(this->ptr_array_accuracy_values[1u] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(T_); }
    if(this->ptr_array_accuracy_values[2u] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(T_); }
    if(this->ptr_array_accuracy_values[3u] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(T_); }
    if(this->ptr_array_accuracy_values[4u] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(T_); }

    // Parameters.
    if(this->ptr_array_derivatives_parameters != nullptr) { tmp_total_size_t += number_threads_received * this->total_parameters_allocated * sizeof(T_); }

    return(tmp_total_size_t);
#else
    size_t tmp_size_t(0),
              *tmp_ptr_device_size_t(NULL);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_size_t, sizeof(size_t)));

    kernel__CNeural_Network__Get__Threads_Sizeof <<< 1u, 1u >>> (tmp_ptr_device_size_t,
                                                                                                   number_threads_received,
                                                                                                   this);
    
    CUDA__Safe_Call(cudaMemcpy(&tmp_size_t,
                                                    tmp_ptr_device_size_t,
                                                    sizeof(size_t),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_size_t));

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(tmp_size_t);
#endif
}

__global__ void kernel__CNeural_Network__Get__Maximum_Allowable_Memory(size_t *const size_t_received, class CUDA_Neural_Network const *const ptr_CNeural_Network_received)
{ *size_t_received = ptr_CNeural_Network_received->Get__Maximum_Allowable_Memory(); }

__host__ __device__ size_t CUDA_Neural_Network::Get__Maximum_Allowable_Memory(void) const
{
#if defined(__CUDA_ARCH__)
    return(this->maximum_allowable_memory_bytes);
#else
    size_t tmp_size_t(0),
              *tmp_ptr_device_size_t(NULL);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_size_t, sizeof(size_t)));

    kernel__CNeural_Network__Get__Maximum_Allowable_Memory <<< 1u, 1u >>> (tmp_ptr_device_size_t, this);
    
    CUDA__Safe_Call(cudaMemcpy(&tmp_size_t,
                                                    tmp_ptr_device_size_t,
                                                    sizeof(size_t),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_size_t));

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(tmp_size_t);
#endif
}

__global__ void kernel__CNeural_Network__Get__Sizeof(size_t *const size_t_received,
                                                                                size_t const number_threads_received,
                                                                                size_t const batch_size_received,
                                                                                class CUDA_Neural_Network const *const ptr_CNeural_Network_received)
{ *size_t_received = ptr_CNeural_Network_received->Get__Sizeof(number_threads_received, batch_size_received); }

__host__ __device__ size_t CUDA_Neural_Network::Get__Sizeof(size_t number_threads_received, size_t batch_size_received) const
{
#if defined(__CUDA_ARCH__)
    size_t tmp_total_size_t(0);

    tmp_total_size_t += sizeof(class CUDA_Neural_Network); // this
    
    tmp_total_size_t += this->Get__Threads_Sizeof(number_threads_received == 0u ? this->number_threads : number_threads_received);
    
    tmp_total_size_t += this->Get__Batch_Sizeof(batch_size_received == 0u ? this->batch_size : batch_size_received);

    //tmp_total_size_t += X * sizeof(struct struct_Block_Parameters); // ptr_array_block_parameters
    
    // Dim3.
    if(this->ptr_array_dim3_grid != NULL) { tmp_total_size_t += TOTAL_KERNEL_PARALLEL * sizeof(struct dim3); }
    if(this->ptr_array_dim3_block != NULL) { tmp_total_size_t += TOTAL_KERNEL_PARALLEL * sizeof(struct dim3); }
    
    if(this->ptr_array_dim3_grid_reduce_threads != NULL) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(struct dim3); }
    if(this->ptr_array_dim3_block_reduce_threads != NULL) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(struct dim3); }
    
    if(this->ptr_array_dim3_grid_reduce_threads_DP != NULL) { tmp_total_size_t += this->total_reduce_batch_DP_size * sizeof(struct dim3); }
    if(this->ptr_array_dim3_block_reduce_threads_DP != NULL) { tmp_total_size_t += this->total_reduce_batch_DP_size * sizeof(struct dim3); }
    
    if(this->ptr_array_layers_dim3_grid_neurons != NULL) { tmp_total_size_t += this->total_layers * sizeof(struct dim3); }
    if(this->ptr_array_layers_dim3_block_neurons != NULL) { tmp_total_size_t += this->total_layers * sizeof(struct dim3); }
    
    if(this->ptr_array_layers_dim3_grid_weights != NULL) { tmp_total_size_t += this->total_layers * sizeof(struct dim3); }
    if(this->ptr_array_layers_dim3_block_weights != NULL) { tmp_total_size_t += this->total_layers * sizeof(struct dim3); }
    
    if(this->ptr_array_neuron_units_dim3_grid_connections != NULL) { tmp_total_size_t += this->total_neuron_units * sizeof(struct dim3); }
    if(this->ptr_array_neuron_units_dim3_block_connections != NULL) { tmp_total_size_t += this->total_neuron_units * sizeof(struct dim3); }

    if(this->ptr_array_neuron_units_dim3_grid_reduce_summation != NULL) { tmp_total_size_t += this->neurons_total_reduce_summation_size * sizeof(struct dim3); }
    if(this->ptr_array_neuron_units_dim3_block_reduce_summation != NULL) { tmp_total_size_t += this->neurons_total_reduce_summation_size * sizeof(struct dim3); }
    
    if(this->ptr_array_neuron_units_dim3_grid_reduce_error != NULL) { tmp_total_size_t += this->neurons_total_reduce_error_size * sizeof(struct dim3); }
    if(this->ptr_array_neuron_units_dim3_block_reduce_error != NULL) { tmp_total_size_t += this->neurons_total_reduce_error_size * sizeof(struct dim3); }
    
    if(this->ptr_array_neuron_units_dim3_grid_reduce_batch != NULL) { tmp_total_size_t += this->neurons_total_reduce_batch_size * sizeof(struct dim3); }
    if(this->ptr_array_neuron_units_dim3_block_reduce_batch != NULL) { tmp_total_size_t += this->neurons_total_reduce_batch_size * sizeof(struct dim3); }
    
    if(this->ptr_array_2D_neurons_dim3_grid_reduce_norms != NULL)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(struct dim3*);

        if(this->ptr_array_2D_neurons_dim3_grid_reduce_norms[0u] != NULL) { tmp_total_size_t += this->neurons_total_reduce_norms_size * sizeof(struct dim3); }
    }

    if(this->ptr_array_2D_neurons_dim3_block_reduce_norms != NULL)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(struct dim3*);

        if(this->ptr_array_2D_neurons_dim3_block_reduce_norms[0u] != NULL) { tmp_total_size_t += this->neurons_total_reduce_norms_size * sizeof(struct dim3); }
    }
    // |END| Dim3. |END|
    
    // Cost reduce.
    if(this->ptr_array_reduce_number_loss != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(size_t); }
    if(this->ptr_array_reduce_bit_fail_values != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(size_t); }
    if(this->ptr_array_reduce_loss_values != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(T_); }
    if(this->ptr_array_reduce_accuracy_values[0u] != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(T_); }
    if(this->ptr_array_reduce_accuracy_values[1u] != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(T_); }
    if(this->ptr_array_reduce_accuracy_values[2u] != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(T_); }
    if(this->ptr_array_reduce_accuracy_values[3u] != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(T_); }
    if(this->ptr_array_reduce_accuracy_values[4u] != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(T_); }

    // Parameters.
    if(this->ptr_array_ptr_connections != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(void*); }

    if(this->ptr_array_transposed_weights != nullptr) { tmp_total_size_t += this->total_weights_allocated * sizeof(T_); }
    if(this->ptr_array_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    if(this->ptr_array_mask_regularized_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }

    //    Optimizer iRPROP.
    if(this->ptr_array_previous_steps != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    if(this->ptr_array_previous_delta_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    if(this->ptr_array_previous_derivatives_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    //    |END| Optimizer iRPROP. |END|

    //    Optimizer Adam.
    if(this->ptr_array_previous_biased_first_moment != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    if(this->ptr_array_previous_biased_second_moment != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    //    |END| Optimizer Adam. |END|

    //    Optimizer AMSGrad.
    if(this->ptr_array_previous_biased_second_moment_hat != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    //    |END| Optimizer AMSGrad. |END|
    // |END| Parameters. |END|
    
    // Dropout variable.
    if(this->ptr_array_af_units_mask_dropout_bernoulli != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(bool); }
    if(this->ptr_array_cell_units_mask_dropout_zoneout != nullptr) { tmp_total_size_t += 2_zu * this->total_cell_units_allocated * sizeof(bool); }
    if(this->ptr_array_mask_dropout_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(T_); }
    // |END| Dropout variable. |END|
    
    // TODO: Create into CUDA_Device_Information_Array a function returning sizeof called Get__Sizeof().
    if(this->_ptr_Class_Device_Information_Array != nullptr)
    {
        tmp_total_size_t += sizeof(class CUDA_Device_Information_Array);

        if(this->_ptr_Class_Device_Information_Array->Get__Number_CUDA_Devices() != 0u)
        {
            tmp_total_size_t += sizeof(class CUDA_Device_Information); // _ptr_Class_Device_Information_sum
            tmp_total_size_t += sizeof(class CUDA_Device_Information); // _ptr_Class_Device_Information_higher
            tmp_total_size_t += sizeof(class CUDA_Device_Information); // _ptr_Class_Device_Information_lower
            tmp_total_size_t += this->_ptr_Class_Device_Information_Array->Get__Number_CUDA_Devices() * sizeof(class CUDA_Device_Information); // _ptr_array_Class_Device_Information
        }
    }

    // Layers.
    if(this->ptr_array_number_neurons_by_layer != nullptr) { tmp_total_size_t += this->total_layers * sizeof(size_t); }

    if(this->ptr_array_layers != nullptr)
    {
        tmp_total_size_t += this->total_layers * sizeof(struct CUDA_Layer);

        if(this->ptr_array_layers->ptr_array_neuron_units != nullptr)
        { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(struct CUDA_Neuron); }
    }

    if(this->ptr_array_layers_Class_Storage_Dim3_Batch != nullptr) { tmp_total_size_t += this->total_layers * sizeof(class CUDA_Storage_Dim3); }
    // |END| Layers. |END|

    // Neurons.
    if(this->ptr_array_neuron_units_first_forward_connection_index != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuron_units_last_forward_connection_index != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuron_units_number_forward_connections != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuron_units_reduce_summation_size != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuron_units_reduce_batch_size != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuron_units_reduce_norms_size != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuroyed_number_neurons_in_layer != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }

    if(this->ptr_array_neuron_units_activation_steepness != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_r_corrections != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_d_corrections != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_means_averages != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(T_); }
    if(this->ptr_array_normalized_batch_units_variances_averages != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(T_); }

    if(this->ptr_array_2D_neurons_reduce_summation != nullptr)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(T_*);
            
        if(*this->ptr_array_2D_neurons_reduce_summation != nullptr)
        { tmp_total_size_t += this->neurons_total_reduce_summation_size * sizeof(T_); }
    }
    
    if(this->ptr_array_2D_neurons_reduce_error != nullptr)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(T_*);
            
        if(*this->ptr_array_2D_neurons_reduce_error != nullptr)
        { tmp_total_size_t += this->neurons_total_reduce_error_size * sizeof(T_); }
    }
    
    if(this->ptr_array_2D_neurons_reduce_batch_mean != nullptr)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(T_*);
            
        if(*this->ptr_array_2D_neurons_reduce_batch_mean != nullptr)
        { tmp_total_size_t += this->neurons_total_reduce_batch_size * sizeof(T_); }
    }
    
    if(this->ptr_array_2D_neurons_reduce_batch_variance != nullptr)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(T_*);
            
        if(*this->ptr_array_2D_neurons_reduce_batch_variance != nullptr)
        { tmp_total_size_t += this->neurons_total_reduce_batch_size * sizeof(T_); }
    }
    
    if(this->ptr_array_2D_neurons_reduce_norms != nullptr)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(T_*);
            
        if(*this->ptr_array_2D_neurons_reduce_norms != nullptr)
        { tmp_total_size_t += this->neurons_total_reduce_norms_size * sizeof(T_); }
    }
    // |END| Neurons. |END|
        
    // cuRAND.
    if(this->ptr_array_cuRAND_State_MTGP32_weighted != nullptr)
    {
        tmp_total_size_t += this->number_cuRAND_State_MTGP32_weighted * sizeof(struct curandStateMtgp32);
        tmp_total_size_t += this->number_cuRAND_State_MTGP32_weighted * sizeof(mtgp32_kernel_params_t);
    }
    
    if(this->ptr_array_cuRAND_State_MTGP32_neuroyed != nullptr)
    {
        tmp_total_size_t += this->number_cuRAND_State_MTGP32_neuroyed * sizeof(struct curandStateMtgp32);
        tmp_total_size_t += this->number_cuRAND_State_MTGP32_neuroyed * sizeof(mtgp32_kernel_params_t);
    }
    // |END| cuRAND. |END|

    return(tmp_total_size_t);
#else
    size_t tmp_size_t(0),
              *tmp_ptr_device_size_t(NULL);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_size_t, sizeof(size_t)));

    kernel__CNeural_Network__Get__Sizeof <<< 1u, 1u >>> (tmp_ptr_device_size_t,
                                                                                      number_threads_received,
                                                                                      batch_size_received,
                                                                                      this);
    
    CUDA__Safe_Call(cudaMemcpy(&tmp_size_t,
                                                    tmp_ptr_device_size_t,
                                                    sizeof(size_t),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_size_t));

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(tmp_size_t);
#endif
}

__device__ void CUDA_Neural_Network::Printf_Parameters(bool const full_description_received)
{
    // TODO: Printf_Parameters.
    PRINT_FORMAT("Input layer : %u neuson(s), 1 bias." NEW_LINE, this->number_inputs);

    PRINT_FORMAT("Output layer : %u neuron(s)." NEW_LINE, this->number_outputs);
}
