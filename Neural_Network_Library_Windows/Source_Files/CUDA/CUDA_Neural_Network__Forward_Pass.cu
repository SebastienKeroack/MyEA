#include <Tools/CUDA_Configuration.cuh>
#include <Math/CUDA_Mathematic.cuh>
#include <Tools/CUDA_Zero_1D.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

__device__ void CUDA_Neural_Network::Forward_Pass(size_t const batch_size_received, T_ const *const *const ptr_array_inputs_received) { this->FF__Forward_Pass_Batch(batch_size_received, ptr_array_inputs_received); }

__device__ void CUDA_Neural_Network::FF__Forward_Pass_Batch(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received)
{
    // By default the synchronized state is set to true.
    bool tmp_synchronized(true);
    
    struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct CUDA_Layer *tmp_ptr_previous_layer_it(this->ptr_array_layers),
                                            *tmp_ptr_layer_it(tmp_ptr_previous_layer_it + 1);
    
    // Variable to cache optimal size to launch dynamic parallelisme through the GPU.
    struct dim3 tmp_dim3_grid,
                     tmp_dim3_block;

#if defined(COMPILE_DEBUG)
    if(batch_size_received > this->batch_size)
    {
        PRINT_FORMAT("%s: ERROR: Batch size (%u) > number threads (%u)." NEW_LINE,
                                    __FUNCTION__,
                                    batch_size_received,
                                    this->batch_size);

        return;
    }
#endif

    // Input layer.
    this->Assign_Inputs_Batch(tmp_synchronized,
                                            batch_size_received,
                                            ptr_matrix_inputs_received);
    // |END| Input layer. |END|
    
    // If we can go into dynamic parallelisme, prepare the dimension kernel.
    if(batch_size_received >= warpSize)
    {
        size_t const tmp_batch_size_scale(MyEA::Math::Minimum<size_t>(batch_size_received, this->number_threads));

        if(tmp_batch_size_scale == this->number_threads)
        {
            tmp_dim3_grid = this->ptr_array_dim3_grid[7u];
            tmp_dim3_block = this->ptr_array_dim3_block[7u];
        }
        else
        {
            this->ptr_array_layers->ptr_Class_Storage_Dim3_Batch->Get__Dim3_Dynamic_Parallelisme(tmp_batch_size_scale,
                                                                                                                                                tmp_dim3_grid,
                                                                                                                                                tmp_dim3_block,
                                                                                                                                                this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
        }
    }
    
    // If the network use batch renormalization.
    if(this->use_Batch_Renormalization)
    {
        // Set all mean to zero.
        Zero_1D<T_>(this->batch_size * this->total_neuron_units_allocated,
                            this->ptr_array_normalized_batch_units_means,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Set all mean to zero. |END|

        // Set all variance to zero.
        Zero_1D<T_>(this->batch_size * this->total_neuron_units_allocated,
                            this->ptr_array_normalized_batch_units_variances,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Set all variance to zero. |END|
        
        // Do we need to synchronise? Based on "Zero_1D" Function.
        // => Synchronisation before using the mean and variance of the network.
        if(this->batch_size * this->total_neuron_units_allocated >= warpSize) { tmp_synchronized = false; }

        if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
        {
            // If the network use dropout.
            if(this->use_Dropout)
            {
                // Loop from the second layer to the last layer.
                for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                                ++tmp_ptr_previous_layer_it)
                {
                    // If the layer use batch normalization/renormalization
                    if(tmp_ptr_layer_it->use_Batch_Renormalization)
                    {
                        this->Forward_Pass__FC_to__Batch_Renormalization__Dropout(tmp_synchronized,
                                                                                                                                        batch_size_received,
                                                                                                                                        tmp_ptr_layer_it,
                                                                                                                                        tmp_ptr_previous_layer_it,
                                                                                                                                        &tmp_dim3_grid,
                                                                                                                                        &tmp_dim3_block);
                    }
                    // Else propagate through dropout layer.
                    else
                    {
                        // Forward propagation through a layer.
                        // With dropout regulariation. At the training state.
                        this->Forward_Pass__FC_to__Dropout(tmp_synchronized,
                                                                                                    batch_size_received,
                                                                                                    tmp_ptr_layer_it,
                                                                                                    tmp_ptr_previous_layer_it,
                                                                                                    &tmp_dim3_grid,
                                                                                                    &tmp_dim3_block);
                    }
                }
            }
            else
            {
                // Loop from the second layer to the last layer.
                for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                                ++tmp_ptr_previous_layer_it)
                {
                    // If the layer use batch normalization/renormalization
                    if(tmp_ptr_layer_it->use_Batch_Renormalization)
                    {
                        this->Forward_Pass__FC_to__Batch_Renormalization__Training(tmp_synchronized,
                                                                                                                         batch_size_received,
                                                                                                                         tmp_ptr_layer_it,
                                                                                                                         tmp_ptr_previous_layer_it,
                                                                                                                         &tmp_dim3_grid,
                                                                                                                         &tmp_dim3_block);
                    }
                    // Else propagate through default layer.
                    else
                    {
                        // Forward propagation through a layer.
                        this->Forward_Pass__FC_to(tmp_synchronized,
                                                                        batch_size_received,
                                                                        tmp_ptr_layer_it,
                                                                        tmp_ptr_previous_layer_it,
                                                                        &tmp_dim3_grid,
                                                                        &tmp_dim3_block);
                    }
                }
            }
        }
        else
        {
            // If the network use dropout.
            if(this->use_Dropout)
            {
                // Loop from the second layer to the last layer.
                for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                                ++tmp_ptr_previous_layer_it)
                {
                    // If the layer use batch normalization/renormalization
                    if(tmp_ptr_layer_it->use_Batch_Renormalization)
                    {
                        this->Forward_Pass__FC_to__Batch_Renormalization__Dropout_Bernoulli__Testing(tmp_synchronized,
                                                                                                                                        batch_size_received,
                                                                                                                                        tmp_ptr_layer_it,
                                                                                                                                        tmp_ptr_previous_layer_it,
                                                                                                                                        &tmp_dim3_grid,
                                                                                                                                        &tmp_dim3_block);
                    }
                    // Else propagate through dropout layer.
                    else
                    {
                        // Forward propagation through a layer.
                        // With dropout regulariation. At the testing state.
                        this->Forward_Pass__FC_to__Dropout_Bernoulli__Testing(tmp_synchronized,
                                                                                                    batch_size_received,
                                                                                                    tmp_ptr_layer_it,
                                                                                                    tmp_ptr_previous_layer_it,
                                                                                                    &tmp_dim3_grid,
                                                                                                    &tmp_dim3_block);
                    }
                }
            }
            else
            {
                // Loop from the second layer to the last layer.
                for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                                ++tmp_ptr_previous_layer_it)
                {
                    // If the layer use batch normalization/renormalization
                    if(tmp_ptr_layer_it->use_Batch_Renormalization)
                    {
                        this->Forward_Pass__FC_to__Batch_Renormalization__Loop(tmp_synchronized,
                                                                                                                         batch_size_received,
                                                                                                                         tmp_ptr_layer_it,
                                                                                                                         tmp_ptr_previous_layer_it,
                                                                                                                         &tmp_dim3_grid,
                                                                                                                         &tmp_dim3_block);
                    }
                    // Else propagate through default layer.
                    else
                    {
                        // Forward propagation through a layer.
                        this->Forward_Pass__FC_to(tmp_synchronized,
                                                                        batch_size_received,
                                                                        tmp_ptr_layer_it,
                                                                        tmp_ptr_previous_layer_it,
                                                                        &tmp_dim3_grid,
                                                                        &tmp_dim3_block);
                    }
                }
            }
        }
    }
    else
    {
        // If the network use dropout.
        if(this->use_Dropout)
        {
            if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
            {
                // Loop from the second layer to the last layer.
                for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                                ++tmp_ptr_previous_layer_it)
                {
                    // Forward propagation through a layer.
                    // With dropout regulariation. At the training state.
                    this->Forward_Pass__FC_to__Dropout(tmp_synchronized,
                                                                                                batch_size_received,
                                                                                                tmp_ptr_layer_it,
                                                                                                tmp_ptr_previous_layer_it,
                                                                                                &tmp_dim3_grid,
                                                                                                &tmp_dim3_block);
                }
            }
            else
            {
                // Loop from the second layer to the last layer.
                for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                                ++tmp_ptr_previous_layer_it)
                {
                    // Forward propagation through a layer.
                    // With dropout regulariation. At the testing state.
                    this->Forward_Pass__FC_to__Dropout_Bernoulli__Testing(tmp_synchronized,
                                                                                                batch_size_received,
                                                                                                tmp_ptr_layer_it,
                                                                                                tmp_ptr_previous_layer_it,
                                                                                                &tmp_dim3_grid,
                                                                                                &tmp_dim3_block);
                }
            }
        }
        else
        {
            // Loop from the second layer to the last layer.
            for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                            ++tmp_ptr_previous_layer_it)
            {
                // Forward propagation through a layer.
                this->Forward_Pass__FC_to(tmp_synchronized,
                                                                batch_size_received,
                                                                tmp_ptr_layer_it,
                                                                tmp_ptr_previous_layer_it,
                                                                &tmp_dim3_grid,
                                                                &tmp_dim3_block);
            }
        }
    }
    
    // Synchronisation before using the output of the neural nework.
    CUDA__Device_Synchronise(tmp_synchronized, MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);
}

__device__ void CUDA_Neural_Network::Forward_Pass__FC_to(bool &ref_synchronized_received,
                                                                                         size_t const batch_size_received,
                                                                                         struct CUDA_Layer *const ptr_layer_it_received,
                                                                                         struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                         struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                         struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(ptr_layer_it_received->type_activation)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC(ref_synchronized_received,
                                                                      batch_size_received,
                                                                      ptr_layer_it_received,
                                                                      ptr_previous_layer_it_received,
                                                                      ptr_dim3_batch_size_grid_received,
                                                                      ptr_dim3_batch_size_block_received);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX:
            this->Forward_Pass__FC_to_FC__Softmax(ref_synchronized_received,
                                                                                    batch_size_received,
                                                                                    ptr_layer_it_received,
                                                                                    ptr_previous_layer_it_received,
                                                                                    ptr_dim3_batch_size_grid_received,
                                                                                    ptr_dim3_batch_size_block_received);
                break;
        default:
            PRINT_FORMAT("%s: ERROR: Can not propagate forward with (%u) as the type activation." NEW_LINE,
                                    __FUNCTION__,
                                    ptr_layer_it_received->type_activation);
                break;
    }
}

__device__ void CUDA_Neural_Network::Forward_Pass__FC_to__Dropout_Bernoulli__Testing(bool &ref_synchronized_received,
                                                                                                                    size_t const batch_size_received,
                                                                                                                    struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                    struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                    struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                    struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(ptr_layer_it_received->type_activation)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing(ref_synchronized_received,
                                                                                                 batch_size_received,
                                                                                                 ptr_layer_it_received,
                                                                                                 ptr_previous_layer_it_received,
                                                                                                 ptr_dim3_batch_size_grid_received,
                                                                                                 ptr_dim3_batch_size_block_received);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX:
            this->Forward_Pass__FC_to_FC__Softmax(ref_synchronized_received,
                                                                                    batch_size_received,
                                                                                    ptr_layer_it_received,
                                                                                    ptr_previous_layer_it_received,
                                                                                    ptr_dim3_batch_size_grid_received,
                                                                                    ptr_dim3_batch_size_block_received);
                break;
        default:
            PRINT_FORMAT("%s: ERROR: Can not propagate forward with (%u) as the type activation." NEW_LINE,
                                    __FUNCTION__,
                                    ptr_layer_it_received->type_activation);
                break;
    }
}
    
__device__ void CUDA_Neural_Network::Forward_Pass__FC_to__Dropout(bool &ref_synchronized_received,
                                                                                                                     size_t const batch_size_received,
                                                                                                                     struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                     struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                     struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                     struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(ptr_layer_it_received->type_activation)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC__Dropout(ref_synchronized_received,
                                                                                                  batch_size_received,
                                                                                                  ptr_layer_it_received,
                                                                                                  ptr_previous_layer_it_received,
                                                                                                  ptr_dim3_batch_size_grid_received,
                                                                                                  ptr_dim3_batch_size_block_received);
                break;
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX:
            this->Forward_Pass__FC_to_FC__Softmax(ref_synchronized_received,
                                                                                    batch_size_received,
                                                                                    ptr_layer_it_received,
                                                                                    ptr_previous_layer_it_received,
                                                                                    ptr_dim3_batch_size_grid_received,
                                                                                    ptr_dim3_batch_size_block_received);
                break;
        default:
            PRINT_FORMAT("%s: ERROR: Can not propagate forward with (%u) as the type activation." NEW_LINE,
                                    __FUNCTION__,
                                    ptr_layer_it_received->type_activation);
                break;
    }
}

__device__ void CUDA_Neural_Network::Forward_Pass__FC_to__Batch_Renormalization__Loop(bool &ref_synchronized_received,
                                                                                                                                          size_t const batch_size_received,
                                                                                                                                          struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                          struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                                          struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                          struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(ptr_layer_it_received->type_activation)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC__Batch_Renormalization__Loop(ref_synchronized_received,
                                                                                                                        batch_size_received,
                                                                                                                        ptr_layer_it_received,
                                                                                                                        ptr_previous_layer_it_received,
                                                                                                                        ptr_dim3_batch_size_grid_received,
                                                                                                                        ptr_dim3_batch_size_block_received);
            break;
        default:
            PRINT_FORMAT("%s: ERROR: Can not propagate forward with (%u) as the type activation." NEW_LINE,
                                    __FUNCTION__,
                                    ptr_layer_it_received->type_activation);
                break;
    }
}

__device__ void CUDA_Neural_Network::Forward_Pass__FC_to__Batch_Renormalization__Dropout_Bernoulli__Testing(bool &ref_synchronized_received,
                                                                                                                                                        size_t const batch_size_received,
                                                                                                                                                        struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                                        struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(ptr_layer_it_received->type_activation)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing(ref_synchronized_received,
                                                                                                                                    batch_size_received,
                                                                                                                                    ptr_layer_it_received,
                                                                                                                                    ptr_previous_layer_it_received,
                                                                                                                                    ptr_dim3_batch_size_grid_received,
                                                                                                                                    ptr_dim3_batch_size_block_received);
            break;
        default:
            PRINT_FORMAT("%s: ERROR: Can not propagate forward with (%u) as the type activation." NEW_LINE,
                                    __FUNCTION__,
                                    ptr_layer_it_received->type_activation);
                break;
    }
}

__device__ void CUDA_Neural_Network::Forward_Pass__FC_to__Batch_Renormalization__Training(bool &ref_synchronized_received,
                                                                                                                                            size_t const batch_size_received,
                                                                                                                                            struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                            struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                                            struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                            struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(ptr_layer_it_received->type_activation)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC__Batch_Renormalization__Training(ref_synchronized_received,
                                                                                                                        batch_size_received,
                                                                                                                        ptr_layer_it_received,
                                                                                                                        ptr_previous_layer_it_received,
                                                                                                                        ptr_dim3_batch_size_grid_received,
                                                                                                                        ptr_dim3_batch_size_block_received);
                break;
        default:
            PRINT_FORMAT("%s: ERROR: Can not propagate forward with (%u) as the type activation." NEW_LINE,
                        __FUNCTION__,
                        ptr_layer_it_received->type_activation);
                break;
    }
}

__device__ void CUDA_Neural_Network::Forward_Pass__FC_to__Batch_Renormalization__Dropout(bool &ref_synchronized_received,
                                                                                                                                                        size_t const batch_size_received,
                                                                                                                                                        struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                                        struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(ptr_layer_it_received->type_activation)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout(ref_synchronized_received,
                                                                                                                                    batch_size_received,
                                                                                                                                    ptr_layer_it_received,
                                                                                                                                    ptr_previous_layer_it_received,
                                                                                                                                    ptr_dim3_batch_size_grid_received,
                                                                                                                                    ptr_dim3_batch_size_block_received);
                break;
        default:
            PRINT_FORMAT("%s: ERROR: Can not propagate forward with (%u) as the type activation." NEW_LINE,
                        __FUNCTION__,
                        ptr_layer_it_received->type_activation);
                break;
    }
}
