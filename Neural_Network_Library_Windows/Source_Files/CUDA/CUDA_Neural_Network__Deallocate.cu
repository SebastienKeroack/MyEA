#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

#include <curand_kernel.h>

__global__ void kernel__CNeural_Network__Deallocate(class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ ptr_CNeural_Network_received->Deallocate(); }

__host__ __device__ bool CUDA_Neural_Network::Deallocate(void)
{
#if defined(__CUDA_ARCH__)
    // Layer variable.
    SAFE_DELETE_ARRAY(this->ptr_array_number_neurons_by_layer); // size_t

    if(this->ptr_array_layers != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_layers->ptr_array_neuron_units);

        delete[](this->ptr_array_layers);
        this->ptr_array_layers = nullptr;
    }

    SAFE_DELETE_ARRAY(this->ptr_array_layers_Class_Storage_Dim3_Batch);
    // |END| Layer variable. |END|

    // Delete neurons variable.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_first_forward_connection_index); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_last_forward_connection_index); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_number_forward_connections); // delete[] array size_t.

    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_summations); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_activation_steepness); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_values); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_values_hats); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_values_normalizes); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_means); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_variances); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_transposed_mean); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_transposed_variance); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_derivatives_means); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_derivatives_variances); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_r_corrections); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_d_corrections); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_means_averages); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_variances_averages); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_errors); // delete[] array T_.
        
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_type_activation_function); // delete[] array enum.

    this->Deallocate__Neuron__Mask_Dropout_Bernoulli();
    
    this->Deallocate__Cell_Unit__Mask_Dropout_Zoneout();

    this->Deallocate__Neurons_Reduce_Summation();
    this->Deallocate__Neurons_Reduce_Error();
    this->Deallocate__Neurons_Reduce_Norms();

    this->Deallocate_Batch_Reduce();
    this->Deallocate__Normalized_Unit__Batch_Normalization();
    // |END| Delete neurons variable. |END|

    // Delete connections.
    SAFE_DELETE_ARRAY(this->ptr_array_transposed_weights); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_parameters); // delete[] array T_.

    SAFE_DELETE_ARRAY(this->ptr_array_ptr_connections); // delete[] array void*.
        
    this->Deallocate__Parameter__Regularization();
    
    SAFE_DELETE_ARRAY(this->ptr_array_derivatives_parameters); // delete[] array T_.
    // |END| Delete connections. |END|

    // Deallocate optimizer array.
    this->Deallocate__Parameter__Optimizer();
    // |END| Deallocate optimizer array. |END|

    // Deallocate cost.
    this->Deallocate_Cost();
    this->Deallocate_Reduce_Batch();
    this->Deallocate_Reduce_Cost();
    // |END| Deallocate cost. |END|

    // Delete cuRAND.
    if(this->ptr_array_cuRAND_State_MTGP32_weighted != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_cuRAND_State_MTGP32_weighted[0u].k);

        delete[](this->ptr_array_cuRAND_State_MTGP32_weighted);
    }
    
    if(this->ptr_array_cuRAND_State_MTGP32_neuroyed != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_cuRAND_State_MTGP32_neuroyed[0u].k);

        delete[](this->ptr_array_cuRAND_State_MTGP32_neuroyed);
    }
    // |END| Delete cuRAND |END|
        
    // Struct dim3 variable.
    SAFE_FREE(this->ptr_array_dim3_grid); // struct dim3.
    SAFE_FREE(this->ptr_array_dim3_block); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_grid_neurons); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_block_neurons); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_grid_neurons_DP); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_block_neurons_DP); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_grid_neurons_cuRAND); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_block_neurons_cuRAND); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_grid_batch_neurons); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_block_batch_neurons); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_grid_weights); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_block_weights); // struct dim3.
    SAFE_FREE(this->ptr_array_neuron_units_dim3_grid_connections); // struct dim3.
    SAFE_FREE(this->ptr_array_neuron_units_dim3_block_connections); // struct dim3.

    SAFE_DELETE(this->ptr_Class_Storage_Dim3_Memcpy);
    // |END| Struct dim3 variable. |END|

    // Delete computation parameters.
    SAFE_DELETE(this->_ptr_Class_Device_Information_Array);
    // |END| Delete computation parameters |END|
#else
    kernel__CNeural_Network__Deallocate <<< 1u, 1u >>> (this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#endif

    return(true);
}
    
__device__ void CUDA_Neural_Network::Deallocate__Parameter__Optimizer(void)
{
    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD: this->Deallocate__Parameter__Gradient_Descent(); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus: this->Deallocate__Parameter__iRPROP_minus(); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus: this->Deallocate__Parameter__iRPROP_plus(); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: this->Deallocate__Parameter__Adam(); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: this->Deallocate__Parameter__AMSGrad(); break;
        default:
            PRINT_FORMAT("%s: ERROR: Unknow type optimizer function (%u)." NEW_LINE,
                    __FUNCTION__,
                    this->type_optimizer_function);
                break;
    }
}

__device__ void CUDA_Neural_Network::Deallocate__Parameter__Gradient_Descent(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_previous_delta_parameters);
}

__device__ void CUDA_Neural_Network::Deallocate__Parameter__iRPROP_minus(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_previous_steps);
    SAFE_DELETE_ARRAY(this->ptr_array_previous_derivatives_parameters);
}

__device__ void CUDA_Neural_Network::Deallocate__Parameter__iRPROP_plus(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_previous_steps);
    SAFE_DELETE_ARRAY(this->ptr_array_previous_delta_parameters);
    SAFE_DELETE_ARRAY(this->ptr_array_previous_derivatives_parameters);
}

__device__ void CUDA_Neural_Network::Deallocate__Parameter__Adam(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_previous_biased_first_moment);
    SAFE_DELETE_ARRAY(this->ptr_array_previous_biased_second_moment);
}
    
__device__ void CUDA_Neural_Network::Deallocate__Parameter__AMSGrad(void)
{
    this->Deallocate__Parameter__Adam();

    SAFE_DELETE_ARRAY(this->ptr_array_previous_biased_second_moment_hat);
}

__device__ void CUDA_Neural_Network::Deallocate__Parameter__Regularization(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_mask_regularized_parameters); // delete[] array T_.
}

__device__ void CUDA_Neural_Network::Deallocate_Cost(void)
{
    // Loss parameters.
    SAFE_DELETE_ARRAY(this->ptr_array_number_loss); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_number_bit_fail); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_loss_values); // delete[] array float.
    // |END| Loss parameters. |END|
    
    // Accuracy parameters.
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[0u]); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[1u]); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[2u]); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[3u]); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[4u]); // delete[] array size_t.
    // |END| Accuracy parameters. |END|
}

__device__ void CUDA_Neural_Network::Deallocate_Reduce_Batch(void)
{
    SAFE_FREE(this->ptr_array_dim3_grid_reduce_threads);
    SAFE_FREE(this->ptr_array_dim3_block_reduce_threads);

    SAFE_FREE(this->ptr_array_dim3_grid_reduce_threads_DP);
    SAFE_FREE(this->ptr_array_dim3_block_reduce_threads_DP);
}

__device__ void CUDA_Neural_Network::Deallocate_Reduce_Cost(void)
{
    // Loss parameters.
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_number_loss);
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_bit_fail_values);
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_loss_values);
    // |END| Loss parameters. |END|
    
    // Accuracy parameters.
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_accuracy_values[0u]);
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_accuracy_values[1u]);
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_accuracy_values[2u]);
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_accuracy_values[3u]);
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_accuracy_values[4u]);
    // |END| Accuracy parameters. |END|
}

__device__ void CUDA_Neural_Network::Deallocate_Batch_Reduce(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_reduce_batch_size); // delete[] array size_t.

    SAFE_FREE(this->ptr_array_neuron_units_dim3_grid_reduce_batch); // free array dim3.
    SAFE_FREE(this->ptr_array_neuron_units_dim3_block_reduce_batch); // free array dim3.
}

__device__ void CUDA_Neural_Network::Deallocate__Normalized_Unit__Batch_Normalization(void)
{
    if(this->ptr_array_2D_neurons_reduce_batch_mean != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_2D_neurons_reduce_batch_mean[0u]); // delete[] array T_.

        delete[](this->ptr_array_2D_neurons_reduce_batch_mean); // delete[] array T_.
        this->ptr_array_2D_neurons_reduce_batch_mean = nullptr;
    }

    if(this->ptr_array_2D_neurons_reduce_batch_variance != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_2D_neurons_reduce_batch_variance[0u]); // delete[] array T_.

        delete[](this->ptr_array_2D_neurons_reduce_batch_variance); // delete[] array T_.
        this->ptr_array_2D_neurons_reduce_batch_variance = nullptr;
    }
}

__device__ void CUDA_Neural_Network::Deallocate__Neuron__Mask_Dropout_Bernoulli(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_af_units_mask_dropout_bernoulli); // delete[] array bool.
}

__device__ void CUDA_Neural_Network::Deallocate__Cell_Unit__Mask_Dropout_Zoneout(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_cell_units_mask_dropout_zoneout); // delete[] array bool.
}

__device__ void CUDA_Neural_Network::Deallocate__Neurons_Reduce_Summation(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_reduce_summation_size); // delete[] array size_t.

    if(this->ptr_array_2D_neurons_reduce_summation != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_2D_neurons_reduce_summation[0u]); // delete[] array T_.

        delete[](this->ptr_array_2D_neurons_reduce_summation); // delete[] array T_.
        this->ptr_array_2D_neurons_reduce_summation = nullptr;
    }

    SAFE_FREE(this->ptr_array_neuron_units_dim3_grid_reduce_summation); // free array dim3.
    SAFE_FREE(this->ptr_array_neuron_units_dim3_block_reduce_summation); // free array dim3.
}

__device__ void CUDA_Neural_Network::Deallocate__Neurons_Reduce_Error(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_reduce_error_size); // delete[] array size_t.

    if(this->ptr_array_2D_neurons_reduce_error != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_2D_neurons_reduce_error[0u]); // delete[] array T_.

        delete[](this->ptr_array_2D_neurons_reduce_error); // delete[] array T_.
        this->ptr_array_2D_neurons_reduce_error = nullptr;
    }

    SAFE_FREE(this->ptr_array_neuron_units_dim3_grid_reduce_error); // free array dim3.
    SAFE_FREE(this->ptr_array_neuron_units_dim3_block_reduce_error); // free array dim3.
}

__device__ void CUDA_Neural_Network::Deallocate__Neurons_Reduce_Norms(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_reduce_norms_size); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_neuroyed_number_neurons_in_layer); // delete[] array size_t.

    if(this->ptr_array_2D_neurons_reduce_norms != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_2D_neurons_reduce_norms[0u]); // delete[] array T_.

        delete[](this->ptr_array_2D_neurons_reduce_norms); // delete[] array T_.
        this->ptr_array_2D_neurons_reduce_norms = nullptr;
    }

    if(this->ptr_array_2D_neurons_dim3_grid_reduce_norms != NULL)
    {
        SAFE_FREE(this->ptr_array_2D_neurons_dim3_grid_reduce_norms[0u]); // free array dim3.

        free(this->ptr_array_2D_neurons_dim3_grid_reduce_norms); // free array dim3.
        this->ptr_array_2D_neurons_dim3_grid_reduce_norms = NULL;
    }
    
    if(this->ptr_array_2D_neurons_dim3_block_reduce_norms != NULL)
    {
        SAFE_FREE(this->ptr_array_2D_neurons_dim3_block_reduce_norms[0u]); // free array dim3.

        free(this->ptr_array_2D_neurons_dim3_block_reduce_norms); // free array dim3.
        this->ptr_array_2D_neurons_dim3_block_reduce_norms = NULL;
    }
}
