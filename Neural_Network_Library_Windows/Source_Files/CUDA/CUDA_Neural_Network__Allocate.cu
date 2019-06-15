#include <Tools/CUDA_Fill_1D.cuh>
#include <Tools/CUDA_Zero_1D.cuh>
#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Transpose.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

__device__ bool CUDA_Neural_Network::Allocate_Weights_Transposed(void)
{
    if(this->total_weights_allocated == 0u)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate memory! Total weights allocated equal zero." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else if(this->ptr_array_transposed_weights == nullptr)
    {
        this->ptr_array_transposed_weights = new T_[this->total_weights_allocated];
        if(this->ptr_array_transposed_weights == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->total_weights_allocated,
                            this->ptr_array_transposed_weights,
                            this->ptr_array_dim3_grid + 2,
                            this->ptr_array_dim3_block + 2);
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Parameter(void)
{
    if(this->total_parameters_allocated == 0u)
    {
        this->ptr_array_parameters = new T_[this->total_parameters];
        if(this->ptr_array_parameters == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->total_parameters,
                            this->ptr_array_parameters,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
        
        this->ptr_array_ptr_connections = new void*[this->total_parameters];
        if(this->ptr_array_ptr_connections == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<void*>(this->total_parameters,
                                this->ptr_array_ptr_connections,
                                this->ptr_array_dim3_grid + 1,
                                this->ptr_array_dim3_block + 1);

        this->total_weights_allocated = this->total_weights;
            
        this->total_parameters_allocated = this->total_parameters;
    }
    else
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate on allocated memory! Use reallocate function." NEW_LINE, __FUNCTION__);

        return(false);
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Parameter__Optimizer(void)
{
    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD: return(this->Allocate__Parameter__Gradient_Descent());
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus: return(this->Allocate__Parameter__iRPROP_minus());
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus: return(this->Allocate__Parameter__iRPROP_plus());
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: return(this->Allocate__Parameter__Adam());
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: return(this->Allocate__Parameter__AMSGrad());
        default:
            PRINT_FORMAT("%s: ERROR: Unknow type optimizer function (%u)." NEW_LINE,
                    __FUNCTION__,
                    this->type_optimizer_function);
                return(false);
    }
}
    
__device__ bool CUDA_Neural_Network::Allocate__Parameter__Gradient_Descent(void)
{
    if(this->learning_momentum != 0_T
        &&
        this->ptr_array_previous_delta_parameters == nullptr)
    {
        this->ptr_array_previous_delta_parameters = new T_[this->total_parameters];
        if(this->ptr_array_previous_delta_parameters == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->total_parameters,
                            this->ptr_array_previous_delta_parameters,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Parameter__iRPROP_minus(void)
{
    if(this->ptr_array_previous_steps == nullptr)
    {
        this->ptr_array_previous_steps = new T_[this->total_parameters];
        if(this->ptr_array_previous_steps == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Memory::Fill_1D<T_>(this->total_parameters,
                                                             this->ptr_array_previous_steps,
                                                             this->rprop_delta_zero,
                                                             this->ptr_array_dim3_grid + 1,
                                                             this->ptr_array_dim3_block + 1);
    }
    
    if(this->ptr_array_previous_derivatives_parameters == nullptr)
    {
        this->ptr_array_previous_derivatives_parameters = new T_[this->total_parameters];
        if(this->ptr_array_previous_derivatives_parameters == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->total_parameters,
                            this->ptr_array_previous_derivatives_parameters,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Parameter__iRPROP_plus(void)
{
    if(this->ptr_array_previous_steps == nullptr)
    {
        this->ptr_array_previous_steps = new T_[this->total_parameters];
        if(this->ptr_array_previous_steps == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Memory::Fill_1D<T_>(this->total_parameters,
                                                             this->ptr_array_previous_steps,
                                                             this->rprop_delta_zero,
                                                             this->ptr_array_dim3_grid + 1,
                                                             this->ptr_array_dim3_block + 1);
    }
    
    if(this->ptr_array_previous_delta_parameters == nullptr)
    {
        this->ptr_array_previous_delta_parameters = new T_[this->total_parameters];
        if(this->ptr_array_previous_delta_parameters == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->total_parameters,
                            this->ptr_array_previous_delta_parameters,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }
    
    if(this->ptr_array_previous_derivatives_parameters == nullptr)
    {
        this->ptr_array_previous_derivatives_parameters = new T_[this->total_parameters];
        if(this->ptr_array_previous_derivatives_parameters == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->total_parameters,
                            this->ptr_array_previous_derivatives_parameters,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Parameter__Adam(void)
{
    if(this->ptr_array_previous_biased_first_moment == nullptr)
    {
        this->ptr_array_previous_biased_first_moment = new T_[this->total_parameters];
        if(this->ptr_array_previous_biased_first_moment == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->total_parameters,
                            this->ptr_array_previous_biased_first_moment,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }
    
    if(this->ptr_array_previous_biased_second_moment == nullptr)
    {
        this->ptr_array_previous_biased_second_moment = new T_[this->total_parameters];
        if(this->ptr_array_previous_biased_second_moment == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->total_parameters,
                            this->ptr_array_previous_biased_second_moment,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }

    return(true);
}
    
__device__ bool CUDA_Neural_Network::Allocate__Parameter__AMSGrad(void)
{
    this->Allocate__Parameter__Adam();
    
    if(this->ptr_array_previous_biased_second_moment_hat == nullptr)
    {
        this->ptr_array_previous_biased_second_moment_hat = new T_[this->total_parameters];
        if(this->ptr_array_previous_biased_second_moment_hat == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->total_parameters,
                            this->ptr_array_previous_biased_second_moment_hat,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }

    return(true);
}

__global__ void kernel__CNeural_Network__Allocate_Structure(size_t const total_layers_received,
                                                                                            size_t const maximum_allowable_memory_bytes_received,
                                                                                            class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    if(ptr_CNeural_Network_received->Allocate__Structure(total_layers_received, maximum_allowable_memory_bytes_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Allocate__Structure(%u, %zu)\" function." NEW_LINE,
                                 __FUNCTION__,
                                 total_layers_received,
                                 maximum_allowable_memory_bytes_received);

        return;
    }
}

__host__ __device__ bool CUDA_Neural_Network::Allocate__Structure(size_t const number_layers_received, size_t const maximum_allowable_memory_bytes_received)
{
#if defined(__CUDA_ARCH__)
    // Dimension.
    this->total_layers = number_layers_received;
    this->number_inputs = 0u;
    this->number_outputs = 0u;
    this->total_neuron_units_allocated = this->total_neuron_units = 0u;
    this->total_block_units_allocated = this->total_block_units = 0u;
    this->total_cell_units_allocated = this->total_cell_units = 0u;
    this->total_parameters_allocated = this->total_parameters = 0u;
    this->total_weights_allocated = this->total_weights = 0u;
    
    size_t *tmp_ptr_array_number_neurons_by_layer(this->ptr_array_number_neurons_by_layer = new size_t[number_layers_received]);
    if(tmp_ptr_array_number_neurons_by_layer == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. new size_t[nLayer(%u)]" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received);

        return(false);
    }
    memset(tmp_ptr_array_number_neurons_by_layer,
                0,
                number_layers_received * sizeof(size_t));
    
    //    Allocate layers.
    struct CUDA_Layer *tmp_ptr_layer_it(this->ptr_array_layers = new struct CUDA_Layer[number_layers_received]);
    if(tmp_ptr_layer_it == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. new size_t[nLayer(%u)]" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received);

        return(false);
    }
    struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer = tmp_ptr_layer_it + number_layers_received);
    //    |END| Allocate layers. |END|
    
    // Allocate dim3 neurons by layer.
    struct dim3 *tmp_ptr_array_layers_dim3_grid_neurons(static_cast<struct dim3*>(malloc(number_layers_received * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_grid_neurons == NULL)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(nLayers(%u) * sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received,
                                sizeof(struct dim3));

        return(false);
    }
    memset(tmp_ptr_array_layers_dim3_grid_neurons,
                    0,
                    number_layers_received * sizeof(struct dim3));
    this->ptr_array_layers_dim3_grid_neurons = tmp_ptr_array_layers_dim3_grid_neurons;

    struct dim3 *tmp_ptr_array_layers_dim3_block_neurons(static_cast<struct dim3*>(malloc(number_layers_received * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_block_neurons == NULL)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(nLayers(%u) * sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received,
                                sizeof(struct dim3));

        return(false);
    }
    memset(tmp_ptr_array_layers_dim3_block_neurons,
                    0,
                    number_layers_received * sizeof(struct dim3));
    this->ptr_array_layers_dim3_block_neurons = tmp_ptr_array_layers_dim3_block_neurons;
    // |END| Allocate dim3 neurons by layer. |END|
    
    // Allocate dim3 neurons dynamic parallelisme by layer.
    struct dim3 *tmp_ptr_array_layers_dim3_grid_neurons_DP(static_cast<struct dim3*>(malloc(number_layers_received * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_grid_neurons_DP == NULL)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(nLayers(%u) * sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received,
                                sizeof(struct dim3));

        return(false);
    }
    memset(tmp_ptr_array_layers_dim3_grid_neurons_DP,
                    0,
                    number_layers_received * sizeof(struct dim3));
    this->ptr_array_layers_dim3_grid_neurons_DP = tmp_ptr_array_layers_dim3_grid_neurons_DP;

    struct dim3 *tmp_ptr_array_layers_dim3_block_neurons_DP(static_cast<struct dim3*>(malloc(number_layers_received * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_block_neurons_DP == NULL)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(nLayers(%u) * sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received,
                                sizeof(struct dim3));

        return(false);
    }
    memset(tmp_ptr_array_layers_dim3_block_neurons_DP,
                    0,
                    number_layers_received * sizeof(struct dim3));
    this->ptr_array_layers_dim3_block_neurons_DP = tmp_ptr_array_layers_dim3_block_neurons_DP;
    // |END| Allocate dim3 neurons dynamic parallelisme by layer. |END|
    
    // Allocate dim3 neurons cuRAND by layer.
    struct dim3 *tmp_ptr_array_layers_dim3_grid_neurons_cuRAND(static_cast<struct dim3*>(malloc(number_layers_received * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_grid_neurons_cuRAND == NULL)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(nLayers(%u) * sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received,
                                sizeof(struct dim3));

        return(false);
    }
    memset(tmp_ptr_array_layers_dim3_grid_neurons_cuRAND,
                    0,
                    number_layers_received * sizeof(struct dim3));
    this->ptr_array_layers_dim3_grid_neurons_cuRAND = tmp_ptr_array_layers_dim3_grid_neurons_cuRAND;

    struct dim3 *tmp_ptr_array_layers_dim3_block_neurons_cuRAND(static_cast<struct dim3*>(malloc(number_layers_received * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_block_neurons_cuRAND == NULL)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(nLayers(%u) * sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received,
                                sizeof(struct dim3));

        return(false);
    }
    memset(tmp_ptr_array_layers_dim3_block_neurons_cuRAND,
                    0,
                    number_layers_received * sizeof(struct dim3));
    this->ptr_array_layers_dim3_block_neurons_cuRAND = tmp_ptr_array_layers_dim3_block_neurons_cuRAND;
    // |END| Allocate dim3 neurons cuRAND by layer. |END|
    
    // Allocate dim3 batch neurons by layer.
    struct dim3 *tmp_ptr_array_layers_dim3_grid_batch_neurons(static_cast<struct dim3*>(malloc(number_layers_received * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_grid_batch_neurons == NULL)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(nLayers(%u) * sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received,
                                sizeof(struct dim3));

        return(false);
    }
    memset(tmp_ptr_array_layers_dim3_grid_batch_neurons,
                    0,
                    number_layers_received * sizeof(struct dim3));
    this->ptr_array_layers_dim3_grid_batch_neurons = tmp_ptr_array_layers_dim3_grid_batch_neurons;

    struct dim3 *tmp_ptr_array_layers_dim3_block_batch_neurons(static_cast<struct dim3*>(malloc(number_layers_received * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_block_batch_neurons == NULL)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(nLayers(%u) * sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received,
                                sizeof(struct dim3));

        return(false);
    }
    memset(tmp_ptr_array_layers_dim3_block_batch_neurons,
                    0,
                    number_layers_received * sizeof(struct dim3));
    this->ptr_array_layers_dim3_block_batch_neurons = tmp_ptr_array_layers_dim3_block_batch_neurons;
    // |END| Allocate dim3 batch neurons by layer. |END|
    
    // Allocate dim3 weights by layer.
    struct dim3 *tmp_ptr_array_layers_dim3_grid_weights(static_cast<struct dim3*>(malloc(number_layers_received * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_grid_weights == NULL)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(nLayers(%u) * sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received,
                                sizeof(struct dim3));

        return(false);
    }
    memset(tmp_ptr_array_layers_dim3_grid_weights,
                    0,
                    number_layers_received * sizeof(struct dim3));
    this->ptr_array_layers_dim3_grid_weights = tmp_ptr_array_layers_dim3_grid_weights;

    struct dim3 *tmp_ptr_array_layers_dim3_block_weights(static_cast<struct dim3*>(malloc(number_layers_received * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_block_weights == NULL)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(nLayers(%u) * sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received,
                                sizeof(struct dim3));

        return(false);
    }
    memset(tmp_ptr_array_layers_dim3_block_weights,
                    0,
                    number_layers_received * sizeof(struct dim3));
    this->ptr_array_layers_dim3_block_weights = tmp_ptr_array_layers_dim3_block_weights;
    // |END| Allocate dim3 weights by layer. |END|
    
    //    Allocate storage dim3 batch.
    class CUDA_Storage_Dim3 *tmp_ptr_array_storage_dim3(this->ptr_array_layers_Class_Storage_Dim3_Batch = new class CUDA_Storage_Dim3[number_layers_received]);
    if(tmp_ptr_array_storage_dim3 == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. new size_t[nLayer(%u)]" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received);

        return(false);
    }
    //    |END| Allocate storage dim3 batch. |END|

    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        tmp_ptr_layer_it->dropout_values[0u] = 1_T;
        tmp_ptr_layer_it->dropout_values[1u] = 0_T;

        tmp_ptr_layer_it->ptr_array_neuron_units = nullptr;
        tmp_ptr_layer_it->ptr_last_neuron_unit = nullptr;

        tmp_ptr_layer_it->ptr_number_neurons = tmp_ptr_array_number_neurons_by_layer++;
        
        tmp_ptr_layer_it->ptr_dim3_grid_neurons = tmp_ptr_array_layers_dim3_grid_neurons++;
        tmp_ptr_layer_it->ptr_dim3_block_neurons = tmp_ptr_array_layers_dim3_block_neurons++;
        
        tmp_ptr_layer_it->ptr_dim3_grid_neurons_DP = tmp_ptr_array_layers_dim3_grid_neurons_DP++;
        tmp_ptr_layer_it->ptr_dim3_block_neurons_DP = tmp_ptr_array_layers_dim3_block_neurons_DP++;
        
        tmp_ptr_layer_it->ptr_dim3_grid_neurons_cuRAND = tmp_ptr_array_layers_dim3_grid_neurons_cuRAND++;
        tmp_ptr_layer_it->ptr_dim3_block_neurons_cuRAND = tmp_ptr_array_layers_dim3_block_neurons_cuRAND++;
        
        tmp_ptr_layer_it->ptr_dim3_grid_batch_neurons = tmp_ptr_array_layers_dim3_grid_batch_neurons++;
        tmp_ptr_layer_it->ptr_dim3_block_batch_neurons = tmp_ptr_array_layers_dim3_block_batch_neurons++;
        
        tmp_ptr_layer_it->ptr_dim3_grid_weights = tmp_ptr_array_layers_dim3_grid_weights++;
        tmp_ptr_layer_it->ptr_dim3_block_weights = tmp_ptr_array_layers_dim3_block_weights++;

        tmp_ptr_layer_it->ptr_Class_Storage_Dim3_Batch = tmp_ptr_array_storage_dim3++;
    }

    this->ptr_array_transposed_weights = nullptr;
    this->ptr_array_parameters = nullptr;
    this->ptr_array_ptr_connections = nullptr;
    // |END| Dimension. |END|
        
    //    Allocate storage dim3 memcpy.
    this->ptr_Class_Storage_Dim3_Memcpy = new class CUDA_Storage_Dim3;
    if(this->ptr_Class_Storage_Dim3_Memcpy == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. new size_t[nLayer(%u)]" NEW_LINE,
                                    __FUNCTION__,
                                number_layers_received);

        return(false);
    }
    //    |END| Allocate storage dim3 memcpy. |END|

    // General parameters.
    this->type_network = MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_FEEDFORWARD;
    this->connection_rate = 0_T;
    this->number_recurrent_depth = 1u;
    this->number_time_delays = 0u;
    this->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_INFERENCE;
    // |END| General parameters. |END|
        
    // Gradient descent parameters.
    this->learning_rate = 0.01_T;
    this->learning_momentum = 0.9_T;
    this->use_Nesterov = true;
    this->ptr_array_previous_delta_parameters = nullptr;
    // |END| Gradient descent parameters. |END|
    
    // Quickprop parameters.
    this->quickprop_decay = -0.0001f;
    this->quickprop_mu = 1.75f;
    // |END| Quickprop parameters. |END|

    // Resillent propagation parameters.
    this->rprop_increase_factor = 1.2f;
    this->rprop_decrease_factor = 0.5f;
    this->rprop_delta_min = 1e-6f;
    this->rprop_delta_max = 50.0f;
    this->rprop_delta_zero = 0.1f;
    this->ptr_array_previous_steps = nullptr;
    //this->ptr_array_previous_delta_parameters = nullptr;
    this->ptr_array_previous_derivatives_parameters = nullptr;
    this->loss_rprop = FLT_MAX;
    this->previous_loss_rprop = FLT_MAX;
    // |END| Resillent propagation parameters. |END|
        
    // SARProp parameters.
     this->sarprop_weight_decay_shift = -6.644f;
     this->sarprop_step_error_threshold_factor = 0.1f;
     this->sarprop_step_error_shift = 1.385f;
     this->sarprop_temperature = 0.015f;
     this->sarprop_epoch = 0u;
    // |END| SARProp parameters. |END|
        
    // AMSGrad parameters.
    //    Adam parameters.
     this->adam_learning_rate = 0.001_T;
     this->adam_beta1 = 0.9_T;
     this->adam_beta2 = 0.999_T;
     this->adam_epsilon = 1.0e-8_T;
     this->use_adam_bias_correction = true;
     this->adam_gamma = 0.1_T;
     this->ptr_array_previous_biased_first_moment = nullptr;
     this->ptr_array_previous_biased_second_moment = nullptr;
    //    |END| Adam parameters. |END|
     this->ptr_array_previous_biased_second_moment_hat = nullptr;
    // |END| AMSGrad parameters. |END|
        
    // Warm restarts parameters.
    this->use_Warm_Restarts = false;
    this->warm_restarts_decay_learning_rate = 1_T;
    this->warm_restarts_maximum_learning_rate = this->warm_restarts_initial_maximum_learning_rate = 1_T;
    this->warm_restarts_minimum_learning_rate = 1.0e-7_T;
    this->warm_restarts_T_i = this->warm_restarts_initial_T_i = 1_T;
    this->warm_restarts_multiplier = 2_T;
    // |END| Warm restarts parameters. |END|

    // Training parameters.
    this->type_optimizer_function = MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NONE;
    this->type_loss_function = MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_NONE;
    this->bit_fail_limit = 0.35_T;
    this->ptr_array_derivatives_parameters = nullptr;
    this->optimizer_time_step = 0_T;
    this->epoch_time_step = 1_T;
    // |END| Training parameters. |END|
        
    // Regularization parameters.
    this->use_Dropout = false;
    this->ptr_array_mask_dropout_parameters = nullptr;
    this->ptr_array_mask_regularized_parameters = nullptr;
    this->regularization__max_norm_constraints = 0_T;
    this->regularization__l1 = 0_T;
    this->regularization__l2 = 0_T;
    this->regularization__weight_decay = 0_T;
    // |END| Regularization parameters. |END|
        
    // Normalization parameters.
    this->use_Batch_Renormalization = false;
    this->normalization_momentum_average = 0.01_T;
    this->normalization_epsilon = 1.0e-5_T;
    this->batch_renormalization_r_correction_maximum = 1_T;
    this->batch_renormalization_d_correction_maximum = 0_T;
    // |END| Normalization parameters. |END|

    // Loss parameters.
    this->ptr_array_number_loss = new size_t[1u]; *this->ptr_array_number_loss = 0_zu;
    this->ptr_array_number_bit_fail = new size_t[1u]; *this->ptr_array_number_bit_fail = 0_zu;
    this->ptr_array_loss_values = new T_[1u]; *this->ptr_array_loss_values = (std::numeric_limits<T_>().max)();
    this->loss_training = FLT_MAX;
    this->loss_validating = FLT_MAX;
    this->loss_testing = FLT_MAX;
    // |END| Loss parameters. |END|
        
    // Accuracy parameters.
    if((this->ptr_array_accuracy_values[0u] = new T_[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 sizeof(T_),
                                 __LINE__);

        return(false);
    }
    else { this->ptr_array_accuracy_values[0u][0u] = 0_T; }

    if((this->ptr_array_accuracy_values[1u] = new T_[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 sizeof(T_),
                                 __LINE__);

        return(false);
    }
    else { this->ptr_array_accuracy_values[1u][0u] = 0_T; }

    if((this->ptr_array_accuracy_values[2u] = new T_[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 sizeof(T_),
                                 __LINE__);

        return(false);
    }
    else { this->ptr_array_accuracy_values[0u][2u] = 0_T; }

    if((this->ptr_array_accuracy_values[3u] = new T_[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 sizeof(T_),
                                 __LINE__);

        return(false);
    }
    else { this->ptr_array_accuracy_values[0u][3u] = 0_T; }
    
    if((this->ptr_array_accuracy_values[4u] = new T_[1u]) == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 sizeof(T_),
                                 __LINE__);

        return(false);
    }
    else { this->ptr_array_accuracy_values[0u][4u] = 0_T; }

    this->number_accuracy_trial = 0u;
    this->accuracy_variance = 0.0f;
    this->accuracy_training = 0.0f;
    this->accuracy_validating = 0.0f;
    this->accuracy_testing = 0.0f;
    // |END| Accuracy parameters. |END|

    // Computation parameters.
    this->limit_device_runtime_pending_launch_count = 2048u; // Default fixed pool size.
    this->number_threads = 1u;
    this->cache_number_threads = 0u;
    this->batch_size = 1u;
    this->cache_batch_size = 0u;
    this->maximum_allowable_memory_bytes = maximum_allowable_memory_bytes_received;
    this->_ptr_Class_Device_Information_Array = nullptr;
    // |END| Computation parameters. |END|

    // cuRAND parameters.
    this->number_cuRAND_State_MTGP32_weighted = 0u;
    this->number_cuRAND_State_MTGP32_neuroyed = 0u;

    this->ptr_array_cuRAND_State_MTGP32_weighted = nullptr;
    this->ptr_array_cuRAND_State_MTGP32_neuroyed = nullptr;
    // |END| cuRAND parameters. |END|

    // Neurons variable.
    this->ptr_array_af_units_mask_dropout_bernoulli = nullptr;
    this->ptr_array_cell_units_mask_dropout_zoneout = nullptr;

    this->neurons_total_reduce_summation_size = 0u;
    this->neurons_total_reduce_error_size = 0u;
    this->neurons_total_reduce_batch_size = 0u;
    this->neurons_total_reduce_norms_size = 0u;

    this->ptr_array_neuron_units_first_forward_connection_index = nullptr;
    this->ptr_array_neuron_units_last_forward_connection_index = nullptr;
    this->ptr_array_neuron_units_number_forward_connections = nullptr;
    this->ptr_array_neuron_units_reduce_summation_size = nullptr;
    this->ptr_array_neuron_units_reduce_error_size = nullptr;
    this->ptr_array_neuron_units_reduce_batch_size = nullptr;
    this->ptr_array_neuron_units_reduce_norms_size = nullptr;
    this->ptr_array_neuroyed_number_neurons_in_layer = nullptr;

    this->ptr_array_neuron_units_summations = nullptr;
    this->ptr_array_neuron_units_activation_steepness = nullptr;
    this->ptr_array_neuron_units_values = nullptr;
    this->ptr_array_normalized_batch_units_values_hats = nullptr;
    this->ptr_array_normalized_batch_units_values_normalizes = nullptr;
    this->ptr_array_normalized_batch_units_means = nullptr;
    this->ptr_array_normalized_batch_units_variances = nullptr;
    this->ptr_array_neuron_units_transposed_mean = nullptr;
    this->ptr_array_neuron_units_transposed_variance = nullptr;
    this->ptr_array_normalized_batch_units_derivatives_means = nullptr;
    this->ptr_array_normalized_batch_units_derivatives_variances = nullptr;
    this->ptr_array_normalized_batch_units_means_averages = nullptr;
    this->ptr_array_normalized_batch_units_variances_averages = nullptr;
    this->ptr_array_normalized_batch_units_r_corrections = nullptr;
    this->ptr_array_normalized_batch_units_d_corrections = nullptr;
    this->ptr_array_normalized_batch_units_scales = nullptr;
    this->ptr_array_normalized_batch_units_shifts = nullptr;
    this->ptr_array_neuron_units_errors = nullptr;
    this->ptr_array_2D_neurons_reduce_summation = nullptr;
    this->ptr_array_2D_neurons_reduce_error = nullptr;
    this->ptr_array_2D_neurons_reduce_batch_mean = nullptr;
    this->ptr_array_2D_neurons_reduce_batch_variance = nullptr;
    this->ptr_array_2D_neurons_reduce_norms = nullptr;
    this->ptr_array_mask_dropout_parameters = nullptr;

    this->ptr_array_neuron_units_type_activation_function = nullptr;
    // |END| Neurons variable. |END|

    this->ptr_array_dim3_grid = static_cast<struct dim3*>(malloc(TOTAL_KERNEL_PARALLEL * sizeof(struct dim3)));
    if(this->ptr_array_dim3_grid == NULL)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. new size_t[TOTAL_KERNEL_PARALLEL(%u)]" NEW_LINE,
                                    __FUNCTION__,
                                TOTAL_KERNEL_PARALLEL);

        return(false);
    }
    memset(this->ptr_array_dim3_grid,
                0,
                TOTAL_KERNEL_PARALLEL * sizeof(struct dim3));

    this->ptr_array_dim3_block = static_cast<struct dim3*>(malloc(TOTAL_KERNEL_PARALLEL * sizeof(struct dim3)));
    if(this->ptr_array_dim3_block == NULL)
    {
        PRINT_FORMAT("%s: ERROR: Can not Allocate memory. new size_t[TOTAL_KERNEL_PARALLEL(%u)]" NEW_LINE,
                                    __FUNCTION__,
                                TOTAL_KERNEL_PARALLEL);

        return(false);
    }
    memset(this->ptr_array_dim3_block,
                0,
                TOTAL_KERNEL_PARALLEL * sizeof(struct dim3));
    
    // Struct dim3 variable.
    this->ptr_array_neuron_units_dim3_grid_connections = NULL;
    this->ptr_array_neuron_units_dim3_block_connections = NULL;
    
    this->ptr_array_dim3_grid_reduce_threads = NULL;
    this->ptr_array_dim3_block_reduce_threads = NULL;
    
    this->ptr_array_dim3_grid_reduce_threads_DP = NULL;
    this->ptr_array_dim3_block_reduce_threads_DP = NULL;
    
    this->ptr_array_neuron_units_dim3_grid_reduce_summation = NULL;
    this->ptr_array_neuron_units_dim3_block_reduce_summation = NULL;
    
    this->ptr_array_neuron_units_dim3_grid_reduce_error = NULL;
    this->ptr_array_neuron_units_dim3_block_reduce_error = NULL;

    this->ptr_array_neuron_units_dim3_grid_reduce_batch = NULL;
    this->ptr_array_neuron_units_dim3_block_reduce_batch = NULL;

    this->ptr_array_2D_neurons_dim3_grid_reduce_norms = NULL;
    this->ptr_array_2D_neurons_dim3_block_reduce_norms = NULL;
    // |END| Struct dim3 variable. |END|

    this->total_reduce_batch_size = 0u;
    this->total_reduce_batch_DP_size = 0u;

    this->ptr_array_reduce_number_loss = nullptr;
    this->ptr_array_reduce_loss_values = nullptr;
    this->ptr_array_reduce_bit_fail_values = nullptr;
    this->ptr_array_reduce_accuracy_values[0u] = nullptr;
    this->ptr_array_reduce_accuracy_values[1u] = nullptr;
    this->ptr_array_reduce_accuracy_values[2u] = nullptr;
    this->ptr_array_reduce_accuracy_values[3u] = nullptr;
    this->ptr_array_reduce_accuracy_values[4u] = nullptr;
#else
    kernel__CNeural_Network__Allocate_Structure <<< 1u, 1u >>> (number_layers_received,
                                                                                                  maximum_allowable_memory_bytes_received,
                                                                                                  this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    if(this->Initialize_CUDA_Device() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialize_CUDA_Device()\" function." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__);

        return(false);
    }
#endif
        
    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate_Reduce_Threads(void)
{
    if(this->ptr_array_dim3_grid_reduce_threads == nullptr || this->ptr_array_dim3_grid_reduce_threads_DP == nullptr)
    {
        if(this->Allocate_Reduce_Threads_Dim() == false)
        {
            PRINT_FORMAT("%s: ERROR: From \"Allocate_Reduce_Threads_Dim\"" NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        else if(this->Allocate_Reduce_Threads_Dim_DP() == false)
        {
            PRINT_FORMAT("%s: ERROR: From \"Allocate_Reduce_Threads_Dim_DP\"" NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate_Reduce_Threads_Dim(void)
{
    if(this->ptr_array_dim3_grid_reduce_threads == nullptr)
    {
        size_t tmp_total_elements_to_reduce,
                          tmp_index_dim3(0u);
        
        class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;

        // Compute dimension reduce data batch.
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = this->number_threads;
        
        // Dimension required to reduce the number of elements.
        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                              0u,
                                                                                              tmp_dim3_grid,
                                                                                              tmp_dim3_block);
        
        // Get remaining elements to reduce and store it.
        this->total_reduce_batch_size = tmp_total_elements_to_reduce = tmp_dim3_grid.x;

        if(tmp_total_elements_to_reduce == 0u)
        {
            PRINT_FORMAT("%s: ERROR: No elements to reduce." NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        // |END| Compute dimension reduce data batch. |END|
        
        // Allocating neurons reduce summation dim3 grid.
        struct dim3 *tmp_ptr_array_dim3_grid_reduce_threads(static_cast<struct dim3*>(malloc(tmp_total_elements_to_reduce * sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_grid_reduce_threads == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_dim3_grid_reduce_threads,
                    0,
                    tmp_total_elements_to_reduce * sizeof(struct dim3));
        this->ptr_array_dim3_grid_reduce_threads = tmp_ptr_array_dim3_grid_reduce_threads;
        // |END| Allocating neurons reduce summation dim3 grid. |END|
            
        // Allocating neurons reduce summation dim3 block.
        struct dim3 *tmp_ptr_array_dim3_block_reduce_threads(static_cast<struct dim3*>(malloc(tmp_total_elements_to_reduce * sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_block_reduce_threads == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_dim3_block_reduce_threads,
                    0,
                    tmp_total_elements_to_reduce * sizeof(struct dim3));
        this->ptr_array_dim3_block_reduce_threads = tmp_ptr_array_dim3_block_reduce_threads;
        // |END| Allocating neurons reduce summation dim3 block. |END|
        
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = this->number_threads;

        // Loop to reduce "number of elements" to one at the end.
        do
        {
            // Compute remaining results to reduce.
            tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                    0u,
                                                                                                    tmp_ptr_array_dim3_grid_reduce_threads[tmp_index_dim3],
                                                                                                    tmp_ptr_array_dim3_block_reduce_threads[tmp_index_dim3]);

            // Get the remaining results to reduce.
            tmp_total_elements_to_reduce = tmp_ptr_array_dim3_grid_reduce_threads[tmp_index_dim3].x;

            // Increment index to dim3.
            ++tmp_index_dim3;
        } while(tmp_total_elements_to_reduce != 1u);
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate_Reduce_Threads_Dim_DP(void)
{
    if(this->ptr_array_dim3_grid_reduce_threads_DP == nullptr)
    {
        size_t tmp_total_elements_to_reduce,
                          tmp_index_dim3(0u);
        
        class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;

        // Compute dimension reduce data batch.
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = this->number_threads;
        
        // Dimension required to reduce the number of elements.
        tmp_ptr_CUDA_Device->Grid_Block_Reduce_Dynamic_Parallelisme(tmp_total_elements_to_reduce,
                                                                                                            tmp_ptr_CUDA_Device->Get__Maximum_Blocks_Per_Multiprocessor(),
                                                                                                            tmp_dim3_grid,
                                                                                                            tmp_dim3_block);
        
        // Get remaining elements to reduce and store it.
        this->total_reduce_batch_DP_size = tmp_total_elements_to_reduce = tmp_dim3_grid.x;

        if(tmp_total_elements_to_reduce == 0u)
        {
            PRINT_FORMAT("%s: ERROR: No elements to reduce." NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        // |END| Compute dimension reduce data batch. |END|
        
        // Allocating neurons reduce summation dim3 grid.
        struct dim3 *tmp_ptr_array_dim3_grid_threads_DP(static_cast<struct dim3*>(malloc(tmp_total_elements_to_reduce * sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_grid_threads_DP == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_dim3_grid_threads_DP,
                    0,
                    tmp_total_elements_to_reduce * sizeof(struct dim3));
        this->ptr_array_dim3_grid_reduce_threads_DP = tmp_ptr_array_dim3_grid_threads_DP;
        // |END| Allocating neurons reduce summation dim3 grid. |END|
            
        // Allocating neurons reduce summation dim3 block.
        struct dim3 *tmp_ptr_array_dim3_block_threads_DP(static_cast<struct dim3*>(malloc(tmp_total_elements_to_reduce * sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_block_threads_DP == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_dim3_block_threads_DP,
                    0,
                    tmp_total_elements_to_reduce * sizeof(struct dim3));
        this->ptr_array_dim3_block_reduce_threads_DP = tmp_ptr_array_dim3_block_threads_DP;
        // |END| Allocating neurons reduce summation dim3 block. |END|
        
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = this->number_threads;

        // Loop to reduce "number of elements" to one at the end.
        do
        {
            // Compute remaining results to reduce.
            tmp_ptr_CUDA_Device->Grid_Block_Reduce_Dynamic_Parallelisme(tmp_total_elements_to_reduce,
                                                                                                                tmp_ptr_CUDA_Device->Get__Maximum_Blocks_Per_Multiprocessor(),
                                                                                                                tmp_ptr_array_dim3_grid_threads_DP[tmp_index_dim3],
                                                                                                                tmp_ptr_array_dim3_block_threads_DP[tmp_index_dim3]);

            // Get the remaining results to reduce.
            tmp_total_elements_to_reduce = tmp_ptr_array_dim3_grid_threads_DP[tmp_index_dim3].x;

            // Increment index to dim3.
            ++tmp_index_dim3;
        } while(tmp_total_elements_to_reduce != 1u);
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate_Reduce_Cost(void)
{
    if(this->ptr_array_reduce_loss_values == nullptr)
    {
        if(this->total_reduce_batch_size == 0u)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory! Reduce size equal zero." NEW_LINE, __FUNCTION__);

            return(false);
        }

        // Allocating reduce number loss.
        size_t *tmp_ptr_array_reduce_number_loss(new size_t[this->total_reduce_batch_size]);
        if(tmp_ptr_array_reduce_number_loss == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_reduce_number_loss,
                    0,
                    this->total_reduce_batch_size * sizeof(size_t));
        this->ptr_array_reduce_number_loss = tmp_ptr_array_reduce_number_loss;
        // |END| Allocating reduce number loss. |END|
        
        // Allocating reduce bit fail values.
        size_t *tmp_ptr_array_reduce_bit_fail_values(new size_t[this->total_reduce_batch_size]);
        if(tmp_ptr_array_reduce_bit_fail_values == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_reduce_bit_fail_values,
                    0,
                    this->total_reduce_batch_size * sizeof(size_t));
        this->ptr_array_reduce_bit_fail_values = tmp_ptr_array_reduce_bit_fail_values;
        // |END| Allocating reduce bit fail values. |END|
        
        // Allocating reduce loss values.
        T_ *tmp_ptr_array_reduce_loss_values(new T_[this->total_reduce_batch_size]);
        if(tmp_ptr_array_reduce_loss_values == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_reduce_loss_values,
                    0,
                    this->total_reduce_batch_size * sizeof(T_));
        this->ptr_array_reduce_loss_values = tmp_ptr_array_reduce_loss_values;
        // |END| Allocating reduce loss values.. |END|
        
        // Allocating reduce accuracy values.
        if((this->ptr_array_reduce_accuracy_values[0u] = new T_[this->total_reduce_batch_size]) == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     this->total_reduce_batch_size * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        else
        {
            memset(this->ptr_array_reduce_accuracy_values[0u],
                        0,
                        this->total_reduce_batch_size * sizeof(T_));
        }
        
        if((this->ptr_array_reduce_accuracy_values[1u] = new T_[this->total_reduce_batch_size]) == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     this->total_reduce_batch_size * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        else
        {
            memset(this->ptr_array_reduce_accuracy_values[1u],
                        0,
                        this->total_reduce_batch_size * sizeof(T_));
        }
        
        if((this->ptr_array_reduce_accuracy_values[2u] = new T_[this->total_reduce_batch_size]) == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     this->total_reduce_batch_size * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        else
        {
            memset(this->ptr_array_reduce_accuracy_values[2u],
                        0,
                        this->total_reduce_batch_size * sizeof(T_));
        }
        
        if((this->ptr_array_reduce_accuracy_values[3u] = new T_[this->total_reduce_batch_size]) == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     this->total_reduce_batch_size * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        else
        {
            memset(this->ptr_array_reduce_accuracy_values[3u],
                        0,
                        this->total_reduce_batch_size * sizeof(T_));
        }
        
        if((this->ptr_array_reduce_accuracy_values[4u] = new T_[this->total_reduce_batch_size]) == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     this->total_reduce_batch_size * sizeof(T_),
                                     __LINE__);

            return(false);
        }
        else
        {
            memset(this->ptr_array_reduce_accuracy_values[4u],
                        0,
                        this->total_reduce_batch_size * sizeof(T_));
        }
        // |END| Allocating reduce accuracy values.. |END|
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Neuron_Units(void)
{
    size_t tmp_number_neuron_units,
              i;
    
    if(this->total_neuron_units != 0_zu)
    {
        struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct CUDA_Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct CUDA_Neuron *tmp_ptr_array_neuron_units(new struct CUDA_Neuron[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }

        // Allocating neurons first index.
        size_t *tmp_ptr_array_neuron_units_first_connection_index(new size_t[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_first_connection_index == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<size_t>(this->total_neuron_units,
                                          tmp_ptr_array_neuron_units_first_connection_index,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons first index. |END|
            
        // Allocating neurons last index.
        size_t *tmp_ptr_array_neuron_units_last_connection_index(new size_t[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_last_connection_index == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<size_t>(this->total_neuron_units,
                                          tmp_ptr_array_neuron_units_last_connection_index,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons last index. |END|
            
        // Allocating neurons number connections.
        size_t *tmp_ptr_array_neuron_units_number_connections(new size_t[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_number_connections == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<size_t>(this->total_neuron_units,
                                          tmp_ptr_array_neuron_units_number_connections,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons number connections. |END|
        
        // Allocating neuron unit(s) summation(s).
        T_ *tmp_ptr_array_neuron_units_summations(new T_[this->batch_size * this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_summations == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->batch_size * this->total_neuron_units,
                            tmp_ptr_array_neuron_units_summations,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neuron unit(s) summation(s). |END|
            
        // Allocating neurons activation steepness.
        T_ *tmp_ptr_array_neuron_units_activation_steepness(new T_[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_activation_steepness == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        Memory::Fill_1D<T_>(this->total_neuron_units,
                                                             tmp_ptr_array_neuron_units_activation_steepness,
                                                             1_T,
                                                             this->ptr_array_dim3_grid + 3,
                                                             this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons activation steepness. |END|
        
        // Allocating neuron unit(s) value(s).
        T_ *tmp_ptr_array_neuron_units_values(new T_[this->batch_size * this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_values == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->batch_size * this->total_neuron_units,
                            tmp_ptr_array_neuron_units_values,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neuron unit(s) value(s). |END|
        
        // Allocating neuron unit(s) error(s).
        T_ *tmp_ptr_array_neuron_units_errors(new T_[this->batch_size * this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_errors == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->batch_size * this->total_neuron_units,
                            tmp_ptr_array_neuron_units_errors,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neuron unit(s) error(s). |END|
        
        // Allocating neurons activation function.
        enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS *tmp_ptr_array_neuron_units_type_activation_function(new enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_type_activation_function == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS>(this->total_neuron_units,
                                                                                                                              tmp_ptr_array_neuron_units_type_activation_function,
                                                                                                                              this->ptr_array_dim3_grid + 3,
                                                                                                                              this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons activation function. |END|
        
        // Allocating neurons grid connections.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_grid_connections(static_cast<struct dim3*>(malloc(this->total_neuron_units * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_grid_connections == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_dim3_grid_connections,
                    0,
                    this->total_neuron_units * sizeof(struct dim3));
        // |END| Allocating neurons grid connections. |END|
            
        // Allocating neurons block connections.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_block_connections(static_cast<struct dim3*>(malloc(this->total_neuron_units * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_block_connections == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_dim3_block_connections,
                    0,
                    this->total_neuron_units * sizeof(struct dim3));
        // |END| Allocating neurons block connections. |END|

        // Assign neurons variable.
        this->ptr_array_neuron_units_first_forward_connection_index = tmp_ptr_array_neuron_units_first_connection_index;
        this->ptr_array_neuron_units_last_forward_connection_index = tmp_ptr_array_neuron_units_last_connection_index;
        this->ptr_array_neuron_units_number_forward_connections = tmp_ptr_array_neuron_units_number_connections;

        this->ptr_array_neuron_units_summations = tmp_ptr_array_neuron_units_summations;
        this->ptr_array_neuron_units_activation_steepness = tmp_ptr_array_neuron_units_activation_steepness;
        this->ptr_array_neuron_units_values = tmp_ptr_array_neuron_units_values;
        this->ptr_array_neuron_units_errors = tmp_ptr_array_neuron_units_errors;

        this->ptr_array_neuron_units_type_activation_function = tmp_ptr_array_neuron_units_type_activation_function;

        this->ptr_array_neuron_units_dim3_grid_connections = tmp_ptr_array_neuron_units_dim3_grid_connections;
        this->ptr_array_neuron_units_dim3_block_connections = tmp_ptr_array_neuron_units_dim3_block_connections;
        // |END| Assign neurons variable. |END|
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            tmp_ptr_layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;

            if((tmp_number_neuron_units = *tmp_ptr_layer_it->ptr_number_neurons) != 0u)
            {
                // Assign neurons variable.
                for(i = 0u; i != tmp_number_neuron_units; ++i)
                {
                    tmp_ptr_array_neuron_units[i].ptr_first_forward_connection_index = tmp_ptr_array_neuron_units_first_connection_index++;
                    tmp_ptr_array_neuron_units[i].ptr_last_forward_connection_index = tmp_ptr_array_neuron_units_last_connection_index++;
                    tmp_ptr_array_neuron_units[i].ptr_number_forward_connections = tmp_ptr_array_neuron_units_number_connections++;

                    tmp_ptr_array_neuron_units[i].ptr_array_summations = tmp_ptr_array_neuron_units_summations++;
                    tmp_ptr_array_neuron_units[i].ptr_activation_steepness = tmp_ptr_array_neuron_units_activation_steepness++;
                    tmp_ptr_array_neuron_units[i].ptr_array_values = tmp_ptr_array_neuron_units_values++;
                    tmp_ptr_array_neuron_units[i].ptr_array_errors = tmp_ptr_array_neuron_units_errors++;
                    
                    tmp_ptr_array_neuron_units[i].ptr_type_activation_function = tmp_ptr_array_neuron_units_type_activation_function++;

                    tmp_ptr_array_neuron_units[i].ptr_dim3_grid_connections = tmp_ptr_array_neuron_units_dim3_grid_connections++;
                    tmp_ptr_array_neuron_units[i].ptr_dim3_block_connections = tmp_ptr_array_neuron_units_dim3_block_connections++;
                }
                // |END| Assign neurons variable. |END|
                
                tmp_ptr_array_neuron_units_summations += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_values += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_errors += (this->batch_size - 1u) * tmp_number_neuron_units;

                tmp_ptr_array_neuron_units += tmp_number_neuron_units;
            }

            tmp_ptr_layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
        }

        this->total_neuron_units_allocated = this->total_neuron_units;
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Neurons_Reduce_Summation(void)
{
    size_t tmp_neurons_reduce_summation_size_so_far,
                      tmp_total_elements_to_reduce,
                      tmp_layer_reduce_summation_size,
                      tmp_number_neurons_in_layer,
                      tmp_index_dim3;
    
    if(this->total_neuron_units_allocated != 0u && this->ptr_array_neuron_units_reduce_summation_size == nullptr)
    {
        // ONLY FOR DENSE LAYER.
        // TODO: Make shortcut layer compatible.
        struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct CUDA_Layer *tmp_ptr_layer_it;
        
        struct CUDA_Neuron const *tmp_ptr_last_neuron_unit;
        struct CUDA_Neuron *tmp_ptr_neuron_unit_it;
        
        class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                            tmp_dim3_block;

        // COMPUTE REDUCE SUMMATION SIZE.
        // Allocating neurons reduce summation size.
        size_t *tmp_ptr_array_neuron_units_reduce_summation_size(new size_t[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_reduce_summation_size == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        Zero_1D<size_t>(this->total_neuron_units_allocated,
                                          tmp_ptr_array_neuron_units_reduce_summation_size,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons reduce summation size. |END|
        
        // Compute dimension reduce summation.
        this->ptr_array_neuron_units_reduce_summation_size = tmp_ptr_array_neuron_units_reduce_summation_size;
        
        for(tmp_neurons_reduce_summation_size_so_far = 0u,
            tmp_ptr_neuron_unit_it = this->ptr_array_layers->ptr_array_neuron_units,
            tmp_ptr_last_neuron_unit = tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                                      ++tmp_ptr_array_neuron_units_reduce_summation_size)
        {
            // Number elements to reduce equal number of connections from the neuron.
            tmp_total_elements_to_reduce = *tmp_ptr_neuron_unit_it->ptr_number_forward_connections;
            
            // If is not the bias. (The bias have no elements to reduce.)
            if(tmp_total_elements_to_reduce != 0u)
            {
                // Dimension required to reduce the number of elements.
                tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                      0u,
                                                                                                      tmp_dim3_grid,
                                                                                                      tmp_dim3_block);
            
                // Get remaining elements to reduce.
                tmp_total_elements_to_reduce = tmp_dim3_grid.x;
            }

            // Maximum remaining elements to reduce.
            *tmp_ptr_array_neuron_units_reduce_summation_size = tmp_total_elements_to_reduce;

            // Assign a pointer to the maximum reduce summation size of that neuron.
            tmp_ptr_neuron_unit_it->ptr_reduce_summation_size = tmp_ptr_array_neuron_units_reduce_summation_size;

            // Summation of the total maximum number of summation result.
            tmp_neurons_reduce_summation_size_so_far += tmp_total_elements_to_reduce;
        }

        this->neurons_total_reduce_summation_size = tmp_neurons_reduce_summation_size_so_far;

        if(tmp_neurons_reduce_summation_size_so_far == 0u)
        {
            PRINT_FORMAT("%s: ERROR: No elements to reduce." NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        // |END| Compute dimension reduce summation. |END|
        // |END| COMPUTE REDUCE SUMMATION SIZE. |END|
        
        // COMPUTE DIMENSION REDUCE SUMMATION.
        // Allocating neurons reduce summation.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        T_ **tmp_ptr_array_2D_neurons_position_reduce_summation_array(new T_*[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_2D_neurons_position_reduce_summation_array == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_*>(this->total_neuron_units_allocated,
                             tmp_ptr_array_2D_neurons_position_reduce_summation_array,
                             this->ptr_array_dim3_grid + 3,
                             this->ptr_array_dim3_block + 3);

        T_ *tmp_ptr_array_neuron_units_reduce_summation_results(new T_[this->batch_size * tmp_neurons_reduce_summation_size_so_far]);
        if(tmp_ptr_array_neuron_units_reduce_summation_results == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_reduce_summation_results,
                    0,
                    this->batch_size * tmp_neurons_reduce_summation_size_so_far * sizeof(T_));
        // |END| Allocating neurons reduce summation. |END|
        
        // Allocating neurons reduce summation dim3 grid.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_grid_summation(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_summation_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_grid_summation == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_dim3_grid_summation,
                    0,
                    tmp_neurons_reduce_summation_size_so_far * sizeof(struct dim3));
        // |END| Allocating neurons reduce summation dim3 grid. |END|
            
        // Allocating neurons reduce summation dim3 block.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_block_summation(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_summation_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_block_summation == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_dim3_block_summation,
                    0,
                    tmp_neurons_reduce_summation_size_so_far * sizeof(struct dim3));
        // |END| Allocating neurons reduce summation dim3 block. |END|
            
        // Assign global array.
        this->ptr_array_2D_neurons_reduce_summation = tmp_ptr_array_2D_neurons_position_reduce_summation_array;
        this->ptr_array_neuron_units_dim3_grid_reduce_summation = tmp_ptr_array_neuron_units_dim3_grid_summation;
        this->ptr_array_neuron_units_dim3_block_reduce_summation = tmp_ptr_array_neuron_units_dim3_block_summation;
        // |END| Assign global array. |END|
        
        // Loop through each layers.
        for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            // Get neurons array from that layer.
            tmp_ptr_neuron_unit_it = tmp_ptr_layer_it->ptr_array_neuron_units;

            // Get the reduce summation size of each neurons in that layer.
            tmp_layer_reduce_summation_size = *tmp_ptr_neuron_unit_it->ptr_reduce_summation_size;
            
            // Get the number of neurons in layer.
            tmp_number_neurons_in_layer = *tmp_ptr_layer_it->ptr_number_neurons;
            
            // Loop through each neurons in the layer.
            for(tmp_ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_last_neuron_unit; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                   ++tmp_ptr_array_2D_neurons_position_reduce_summation_array)
            {
                // Result.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_reduce_summation_array = tmp_ptr_array_neuron_units_reduce_summation_results;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_reduce_summation = tmp_ptr_array_2D_neurons_position_reduce_summation_array;
                // |END| Result. |END|
                
                // Number elements to reduce equal number of connections from the neuron.
                tmp_total_elements_to_reduce = *tmp_ptr_neuron_unit_it->ptr_number_forward_connections;

                // If is not the bias. (The bias have no elements to reduce.)
                if(tmp_total_elements_to_reduce != 0u)
                {
                    // Assign dim3 grid to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation = tmp_ptr_array_neuron_units_dim3_grid_summation++;
                    // Assign dim3 block to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation = tmp_ptr_array_neuron_units_dim3_block_summation++;

                    // Initialize index to zero.
                    tmp_index_dim3 = 0u;

                    // Loop to reduce "number of elements" to one at the end.
                    do
                    {
                        // Compute remaining results to reduce.
                        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                                0u,
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)],
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)]);

                        // Get the remaining results to reduce.
                        tmp_total_elements_to_reduce = tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)].x;

                        // Increment index to dim3.
                        ++tmp_index_dim3;
                    } while(tmp_total_elements_to_reduce != 1u);
                    // |END| dim3. |END|

                    // Increment the begining results by the layer reduce summation size.
                    tmp_ptr_array_neuron_units_reduce_summation_results += tmp_layer_reduce_summation_size;
                }
            }
                
            // If some elements need to be reduce in the layer.
            if(tmp_layer_reduce_summation_size != 0u)
            {
                // Increment pointer by (number of neurons in layer minus bias) times (layer reduce summation size minus one).
                tmp_ptr_array_neuron_units_dim3_grid_summation += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_summation_size - 1u);
                tmp_ptr_array_neuron_units_dim3_block_summation += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_summation_size - 1u);
            }
        }
        // |END| COMPUTE DIMENSION REDUCE SUMMATION. |END|
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Neurons_Reduce_Error(void)
{
    size_t tmp_neurons_reduce_error_size_so_far,
                      tmp_total_elements_to_reduce_layer,
                      tmp_total_elements_to_reduce,
                      tmp_layer_reduce_error_size,
                      tmp_number_neurons_in_layer,
                      tmp_index_dim3;
    
    if(this->total_neuron_units_allocated != 0u && this->ptr_array_neuron_units_reduce_error_size == nullptr)
    {
        // ONLY FOR DENSE LAYER.
        // TODO: Make shortcut layer compatible.
        struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct CUDA_Layer *tmp_ptr_next_layer,
                                               *tmp_ptr_layer_it;
        
        struct CUDA_Neuron const *tmp_ptr_last_neuron_unit;
        struct CUDA_Neuron *tmp_ptr_neuron_unit_it;
        
        class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                            tmp_dim3_block;

        // COMPUTE REDUCE ERROR SIZE.
        // Allocating neurons reduce error size.
        size_t *tmp_ptr_array_neuron_units_reduce_error_size(new size_t[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_reduce_error_size == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        Zero_1D<size_t>(this->total_neuron_units_allocated,
                                          tmp_ptr_array_neuron_units_reduce_error_size,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons reduce error size. |END|
        
        // Compute dimension reduce error.
        this->ptr_array_neuron_units_reduce_error_size = tmp_ptr_array_neuron_units_reduce_error_size;
        
        // Loop through each layers.
        for(tmp_neurons_reduce_error_size_so_far = 0u,
            tmp_ptr_layer_it = this->ptr_array_layers,
            tmp_ptr_next_layer = tmp_ptr_layer_it + 1; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                                                                                   ++tmp_ptr_next_layer)
        {
            if(tmp_ptr_layer_it == this->ptr_array_layers // Input layer.
                ||
                tmp_ptr_layer_it == this->ptr_last_layer - 1) // Output layer.
            { tmp_total_elements_to_reduce_layer = 0u; }
            else
            // Number elements to reduce equal number of connections to the neuron.
            { tmp_total_elements_to_reduce_layer = *tmp_ptr_next_layer->ptr_number_neurons - 1u; } // Subtract bias.
            
            for(tmp_ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_last_neuron_unit,
                tmp_ptr_neuron_unit_it = tmp_ptr_layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                   ++tmp_ptr_array_neuron_units_reduce_error_size)
            {
                // If no elements to reduce or the neuron is a bias.
                if(tmp_total_elements_to_reduce_layer == 0u || tmp_ptr_neuron_unit_it == tmp_ptr_last_neuron_unit - 1)
                { tmp_total_elements_to_reduce = 0u; }
                else
                {
                    // Number elements to reduce equal number of connections to the neuron.
                    tmp_total_elements_to_reduce = tmp_total_elements_to_reduce_layer;

                    // Dimension required to reduce the number of elements.
                    tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                          0u,
                                                                                                          tmp_dim3_grid,
                                                                                                          tmp_dim3_block);
                
                    // Get remaining elements to reduce.
                    tmp_total_elements_to_reduce = tmp_dim3_grid.x;
                }
                
                // Maximum remaining elements to reduce.
                *tmp_ptr_array_neuron_units_reduce_error_size = tmp_total_elements_to_reduce;

                // Assign a pointer to the maximum reduce error size of that neuron.
                tmp_ptr_neuron_unit_it->ptr_reduce_error_size = tmp_ptr_array_neuron_units_reduce_error_size;

                // Summation of the total maximum number of error result.
                tmp_neurons_reduce_error_size_so_far += tmp_total_elements_to_reduce;
            }
        }

        this->neurons_total_reduce_error_size = tmp_neurons_reduce_error_size_so_far;

        if(tmp_neurons_reduce_error_size_so_far == 0u)
        {
            PRINT_FORMAT("%s: ERROR: No elements to reduce." NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        // |END| Compute dimension reduce error. |END|
        // |END| COMPUTE REDUCE ERROR SIZE. |END|
        
        // COMPUTE DIMENSION REDUCE ERROR.
        // Allocating neurons reduce error.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        T_ **tmp_ptr_array_2D_neurons_position_reduce_error_array(new T_*[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_2D_neurons_position_reduce_error_array == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_*>(this->total_neuron_units_allocated,
                              tmp_ptr_array_2D_neurons_position_reduce_error_array,
                              this->ptr_array_dim3_grid + 3,
                              this->ptr_array_dim3_block + 3);

        T_ *tmp_ptr_array_neuron_units_reduce_error_results(new T_[this->batch_size * tmp_neurons_reduce_error_size_so_far]);
        if(tmp_ptr_array_neuron_units_reduce_error_results == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_reduce_error_results,
                    0,
                    this->batch_size * tmp_neurons_reduce_error_size_so_far * sizeof(T_));
        // |END| Allocating neurons reduce error. |END|
        
        // Allocating neurons reduce error dim3 grid.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_grid_error(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_error_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_grid_error == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_dim3_grid_error,
                    0,
                    tmp_neurons_reduce_error_size_so_far * sizeof(struct dim3));
        // |END| Allocating neurons reduce error dim3 grid. |END|
            
        // Allocating neurons reduce error dim3 block.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_block_error(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_error_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_block_error == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_dim3_block_error,
                    0,
                    tmp_neurons_reduce_error_size_so_far * sizeof(struct dim3));
        // |END| Allocating neurons reduce error dim3 block. |END|
            
        // Assign global array.
        this->ptr_array_2D_neurons_reduce_error = tmp_ptr_array_2D_neurons_position_reduce_error_array;
        this->ptr_array_neuron_units_dim3_grid_reduce_error = tmp_ptr_array_neuron_units_dim3_grid_error;
        this->ptr_array_neuron_units_dim3_block_reduce_error = tmp_ptr_array_neuron_units_dim3_block_error;
        // |END| Assign global array. |END|
        
        // Loop through each layers.
        for(tmp_ptr_layer_it = this->ptr_array_layers,
            tmp_ptr_next_layer = tmp_ptr_layer_it + 1; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                                                                                   ++tmp_ptr_next_layer)
        {
            // Get neurons array from that layer.
            tmp_ptr_neuron_unit_it = tmp_ptr_layer_it->ptr_array_neuron_units;

            // Get the reduce error size of each neurons in that layer.
            tmp_layer_reduce_error_size = *tmp_ptr_neuron_unit_it->ptr_reduce_error_size;

            // Get the number of neurons in layer.
            tmp_number_neurons_in_layer = *tmp_ptr_layer_it->ptr_number_neurons;

            if(tmp_ptr_layer_it == this->ptr_array_layers // Input layer.
                ||
                tmp_ptr_layer_it == this->ptr_last_layer - 1) // Output layer.
            { tmp_total_elements_to_reduce_layer = 0u; }
            else
            // Number elements to reduce equal number of connections to the neuron.
            { tmp_total_elements_to_reduce_layer = *tmp_ptr_next_layer->ptr_number_neurons - 1u; } // Subtract bias.
            
            // Loop through each neurons in the layer.
            for(tmp_ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_last_neuron_unit; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                   ++tmp_ptr_array_2D_neurons_position_reduce_error_array)
            {
                // Result.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_reduce_error_array = tmp_ptr_array_neuron_units_reduce_error_results;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_reduce_error = tmp_ptr_array_2D_neurons_position_reduce_error_array;
                // |END| Result. |END|
                
                // If we have elements to reduce and the neuron is not a bias.
                if(tmp_total_elements_to_reduce_layer != 0u && tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit - 1)
                {
                    // Number elements to reduce equal number of connections to the neuron.
                    tmp_total_elements_to_reduce = tmp_total_elements_to_reduce_layer;

                    // Assign dim3 grid to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error = tmp_ptr_array_neuron_units_dim3_grid_error++;
                    // Assign dim3 block to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error = tmp_ptr_array_neuron_units_dim3_block_error++;

                    // Initialize index to zero.
                    tmp_index_dim3 = 0u;

                    // Loop to reduce "number of elements" to one at the end.
                    do
                    {
                        // Compute remaining results to reduce.
                        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                                0u,
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)],
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)]);

                        // Get the remaining results to reduce.
                        tmp_total_elements_to_reduce = tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)].x;

                        // Increment index to dim3.
                        ++tmp_index_dim3;
                    } while(tmp_total_elements_to_reduce != 1u);
                    // |END| dim3. |END|

                    // Increment the begining results by the layer reduce error size.
                    tmp_ptr_array_neuron_units_reduce_error_results += tmp_layer_reduce_error_size;
                }
            }
                
            // If some elements need to be reduce in the layer.
            if(tmp_layer_reduce_error_size != 0u)
            {
                // Increment pointer by (number of neurons in layer minus bias) times (layer reduce error size minus one).
                tmp_ptr_array_neuron_units_dim3_grid_error += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_error_size - 1u);
                tmp_ptr_array_neuron_units_dim3_block_error += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_error_size - 1u);
            }
        }
        // |END| COMPUTE DIMENSION REDUCE ERROR. |END|
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Neurons_Reduce_Batch_Normalization(void)
{
    size_t tmp_neurons_reduce_batch_size_so_far,
                      tmp_total_elements_to_reduce,
                      tmp_layer_reduce_batch_size,
                      tmp_number_neurons_in_layer,
                      tmp_index_dim3;
    
    if(this->total_neuron_units_allocated != 0u && this->ptr_array_neuron_units_reduce_batch_size == nullptr)
    {
        // ONLY FOR DENSE LAYER.
        // TODO: Make shortcut layer compatible.
        struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct CUDA_Layer *tmp_ptr_layer_it;
        
        struct CUDA_Neuron const *tmp_ptr_last_neuron_unit;
        struct CUDA_Neuron *tmp_ptr_neuron_unit_it;
        
        class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                            tmp_dim3_block;

        // COMPUTE REDUCE BATCH SIZE.
        // Allocating neurons reduce batch size.
        size_t *tmp_ptr_array_neuron_units_reduce_batch_size(new size_t[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_reduce_batch_size == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<size_t>(this->total_neuron_units_allocated,
                                          tmp_ptr_array_neuron_units_reduce_batch_size,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        this->ptr_array_neuron_units_reduce_batch_size = tmp_ptr_array_neuron_units_reduce_batch_size;
        // |END| Allocating neurons reduce batch size. |END|
        
        // Compute dimension reduce batch.
        for(tmp_neurons_reduce_batch_size_so_far = 0u,
            tmp_ptr_neuron_unit_it = this->ptr_array_layers->ptr_array_neuron_units,
            tmp_ptr_last_neuron_unit = tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                                       ++tmp_ptr_array_neuron_units_reduce_batch_size)
        {
            // Number elements to reduce equal the size of batch.
            tmp_total_elements_to_reduce = this->batch_size;

            // If the neuron is a bias. Number of elements to reduce equal zero.
            if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections == 0u)
            { tmp_total_elements_to_reduce = 0u; }
            
            // If is not the bias. (The bias have no elements to reduce.)
            if(tmp_total_elements_to_reduce != 0u)
            {
                // Dimension required to reduce the number of elements.
                tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                      0u,
                                                                                                      tmp_dim3_grid,
                                                                                                      tmp_dim3_block);
            
                // Get remaining elements to reduce.
                tmp_total_elements_to_reduce = tmp_dim3_grid.x;
            }

            // Maximum remaining elements to reduce.
            *tmp_ptr_array_neuron_units_reduce_batch_size = tmp_total_elements_to_reduce;

            // Assign a pointer to the maximum reduce norm size of that neuron.
            tmp_ptr_neuron_unit_it->ptr_reduce_batch_size = tmp_ptr_array_neuron_units_reduce_batch_size;

            // Summation of the total maximum number of batch result.
            tmp_neurons_reduce_batch_size_so_far += tmp_total_elements_to_reduce;
        }

        if(tmp_neurons_reduce_batch_size_so_far == 0u)
        {
            PRINT_FORMAT("%s: ERROR: No elements to reduce." NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        // |END| Compute dimension reduce batch. |END|
        // |END| COMPUTE REDUCE BATCH SIZE. |END|
        
        // COMPUTE DIMENSION REDUCE BATCH.
        // Allocating neurons reduce batch mean.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        T_ **tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array(new T_*[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_*>(this->total_neuron_units_allocated,
                              tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array,
                              this->ptr_array_dim3_grid + 3,
                              this->ptr_array_dim3_block + 3);
        this->ptr_array_2D_neurons_reduce_batch_mean = tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array;

        T_ *tmp_ptr_array_neuron_units_reduce_batch_mean_results(new T_[tmp_neurons_reduce_batch_size_so_far]);
        if(tmp_ptr_array_neuron_units_reduce_batch_mean_results == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_reduce_batch_mean_results,
                    0,
                    tmp_neurons_reduce_batch_size_so_far * sizeof(T_));
        // |END| Allocating neurons reduce batch mean. |END|
        
        // Allocating neurons reduce batch variance.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        T_ **tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array(new T_*[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_*>(this->total_neuron_units_allocated,
                              tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array,
                              this->ptr_array_dim3_grid + 3,
                              this->ptr_array_dim3_block + 3);
        this->ptr_array_2D_neurons_reduce_batch_variance = tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array;

        T_ *tmp_ptr_array_neuron_units_reduce_batch_variance_results(new T_[tmp_neurons_reduce_batch_size_so_far]);
        if(tmp_ptr_array_neuron_units_reduce_batch_variance_results == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_reduce_batch_variance_results,
                    0,
                    tmp_neurons_reduce_batch_size_so_far * sizeof(T_));
        // |END| Allocating neurons reduce batch variance. |END|
        
        // Allocating neurons reduce batch dim3 grid.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_grid_batch(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_batch_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_grid_batch == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_dim3_grid_batch,
                    0,
                    tmp_neurons_reduce_batch_size_so_far * sizeof(struct dim3));
        this->ptr_array_neuron_units_dim3_grid_reduce_batch = tmp_ptr_array_neuron_units_dim3_grid_batch;
        // |END| Allocating neurons reduce batch dim3 grid. |END|
            
        // Allocating neurons reduce batch dim3 block.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_block_batch(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_batch_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_block_batch == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_dim3_block_batch,
                    0,
                    tmp_neurons_reduce_batch_size_so_far * sizeof(struct dim3));
        this->ptr_array_neuron_units_dim3_block_reduce_batch = tmp_ptr_array_neuron_units_dim3_block_batch;
        // |END| Allocating neurons reduce batch dim3 block. |END|
        
        // Loop through each layers.
        for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            // Get neurons array from that layer.
            tmp_ptr_neuron_unit_it = tmp_ptr_layer_it->ptr_array_neuron_units;

            // Get the reduce batch size of each neurons in that layer.
            tmp_layer_reduce_batch_size = *tmp_ptr_neuron_unit_it->ptr_reduce_batch_size;
            
            // Get the number of neurons in layer.
            tmp_number_neurons_in_layer = *tmp_ptr_layer_it->ptr_number_neurons;
            
            // Loop through each neurons in the layer.
            for(tmp_ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_last_neuron_unit; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                   ++tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array,
                                                                                                                                                                   ++tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array)
            {
                // Result.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array = tmp_ptr_array_neuron_units_reduce_batch_mean_results;
                *tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array = tmp_ptr_array_neuron_units_reduce_batch_variance_results;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_reduce_mean = tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array;
                tmp_ptr_neuron_unit_it->ptr_array_reduce_variance = tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array;
                // |END| Result. |END|
                
                // Number elements to reduce equal the size of batch
                tmp_total_elements_to_reduce = this->batch_size;
                
                // If the neuron is a bias. Number of elements to reduce equal zero.
                if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections == 0u)
                { tmp_total_elements_to_reduce = 0u; }
                
                // If is not the bias. (The bias have no elements to reduce.)
                if(tmp_total_elements_to_reduce != 0u)
                {
                    // Assign dim3 grid to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_threads = tmp_ptr_array_neuron_units_dim3_grid_batch++;
                    // Assign dim3 block to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_threads = tmp_ptr_array_neuron_units_dim3_block_batch++;

                    // Initialize index to zero.
                    tmp_index_dim3 = 0u;

                    // Loop to reduce "number of elements" to one at the end.
                    do
                    {
                        // Compute remaining results to reduce.
                        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                                0u,
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_threads[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)],
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_threads[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)]);

                        // Get the remaining results to reduce.
                        tmp_total_elements_to_reduce = tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_threads[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)].x;

                        // Increment index to dim3.
                        ++tmp_index_dim3;
                    } while(tmp_total_elements_to_reduce != 1u);
                    // |END| dim3. |END|

                    // Increment the begining results by the layer reduce batch size.
                    tmp_ptr_array_neuron_units_reduce_batch_mean_results += tmp_layer_reduce_batch_size;
                    tmp_ptr_array_neuron_units_reduce_batch_variance_results += tmp_layer_reduce_batch_size;
                }
            }
                
            // If some elements need to be reduce in the layer.
            if(tmp_layer_reduce_batch_size != 0u)
            {
                // Increment pointer by (number of neurons in layer minus bias) times (layer reduce batch size minus one).
                tmp_ptr_array_neuron_units_dim3_grid_batch += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_batch_size - 1u);
                tmp_ptr_array_neuron_units_dim3_block_batch += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_batch_size - 1u);
            }
        }
        // |END| COMPUTE DIMENSION REDUCE BATCH. |END|

        this->neurons_total_reduce_batch_size = tmp_neurons_reduce_batch_size_so_far;
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Neurons_Reduce_Norms(void)
{
    size_t tmp_neurons_reduce_norms_size_so_far,
                      tmp_total_elements_to_reduce,
                      tmp_layer_reduce_norms_size,
                      tmp_number_neurons_in_layer,
                      tmp_index_dim3;
    
    if(this->total_neuron_units_allocated != 0u && this->ptr_array_neuron_units_reduce_norms_size == nullptr)
    {
        // ONLY FOR DENSE LAYER.
        // TODO: Make shortcut layer compatible.
        struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct CUDA_Layer *tmp_ptr_layer_it;
            
        struct CUDA_Neuron const *tmp_ptr_last_neuron_unit;
        struct CUDA_Neuron *tmp_ptr_neuron_unit_it;
            
        class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                            tmp_dim3_block;

        // COMPUTE REDUCE NORMS SIZE.
        // Allocating neurons reduce norms size.
        size_t *tmp_ptr_array_neuron_units_reduce_norms_size(new size_t[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_reduce_norms_size == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<size_t>(this->total_neuron_units_allocated,
                                          tmp_ptr_array_neuron_units_reduce_norms_size,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons reduce norms size. |END|
        
        // Compute dimension reduce norms.
        this->ptr_array_neuron_units_reduce_norms_size = tmp_ptr_array_neuron_units_reduce_norms_size;
        
        for(tmp_neurons_reduce_norms_size_so_far = 0u,
            tmp_ptr_neuron_unit_it = this->ptr_array_layers->ptr_array_neuron_units,
            tmp_ptr_last_neuron_unit = tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
        {
            // Number elements to reduce equal number of connections from the neuron.
            tmp_total_elements_to_reduce = *tmp_ptr_neuron_unit_it->ptr_number_forward_connections;
            
            // If is not the bias. (The bias have no elements to reduce.)
            if(tmp_total_elements_to_reduce != 0u)
            {
                // Dimension required to reduce the number of elements.
                tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                      0u,
                                                                                                      tmp_dim3_grid,
                                                                                                      tmp_dim3_block);

                // Get remaining elements to reduce.
                tmp_total_elements_to_reduce = tmp_dim3_grid.x;
            }

            // Maximum remaining elements to reduce.
            *tmp_ptr_array_neuron_units_reduce_norms_size = tmp_total_elements_to_reduce;

            // Assign a pointer to the maximum reduce norm size of that neuron.
            tmp_ptr_neuron_unit_it->ptr_reduce_norms_size = tmp_ptr_array_neuron_units_reduce_norms_size++;

            // Summation of the total maximum number of norms result.
            tmp_neurons_reduce_norms_size_so_far += tmp_total_elements_to_reduce;
        }

        this->neurons_total_reduce_norms_size = tmp_neurons_reduce_norms_size_so_far;

        if(tmp_neurons_reduce_norms_size_so_far == 0u)
        {
            PRINT_FORMAT("%s: ERROR: No elements to reduce." NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
        // |END| Compute dimension reduce norms. |END|
        // |END| COMPUTE REDUCE NORMS SIZE. |END|
        
        // COMPUTE DIMENSION REDUCE NORMS.
        // Allocating neuroyed number neurons in layer.
        // "load" and "plus" technique is equivalent to the 2D array technique because both need to be at the size of "total_neuron_units_allocated"
        // in term of storage. But "load" and "plus" technique use the arithmetic power of coalescing threads in a warp.
        size_t *tmp_ptr_array_neuroyed_number_neurons_in_layer(new size_t[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuroyed_number_neurons_in_layer == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<size_t>(this->total_neuron_units_allocated,
                                          tmp_ptr_array_neuroyed_number_neurons_in_layer,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neuroyed number neurons in layer. |END|
        
        // Allocating neurons reduce norms.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        T_ **tmp_ptr_array_2D_neurons_position_reduce_norms_array(new T_*[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_2D_neurons_position_reduce_norms_array == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_*>(this->total_neuron_units_allocated,
                              tmp_ptr_array_2D_neurons_position_reduce_norms_array,
                              this->ptr_array_dim3_grid + 3,
                              this->ptr_array_dim3_block + 3);

        T_ *tmp_ptr_array_neuron_units_reduce_norms_results(new T_[tmp_neurons_reduce_norms_size_so_far]);
        if(tmp_ptr_array_neuron_units_reduce_norms_results == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_reduce_norms_results,
                    0,
                    tmp_neurons_reduce_norms_size_so_far * sizeof(T_));
        // |END| Allocating neurons reduce norms. |END|
        
        // Allocating neurons reduce norms dim3 grid.
        struct dim3 **tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms(static_cast<struct dim3**>(malloc(this->total_neuron_units_allocated * sizeof(struct dim3*))));
        if(tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms,
                        0,
                        this->total_neuron_units_allocated * sizeof(struct dim3*));

        struct dim3 *tmp_ptr_array_neuron_units_dim3_grid_reduce_norms(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_norms_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_grid_reduce_norms == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_dim3_grid_reduce_norms,
                        0,
                        tmp_neurons_reduce_norms_size_so_far * sizeof(struct dim3));
        // |END| Allocating neurons reduce norms dim3 grid. |END|
            
        // Allocating neurons reduce norms dim3 block.
        struct dim3 **tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms(static_cast<struct dim3**>(malloc(this->total_neuron_units_allocated * sizeof(struct dim3*))));
        if(tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms,
                        0,
                        this->total_neuron_units_allocated * sizeof(struct dim3*));

        struct dim3 *tmp_ptr_array_neuron_units_dim3_block_reduce_norms(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_norms_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_block_reduce_norms == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        memset(tmp_ptr_array_neuron_units_dim3_block_reduce_norms,
                        0,
                        tmp_neurons_reduce_norms_size_so_far * sizeof(struct dim3));
        // |END| Allocating neurons reduce norms dim3 block. |END|
        
        // Assign global array.
        this->ptr_array_neuroyed_number_neurons_in_layer = tmp_ptr_array_neuroyed_number_neurons_in_layer;
        this->ptr_array_2D_neurons_reduce_norms = tmp_ptr_array_2D_neurons_position_reduce_norms_array;
        this->ptr_array_2D_neurons_dim3_grid_reduce_norms = tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms;
        this->ptr_array_2D_neurons_dim3_block_reduce_norms = tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms;
        // |END| Assign global array. |END|
        
        // Loop through each layers.
        for(tmp_ptr_layer_it = this->ptr_array_layers; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            // Get neurons array from that layer.
            tmp_ptr_neuron_unit_it = tmp_ptr_layer_it->ptr_array_neuron_units;

            // Get the reduce norms size of each neurons in that layer.
            tmp_layer_reduce_norms_size = *tmp_ptr_neuron_unit_it->ptr_reduce_norms_size;
            
            // Get the number of neurons in layer.
            tmp_number_neurons_in_layer = *tmp_ptr_layer_it->ptr_number_neurons;

            // Loop through each neurons in the layer.
            for(tmp_ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_last_neuron_unit; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                    ++tmp_ptr_array_neuroyed_number_neurons_in_layer,
                                                                                                                                                                    ++tmp_ptr_array_2D_neurons_position_reduce_norms_array,
                                                                                                                                                                    ++tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms,
                                                                                                                                                                    ++tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms)
            {
                // Assign the number of neurons in the layer to the pointer.
                *tmp_ptr_array_neuroyed_number_neurons_in_layer = tmp_number_neurons_in_layer;
                
                // Result.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_reduce_norms_array = tmp_ptr_array_neuron_units_reduce_norms_results;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_reduce_norms = tmp_ptr_array_2D_neurons_position_reduce_norms_array;
                // |END| Result. |END|
                
                // Dim3 grid.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms = tmp_ptr_array_neuron_units_dim3_grid_reduce_norms;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_2D_dim3_grid_reduce_norms = tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms;
                // |END| Dim3 grid. |END|

                // Dim3 block.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms = tmp_ptr_array_neuron_units_dim3_block_reduce_norms;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_2D_dim3_block_reduce_norms = tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms;
                // |END| Dim3 block. |END|

                // Number elements to reduce equal number of connections from the neuron.
                tmp_total_elements_to_reduce = *tmp_ptr_neuron_unit_it->ptr_number_forward_connections;
                
                // If is not the bias. (The bias have no elements to reduce.)
                if(tmp_total_elements_to_reduce != 0u)
                {
                    // Initialize index to zero.
                    tmp_index_dim3 = 0u;

                    // Loop to reduce "number of elements" to one at the end.
                    do
                    {
                        // Compute remaining results to reduce.
                        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                                0u,
                                                                                                                tmp_ptr_array_neuron_units_dim3_grid_reduce_norms[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)],
                                                                                                                tmp_ptr_array_neuron_units_dim3_block_reduce_norms[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)]);

                        // Get the remaining results to reduce.
                        tmp_total_elements_to_reduce = tmp_ptr_array_neuron_units_dim3_grid_reduce_norms[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)].x;

                        // Increment index to dim3.
                        ++tmp_index_dim3;
                    } while(tmp_total_elements_to_reduce != 1u);
                    // |END| dim3. |END|
                    
                    // Increment the begining results by the layer reduce norms size.
                    tmp_ptr_array_neuron_units_reduce_norms_results += tmp_layer_reduce_norms_size;

                    // Increment the dim3 grid by one. (Access it by "iteration reduce" times "number neurons in layer minus bias".
                    ++tmp_ptr_array_neuron_units_dim3_grid_reduce_norms;

                    // Increment the dim3 grid by one. (Access it by "iteration reduce" times "number neurons in layer minus bias".
                    ++tmp_ptr_array_neuron_units_dim3_block_reduce_norms;
                }
            }
            
            // If some elements need to be reduce in the layer.
            if(tmp_layer_reduce_norms_size != 0u)
            {
                // Increment pointer by (number of neurons in layer minus bias) times (layer reduce summation size minus one).
                tmp_ptr_array_neuron_units_dim3_grid_reduce_norms += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_norms_size - 1u);
                tmp_ptr_array_neuron_units_dim3_block_reduce_norms += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_norms_size - 1u);
            }
        }
        // |END| COMPUTE DIMENSION REDUCE NORMS. |END|
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Normalized_Unit__Batch_Renormalization(void)
{
    if(this->total_neuron_units_allocated != 0u)
    {
        size_t tmp_number_neuron_units;

        T_ *tmp_ptr_array_parameters_scale_it(this->ptr_array_parameters + this->total_weights_allocated),
            *tmp_ptr_array_parameters_shift_it(this->ptr_array_parameters + this->total_weights_allocated + this->total_neuron_units_allocated),
        // TODO: Use only at training.
            *tmp_ptr_array_derivatives_parameters_scale_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated),
            *tmp_ptr_array_derivatives_parameters_shift_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated + this->total_neuron_units_allocated);
        
        struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct CUDA_Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct CUDA_Neuron const *tmp_ptr_last_neuron_unit;
        struct CUDA_Neuron *tmp_ptr_neuron_unit_it;
        
        // Allocating neuron unit(s) value(s) hat.
        T_ *tmp_ptr_array_neuron_units_values_hat(new T_[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_values_hat == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_values_hat,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neuron unit(s) value(s) hat. |END|
        
        // Allocating neuron unit(s) value(s) normalize.
        T_ *tmp_ptr_array_neuron_units_values_normalize(new T_[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_values_normalize == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_values_normalize,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neuron unit(s) value(s) normalize. |END|
        
        // Allocating neurons mean.
        T_ *tmp_ptr_array_neuron_units_mean_it(new T_[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_mean_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_mean_it,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neurons mean. |END|
        
        // Allocating neurons variance.
        T_ *tmp_ptr_array_neuron_units_variance_it(new T_[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_variance_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_variance_it,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neurons variance. |END|
        
        // Allocating neurons derivative mean.
        T_ *tmp_ptr_array_neuron_units_derivative_mean_it(new T_[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_derivative_mean_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_derivative_mean_it,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neurons derivative mean. |END|
        
        // Allocating neurons derivative variance.
        T_ *tmp_ptr_array_neuron_units_derivative_variance_it(new T_[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_derivative_variance_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_derivative_variance_it,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neurons derivative variance. |END|
        
        // Allocating neurons r correction.
        T_ *tmp_ptr_array_neuron_units_r_correction_it(new T_[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_r_correction_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_r_correction_it,
                            this->ptr_array_dim3_grid + 3,
                            this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons r correction. |END|
        
        // Allocating neurons d correction.
        T_ *tmp_ptr_array_neuron_units_d_correction_it(new T_[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_d_correction_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_d_correction_it,
                            this->ptr_array_dim3_grid + 3,
                            this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons d correction. |END|
        
        // Allocating neurons mean average.
        T_ *tmp_ptr_array_neuron_units_mean_average_it(new T_[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_mean_average_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_mean_average_it,
                            this->ptr_array_dim3_grid + 3,
                            this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons mean average. |END|
        
        // Allocating neurons variance average.
        T_ *tmp_ptr_array_neuron_units_variance_average_it(new T_[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_variance_average_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        Memory::Fill_1D<T_>(this->total_neuron_units_allocated,
                                                             tmp_ptr_array_neuron_units_variance_average_it,
                                                             1_T,
                                                             this->ptr_array_dim3_grid + 3,
                                                             this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons variance average. |END|
        
        this->ptr_array_normalized_batch_units_values_hats = tmp_ptr_array_neuron_units_values_hat;
        this->ptr_array_normalized_batch_units_values_normalizes = tmp_ptr_array_neuron_units_values_normalize;
        this->ptr_array_normalized_batch_units_scales = tmp_ptr_array_parameters_scale_it;
        this->ptr_array_normalized_batch_units_shifts = tmp_ptr_array_parameters_shift_it;
        this->ptr_array_normalized_batch_units_means = tmp_ptr_array_neuron_units_mean_it;
        this->ptr_array_normalized_batch_units_variances = tmp_ptr_array_neuron_units_variance_it;
        this->ptr_array_normalized_batch_units_derivatives_means = tmp_ptr_array_neuron_units_derivative_mean_it;
        this->ptr_array_normalized_batch_units_derivatives_variances = tmp_ptr_array_neuron_units_derivative_variance_it;
        this->ptr_array_normalized_batch_units_r_corrections = tmp_ptr_array_neuron_units_r_correction_it;
        this->ptr_array_normalized_batch_units_d_corrections = tmp_ptr_array_neuron_units_d_correction_it;
        this->ptr_array_normalized_batch_units_means_averages = tmp_ptr_array_neuron_units_mean_average_it;
        this->ptr_array_normalized_batch_units_variances_averages = tmp_ptr_array_neuron_units_variance_average_it;
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            if((tmp_number_neuron_units = *tmp_ptr_layer_it->ptr_number_neurons) != 0u)
            {
                for(tmp_ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_last_neuron_unit,
                    tmp_ptr_neuron_unit_it = tmp_ptr_layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_values_hat,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_values_normalize,
                                                                                                                                                                      ++tmp_ptr_array_parameters_scale_it,
                                                                                                                                                                      ++tmp_ptr_array_parameters_shift_it,
                                                                                                                                                                      ++tmp_ptr_array_derivatives_parameters_scale_it,
                                                                                                                                                                      ++tmp_ptr_array_derivatives_parameters_shift_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_r_correction_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_d_correction_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_mean_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_variance_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_derivative_mean_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_derivative_variance_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_mean_average_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_variance_average_it)
                {
                    tmp_ptr_neuron_unit_it->ptr_array_values_hats = tmp_ptr_array_neuron_units_values_hat;
                    tmp_ptr_neuron_unit_it->ptr_array_values_normalizes = tmp_ptr_array_neuron_units_values_normalize;
                    tmp_ptr_neuron_unit_it->ptr_scale = tmp_ptr_array_parameters_scale_it; *tmp_ptr_array_parameters_scale_it = 1_T;
                    tmp_ptr_neuron_unit_it->ptr_shift = tmp_ptr_array_parameters_shift_it;
                    tmp_ptr_neuron_unit_it->ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it;
                    tmp_ptr_neuron_unit_it->ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it;
                    tmp_ptr_neuron_unit_it->ptr_array_means = tmp_ptr_array_neuron_units_mean_it;
                    tmp_ptr_neuron_unit_it->ptr_array_variances = tmp_ptr_array_neuron_units_variance_it;
                    tmp_ptr_neuron_unit_it->ptr_array_derivatives_means = tmp_ptr_array_neuron_units_derivative_mean_it;
                    tmp_ptr_neuron_unit_it->ptr_array_derivatives_variances = tmp_ptr_array_neuron_units_derivative_variance_it;
                    tmp_ptr_neuron_unit_it->ptr_r_correction = tmp_ptr_array_neuron_units_r_correction_it;
                    tmp_ptr_neuron_unit_it->ptr_d_correction = tmp_ptr_array_neuron_units_d_correction_it;
                    tmp_ptr_neuron_unit_it->ptr_mean_average = tmp_ptr_array_neuron_units_mean_average_it;
                    tmp_ptr_neuron_unit_it->ptr_variance_average = tmp_ptr_array_neuron_units_variance_average_it;
                }

                tmp_ptr_array_neuron_units_values_hat += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_values_normalize += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_mean_it += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_variance_it += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_derivative_mean_it += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_derivative_variance_it += (this->batch_size - 1u) * tmp_number_neuron_units;
            }
        }
    }
    else { return(false); }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Neuron__Batch_Renormalization_Transpose(void)
{
    if(this->total_neuron_units_allocated != 0u)
    {
        struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct CUDA_Layer *tmp_ptr_layer_it(this->ptr_array_layers);

        struct CUDA_Neuron const *tmp_ptr_last_neuron_unit;
        struct CUDA_Neuron *tmp_ptr_neuron_unit_it;
        
        // Allocating neurons mean.
        T_ *tmp_ptr_array_neuron_units_transposed_mean_it(new T_[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_transposed_mean_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_transposed_mean_it,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neurons mean. |END|
        
        // Allocating neurons variance.
        T_ *tmp_ptr_array_neuron_units_transposed_variance_it(new T_[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_transposed_variance_it == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory." NEW_LINE,
                                        __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_transposed_variance_it,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neurons variance. |END|
        
        this->ptr_array_neuron_units_transposed_mean = tmp_ptr_array_neuron_units_transposed_mean_it;
        this->ptr_array_neuron_units_transposed_variance = tmp_ptr_array_neuron_units_transposed_variance_it;
        
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            for(tmp_ptr_last_neuron_unit = tmp_ptr_layer_it->ptr_last_neuron_unit,
                tmp_ptr_neuron_unit_it = tmp_ptr_layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
            {
                tmp_ptr_neuron_unit_it->ptr_array_transposed_mean = tmp_ptr_array_neuron_units_transposed_mean_it;
                tmp_ptr_neuron_unit_it->ptr_array_transposed_variance = tmp_ptr_array_neuron_units_transposed_variance_it;

                tmp_ptr_array_neuron_units_transposed_mean_it += this->batch_size;
                tmp_ptr_array_neuron_units_transposed_variance_it += this->batch_size;
            }
        }
    }
    else { return(false); }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Neuron__Mask_Dropout_Bernoulli(void)
{
    if(this->ptr_array_af_units_mask_dropout_bernoulli == nullptr)
    {
        if(this->total_neuron_units == 0u)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate neurons mask dropout. no neuron available." NEW_LINE,
                        __FUNCTION__);

            return(false);
        }

        bool *tmp_ptr_array_af_units_mask_dropout_bernoulli(new bool[this->total_neuron_units]);
        if(tmp_ptr_array_af_units_mask_dropout_bernoulli == nullptr)
        {
            PRINT_FORMAT("%s: %s: ERROR: Can not allocate neurons mask dropout." NEW_LINE,
                        __FUNCTION__);

            return(false);
        }
        Zero_1D<bool>(this->total_neuron_units,
                              tmp_ptr_array_af_units_mask_dropout_bernoulli,
                              this->ptr_array_dim3_grid + 3,
                              this->ptr_array_dim3_block + 3);

        this->Reset__Parameter__AF_Units__Mask_Dropout(tmp_ptr_array_af_units_mask_dropout_bernoulli);
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Batch_Normalization()
{
    // TODO: Reorganasition of the array. [------Weights-----][----Bias----][----Batch renormalization----]. Allocating with the size of each layer. No waste of memory.
    if(this->ptr_array_parameters != nullptr)
    {
        size_t const tmp_new_size(2u * this->total_neuron_units_allocated + this->total_parameters_allocated);
        
        if(this->Reallocate__Parameter(tmp_new_size) == false)
        {
            PRINT_FORMAT("%s: ERROR: From \"Reallocate__Parameter\"." NEW_LINE, __FUNCTION__);

            return(false);
        }
        else if(this->Allocate__Normalized_Unit__Batch_Renormalization() == false)
        {
            PRINT_FORMAT("%s: ERROR: From \"Allocate__Normalized_Unit__Batch_Renormalization\"." NEW_LINE, __FUNCTION__);

            return(false);
        }
        // TODO: Allocate only at training.
        else if(this->Allocate__Neuron__Batch_Renormalization_Transpose() == false)
        {
            PRINT_FORMAT("%s: ERROR: From \"Allocate__Neuron__Batch_Renormalization_Transpose\"." NEW_LINE, __FUNCTION__);

            return(false);
        }
    }
    else { return(false); }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Parameter__Regularization(void)
{
    if(this->ptr_array_mask_regularized_parameters == nullptr)
    {
        this->ptr_array_mask_regularized_parameters = new T_[this->total_parameters_allocated];
        if(this->ptr_array_mask_regularized_parameters == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory!" NEW_LINE, __FUNCTION__);

            return(false);
        }
        Zero_1D<T_>(this->total_parameters_allocated,
                            this->ptr_array_mask_regularized_parameters,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }

    return(true);
}

__device__ bool CUDA_Neural_Network::Allocate__Neuron(struct CUDA_Neuron *ptr_neuron_received)
{
    PRINT_FORMAT("%s: [FUNCTION DEPRECATED] Need to Fix \"Allocate__Neuron\" algorithm." NEW_LINE, __FUNCTION__);

    /*
    if(ptr_neuron_received == NULL)
    {
        PRINT_FORMAT("ERROR: Allocate__Neuron => Neuron_unit is NULL" NEW_LINE);
        return(false);
    }

    if(ptr_neuron_received->ptr_first_forward_connection_index == NULL)
    {
        ptr_neuron_received->ptr_first_forward_connection_index = static_cast<size_t*>(malloc(sizeof(size_t)));
        if(ptr_neuron_received->ptr_first_forward_connection_index == NULL)
        {
            PRINT_FORMAT("ERROR: Allocate__Neuron => Can not allocate memory. ptr_first_forward_connection_index = malloc(%u)" NEW_LINE, sizeof(size_t));
            return(false);
        }
    }
    *ptr_neuron_received->ptr_first_forward_connection_index = 0u;

    if(ptr_neuron_received->ptr_last_forward_connection_index == NULL)
    {
        ptr_neuron_received->ptr_last_forward_connection_index = static_cast<size_t*>(malloc(sizeof(size_t)));
        if(ptr_neuron_received->ptr_last_forward_connection_index == NULL)
        {
            PRINT_FORMAT("ERROR: Allocate__Neuron => Can not allocate memory. ptr_last_forward_connection_index = malloc(%u)" NEW_LINE, sizeof(size_t));
            return(false);
        }
    }
    *ptr_neuron_received->ptr_last_forward_connection_index = 0u;

    if(ptr_neuron_received->ptr_type_activation_function == NULL)
    {
        ptr_neuron_received->ptr_type_activation_function = static_cast<enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS*>(malloc(sizeof(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS)));
        if(ptr_neuron_received->ptr_type_activation_function == NULL)
        {
            PRINT_FORMAT("ERROR: Allocate__Neuron => Can not allocate memory. ptr_type_activation_function = malloc(%u)" NEW_LINE, sizeof(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS));
            return(false);
        }
    }
    *ptr_neuron_received->ptr_type_activation_function = MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTIONS::TYPE_NN_A_F_SIGMOID;
    this->Set__Activation_Function_Neuron(ptr_neuron_received);

    if(ptr_neuron_received->sum == NULL)
    {
        ptr_neuron_received->sum = static_cast<T_*>(malloc(sizeof(T_)));
        if(ptr_neuron_received->sum == NULL)
        {
            PRINT_FORMAT("ERROR: Allocate__Neuron => Can not allocate memory. sum = malloc(%u)" NEW_LINE, sizeof(T_));
            return(false);
        }
    }
    *ptr_neuron_received->sum = 0_T;

    if(ptr_neuron_received->value == NULL)
    {
        ptr_neuron_received->value = static_cast<T_*>(malloc(sizeof(T_)));
        if(ptr_neuron_received->value == NULL)
        {
            PRINT_FORMAT("ERROR: Allocate__Neuron => Can not allocate memory. value = malloc(%u)" NEW_LINE, sizeof(T_));
            return(false);
        }
    }
    *ptr_neuron_received->value = 0_T;

    if(*ptr_neuron_received->ptr_activation_steepness == NULL)
    {
        *ptr_neuron_received->ptr_activation_steepness = static_cast<T_*>(malloc(sizeof(T_)));
        if(*ptr_neuron_received->ptr_activation_steepness == NULL)
        {
            PRINT_FORMAT("ERROR: Allocate__Neuron => Can not allocate memory. activation_steepness = malloc(%u)" NEW_LINE, sizeof(T_));
            return(false);
        }
    }
    **ptr_neuron_received->ptr_activation_steepness = 1_T;
    */

    return(true);
}
