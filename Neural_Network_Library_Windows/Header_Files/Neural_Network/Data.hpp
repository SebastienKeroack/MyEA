#pragma once

#include <stdio.h>
#include <functional>

#include <Enums/Enum_Type_Dataset.hpp>
#include <Enums/Enum_Type_Networks.hpp>
#include <Enums/Enum_Type_Layer.hpp>
#include <Enums/Enum_Type_Layer_Activation.hpp>
#include <Enums/Enum_Type_Layer_Dropout.hpp>
#include <Enums/Enum_Type_Layer_Normalization.hpp>
#include <Enums/Enum_Type_Loss_Functions.hpp>
#include <Enums/Enum_Type_Optimizer_Functions.hpp>
#include <Enums/Enum_Type_Activation_Functions.hpp>
#include <Enums/Enum_Type_Weights_Initializers.hpp>
#include <Enums/Enum_Type_State_Propagation.hpp>

#if defined(COMPILE_CUDA)
    #include <CUDA/CUDA_Neural_Network.cuh>
    //#include <CUDA/CUDA_Dataset_Manager.cuh>
#endif // COMPILE_CUDA

#include <Tools/Configuration.hpp>
#include <Tools/Class_Generator_Random.hpp>
#include <Neural_Network/Activation_Functions.hpp>

// Forward declaration.
class Neural_Network;

#if defined(COMPILE_CUDA)
    template<typename T> class CUDA_Dataset_Manager;
#endif

template<typename T> class Dataset_Manager;
template<typename T> class Dataset;
// |END| Forward declaration. |END|

struct Layer_Parameters
{
    Layer_Parameters(void) { }
    ~Layer_Parameters(void) { }

    bool use_bidirectional = false;

    enum MyEA::Common::ENUM_TYPE_LAYER type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_NONE;

    /* [0]:
        FC: Number of neurons.
        LSTM: Number of blocks.
        RESIDUAL: Block depth.
        POOLING: Kernel size.
       [1]:
        LSTM: Number of cells.
        POOLING: Stride.
       [2]:
        POOLING: Padding.
       [3]:
        POOLING: Dilation.
       [4]:
        POOLING: Ceil mode.*/
    size_t unit_parameters[5u] = {0};
};

struct Neural_Network_Initializer
{
    Neural_Network_Initializer(void) { }
    ~Neural_Network_Initializer(void);

    bool Input_Initialize(void);
    bool Template_Initialize(void);
    bool Build__Layer__FC(struct Layer_Parameters &ref_Layer_Parameters_received);
    bool Build__Layer__Pooling(struct Layer_Parameters &ref_Layer_Parameters_received);
    bool Build__Layer__LSTM(struct Layer_Parameters &ref_Layer_Parameters_received);
    bool Build__Layer__Residual(void);
    bool While__Push_Back__Layer(void);

    class Neural_Network *Output_Initialize(size_t const maximum_allowable_memory_received = 32_zu * KILOBYTE * KILOBYTE) const;

    size_t number_recurrent_depth = 0_zu;
    size_t number_time_delays = 0_zu;

    std::vector<struct Layer_Parameters> vector_layers_parameters;

    enum MyEA::Common::ENUM_TYPE_NETWORKS type_neural_network = MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_NONE;
};

struct Activation_Steepness_Initializer
{
    Activation_Steepness_Initializer(void) { }
    ~Activation_Steepness_Initializer(void);

    void Deallocate_Layers_Activation_Steepness(void);

    bool Allocate__Layers_Activation_Steepness(size_t const number_layers_received);
    bool Input_Initialize(size_t const number_layers_received, enum MyEA::Common::ENUM_TYPE_NETWORKS const type_network_received);
    bool Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const;
    
    size_t number_layers = 0;

    T_ *ptr_array_value_layers_activation_steepness = nullptr;
};

struct Activation_Function_Initializer
{
    Activation_Function_Initializer(void) { }
    ~Activation_Function_Initializer(void);

    void Deallocate_Layers_Activation_Function(void);

    bool Allocate__Layers_Activation_Function(size_t const number_layers_received);
    bool Input_Initialize(size_t const number_layers_received, enum MyEA::Common::ENUM_TYPE_NETWORKS const type_network_received);
    bool Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const;
    
    size_t number_layers = 0u;

    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION *ptr_array_type_layers_activation_function = nullptr;
};

struct Loss_Function_Initializer
{
    Loss_Function_Initializer(void) { }
    ~Loss_Function_Initializer(void) { }

    void Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const;

    bool Input_Initialize(void);

    // Bit.
    T_ bit_fail_limit = 0.5_T;

    enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS type_loss_function = MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MSE;
};

struct Accuracy_Function_Initializer
{
    Accuracy_Function_Initializer(void) { }
    ~Accuracy_Function_Initializer(void) { }

    void Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const;

    bool Input_Initialize(void);

    enum MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS type_accuracy_function = MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_DISTANCE;
};

struct Optimizer_Function_Initializer
{
    Optimizer_Function_Initializer(void) { }
    ~Optimizer_Function_Initializer(void) { }

    bool Input_Initialize(void);
    bool Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const;
    
    /* values
        [0]:
            GD, AdaBound, AMSBound, Adam, AMSGrad, NosAdam:
                learning_rate=0.01, "learning rate."
            iRPROP+:
                rprop_increase_factor=1.2
        [1]:
            AdaBound, AMSBound:
                learning_rate_final=0.1, "Final (SGD) learning rate."
            Adam, AMSGrad, NosAdam:
                beta1=0.9, "Coefficients used for computing running averages of gradient."
            GD:
                learning_momentum=0.9.
            iRPROP+:
                rprop_decrease_factor=0.5
        [2]:
            AdaBound, AMSBound:
                beta1=0.9, "Coefficients used for computing running averages of gradient."
            Adam, AMSGrad, NosAdam:
                beta2=0.999, "Coefficients used for computing running averages of square gradient."
            GD:
                use_Nesterov=1
            iRPROP+:
                rprop_delta_max=50
        [3]:
            AdaBound, AMSBound:
                beta2=0.999, "Coefficients used for computing running averages of square gradient."
            Adam, AMSGrad, NosAdam:
                epsilon=1.0e-8, "Term added to the denominator to improve numerical stability."
            iRPROP+:
                rprop_delta_min=1.0e-6
        [4]:
            AdaBound, AMSBound:
                epsilon=1.0e-8, "Term added to the denominator to improve numerical stability."
            Adam, AMSGrad, NosAdam:
                bias_correction=true, "Moving average to estimate the first and second moments."
            iRPROP+:
                rprop_delta_zero=0.1
        [5]:
            AdaBound, AMSBound:
                bias_correction=true, "Moving average to estimate the first and second moments."
            NosAdam:
                gamma=0.1, "Hyperharmonic."
        [6]:
            AdaBound, AMSBound:
                learning_gamma=1e-3, "Convergence speed of the bound functions."
    */
    T_ values[7u] = {0_T};

    T_ weight_decay = 0_T;

    enum MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS type_optimizer_function = MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD;
};

struct Warm_Restarts_Initializer
{
    Warm_Restarts_Initializer(void) { }
    ~Warm_Restarts_Initializer(void) { }

    void Input_Initialize(void);
    
    bool Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const;
    bool use_Warm_Restarts = false;
    
    T_ warm_restarts_decay_learning_rate = 1_T;
    T_ warm_restarts_maximum_learning_rate = 1_T;
    T_ warm_restarts_minimum_learning_rate = 1.0e-7_T;
    T_ warm_restarts_initial_T_i = 1_T;
    T_ warm_restarts_multiplier = 2_T;
};

struct LSUV_Parameters
{
    LSUV_Parameters(void) { }
    ~LSUV_Parameters(void) { }

    T_ initial_bias = 0_T;
    T_ epsilon = 1e-7_T;
    T_ variance_target = 1_T;
    T_ variance_tolerance = 0.01_T;
    
    size_t maximum_number_trials = 10_zu;
    size_t maximum_batch_size = 32_zu;
};

struct Weights_Initializer
{
    Weights_Initializer(void) { }
    ~Weights_Initializer(void) { }

    bool Input_Initialize(void);
    bool Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const;
    
    T_ initial_bias = 0_T;

    /* values
        [0]:
            Uniform:
                lower_bound=-1
            LSUV:
                maximum_number_trials=10
        [1]:
            Uniform:
                upper_bound=1
            LSUV:
                maximum_batch_size=32
        [2]:
            LSUV:
                variance_target=1
        [3]:
            LSUV:
                variance_tolerance=1
    */
    T_ values[3u] = {0_T};

    enum MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS type_weights_initializer = MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_GLOROT_GAUSSIAN;
};

struct Dropout_Initializer
{
    Dropout_Initializer(void) { }
    ~Dropout_Initializer(void);

    void Deallocate__Layers_Using_Dropout(void);

    bool Allocate__Layers_Using_Dropout(size_t const number_layers_received);
    bool Input_Initialize(size_t const number_layers_received, enum MyEA::Common::ENUM_TYPE_NETWORKS const type_network_received);
    bool Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const;
    
    size_t number_layers = 0u;

    bool *ptr_array_layers_use_coded_dropout = nullptr;

    T_ **ptr_array_layers_dropout_array_values = nullptr;

    enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT *ptr_array_layers_type_dropout = nullptr;
};

struct Normalization_Initializer
{
    Normalization_Initializer(void) { }
    ~Normalization_Initializer(void);

    void Deallocate__Layers_Using_Normalization(void);
    
    bool Allocate__Layers_Using_Normalization(size_t const number_layers_received);
    bool Input_Initialize(size_t const number_layers_received,
                                size_t const number_batch_received,
                                enum MyEA::Common::ENUM_TYPE_NETWORKS const type_network_received);
    bool Output_Initialize(class Neural_Network *const ptr_Neural_Network_received) const;
    bool *ptr_array_layers_normalization_before_activation = nullptr;

    size_t number_layers = 0u;

    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION *ptr_array_layers_using_normalization = nullptr;

    T_ normalization_momentum_average = 0.01_T; // 1 / number of mini-batch
    T_ normalization_epsilon = 1.0e-5_T;

    // Batch renormalization parameter.
    T_ batch_renormalization_r_correction_maximum = 1_T;
    T_ batch_renormalization_d_correction_maximum = 0_T;
    // |END| Batch renormalization parameter. |END|
};

struct Normalized_batch_unit
{
    // N: Number of threads.
    // B: Batch size.
    // T: Number of times to predict.
    // P: Number of parameters.

    T_ *ptr_array_values_hats = nullptr; // size[B, T].
    T_ *ptr_array_values_normalizes = nullptr; // size[B, T].
    T_ *ptr_scale = nullptr; // size[1].
    T_ *ptr_shift = nullptr; // size[1].
    T_ *ptr_array_derivatives_scales = nullptr; // size[N].
    T_ *ptr_array_derivatives_shifts = nullptr; // size[N].
    T_ *ptr_array_means = nullptr; // size[N * T].
    T_ *ptr_array_variances = nullptr; // size[N * T].
    T_ *ptr_array_derivatives_means = nullptr; // size[N * T].
    T_ *ptr_array_derivatives_variances = nullptr; // size[N * T].
    T_ *ptr_r_correction = nullptr; // size[T].
    T_ *ptr_d_correction = nullptr; // size[T].
    T_ *ptr_mean_average = nullptr; // size[T].
    T_ *ptr_variance_average = nullptr; // size[T].
    T_ *ptr_array_errors = nullptr; // size[B, T].
};

struct Normalized_streaming_unit
{
    // N: Number of threads.
    // B: Batch size.
    // T: Number of times to predict.
    // P: Number of parameters.
    
    T_ *ptr_array_values_hats = nullptr; // size[B, T].
    T_ *ptr_array_values_normalizes = nullptr; // size[B, T].
    T_ *ptr_scale = nullptr; // size[1].
    T_ *ptr_shift = nullptr; // size[1].
    T_ *ptr_array_derivatives_scales = nullptr; // size[N].
    T_ *ptr_array_derivatives_shifts = nullptr; // size[N].
    T_ *ptr_array_means = nullptr; // size[N * T].
    T_ *ptr_array_variances = nullptr; // size[N * T].
    T_ *ptr_array_derivatives_means = nullptr; // size[N * T].
    T_ *ptr_array_derivatives_variances = nullptr; // size[N * T].
    T_ *ptr_r_correction = nullptr; // size[T].
    T_ *ptr_d_correction = nullptr; // size[T].
    T_ *ptr_mean_average = nullptr; // size[T].
    T_ *ptr_variance_average = nullptr; // size[T].
    T_ *ptr_array_errors = nullptr; // size[B, T].
};

union Normalized_unit
{
    Normalized_unit(void) { };

    struct Normalized_batch_unit normalized_batch_units;

    struct Normalized_streaming_unit normalized_streaming_units;
};

struct Neuron_Ind
{
    // N: Number of threads.
    // B: Batch size.
    // T: Number of times to predict.
    // P: Number of parameters.

    Neuron_Ind(void) { }
    ~Neuron_Ind(void) { }
};

struct Neuron_unit
{
    // N: Number of threads.
    // B: Batch size.
    // T: Number of times to predict.
    // P: Number of parameters.

    Neuron_unit(void) { }
    ~Neuron_unit(void) { }

    size_t *ptr_first_connection_index = nullptr; // size[1].
    size_t *ptr_last_connection_index = nullptr; // size[1].
    size_t *ptr_number_connections = nullptr; // size[1].

    T_ *ptr_array_summations = nullptr; // size[B, T].
    T_ *ptr_array_errors = nullptr; // size[B, T].
};

struct AF_unit
{
    AF_unit(void) { }
    ~AF_unit(void) { }

    T_ *ptr_activation_steepness = nullptr; // size[1].
    T_ *ptr_array_values = nullptr; // size[B, T].
    T_ *ptr_array_errors = nullptr; // size[B, T].

    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION *ptr_type_activation_function = nullptr; // size[1].
};

struct AF_Ind_recurrent_unit
{
    AF_Ind_recurrent_unit(void) { }
    ~AF_Ind_recurrent_unit(void) { }
    
    size_t *ptr_recurrent_connection_index = nullptr; // size[1].

    T_ *ptr_activation_steepness = nullptr; // size[1].
    T_ *ptr_array_pre_AFs = nullptr; // size[B, T].
    T_ *ptr_array_AFs = nullptr; // size[B, T].
    T_ *ptr_array_errors = nullptr; // size[B, T].
    T_ *ptr_array_dAFs = nullptr; // size[B, T].

    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION *ptr_type_activation_function = nullptr; // size[1].
};

struct Basic_unit
{
    // B: Batch size.
    // T: Number of times to predict.

    Basic_unit(void) { }
    ~Basic_unit(void) { }

    T_ *ptr_array_values = nullptr; // size[B, T].
    T_ *ptr_array_errors = nullptr; // size[B, T].
};

struct Basic_indice_unit
{
    // B: Batch size.
    // T: Number of times to predict.

    Basic_indice_unit(void) { }
    ~Basic_indice_unit(void) { }

    size_t *ptr_array_indices = nullptr; // size[B, T].

    T_ *ptr_array_values = nullptr; // size[B, T].
    T_ *ptr_array_errors = nullptr; // size[B, T].
};

//#define NO_PEEPHOLE

struct Cell_unit
{
    Cell_unit(void) { }
    ~Cell_unit(void) { }
    
    bool *ptr_mask_dropout_zoneout_state = nullptr;
    bool *ptr_mask_dropout_zoneout_output = nullptr;

    size_t first_index_feedforward_connection_cell_input = 0_zu;
    size_t last_index_feedforward_connection_cell_input = 0_zu;
    size_t first_index_recurrent_connection_cell_input = 0_zu;
    size_t last_index_recurrent_connection_cell_input = 0_zu;
#ifndef NO_PEEPHOLE
    size_t index_peephole_input_gate = 0_zu;
    size_t index_peephole_forget_gate = 0_zu;
    size_t index_peephole_output_gate = 0_zu;
#endif

    T_ *ptr_summation_cell_input = nullptr;
    T_ *ptr_summation_input_cell_input = nullptr;
    T_ *ptr_summation_recurrent_cell_input = nullptr;
    T_ *ptr_cell_input = nullptr;
    T_ *ptr_cell_state = nullptr;
    T_ *ptr_cell_state_activate = nullptr;
    T_ *ptr_cell_output = nullptr;
    T_ *ptr_delta_cell_input = nullptr;
    T_ *ptr_delta_cell_input_input = nullptr;
    T_ *ptr_delta_cell_recurrent_input = nullptr;
    T_ *ptr_delta_cell_state = nullptr;
    T_ *ptr_delta_cell_output = nullptr;

    // Normalized unit.
    union Normalized_unit *ptr_array_normalized_units = nullptr; // size[3].
    union Normalized_unit *ptr_last_normalized_unit = nullptr; // size[1].
    // |END| Normalized unit. |END|
};

struct Block_unit
{
    Block_unit(void) { }
    ~Block_unit(void) { }
    
    bool *ptr_array_mask_dropout_zoneout = nullptr;

    size_t first_index_connection = 0_zu;
    size_t last_index_connection = 0_zu;
    size_t first_index_feedforward_connection_input_gate = 0_zu;
    size_t last_index_feedforward_connection_input_gate = 0_zu;
    size_t first_index_feedforward_connection_forget_gate = 0_zu;
    size_t last_index_feedforward_connection_forget_gate = 0_zu;
    size_t first_index_feedforward_connection_output_gate = 0_zu;
    size_t last_index_feedforward_connection_output_gate = 0_zu;
    size_t first_index_recurrent_connection_input_gate = 0_zu;
    size_t last_index_recurrent_connection_input_gate = 0_zu;
    size_t first_index_recurrent_connection_forget_gate = 0_zu;
    size_t last_index_recurrent_connection_forget_gate = 0_zu;
    size_t first_index_recurrent_connection_output_gate = 0_zu;
    size_t last_index_recurrent_connection_output_gate = 0_zu;
#ifndef NO_PEEPHOLE
    size_t first_index_peephole_input_gate = 0_zu;
    size_t last_index_peephole_input_gate = 0_zu;
    size_t first_index_peephole_forget_gate = 0_zu;
    size_t last_index_peephole_forget_gate = 0_zu;
    size_t first_index_peephole_output_gate = 0_zu;
    size_t last_index_peephole_output_gate = 0_zu;
#endif
    
    T_ *ptr_array_summation_cells_inputs = nullptr;
    T_ *ptr_array_summation_input_cells_inputs = nullptr;
    T_ *ptr_array_summation_recurrent_cells_inputs = nullptr;
    T_ *ptr_summation_inputs_gates = nullptr;
    T_ *ptr_summation_input_inputs_gates = nullptr;
    T_ *ptr_summation_recurrent_inputs_gates = nullptr;
    T_ *ptr_summation_forgets_gates = nullptr;
    T_ *ptr_summation_input_forgets_gates = nullptr;
    T_ *ptr_summation_recurrent_forgets_gates = nullptr;
    T_ *ptr_summation_outputs_gates = nullptr;
    T_ *ptr_summation_input_outputs_gates = nullptr;
    T_ *ptr_summation_recurrent_outputs_gates = nullptr;
    T_ *ptr_array_cells_inputs = nullptr;
    T_ *ptr_array_cells_states = nullptr;
    T_ *ptr_array_cells_states_activates = nullptr;
    T_ *ptr_array_cells_outputs = nullptr;
    T_ *ptr_inputs_gates = nullptr;
    T_ *ptr_forgets_gates = nullptr;
    T_ *ptr_outputs_gates = nullptr;
    T_ *ptr_array_delta_cells_inputs = nullptr;
    T_ *ptr_array_delta_cells_input_inputs = nullptr;
    T_ *ptr_array_delta_cells_recurrent_inputs = nullptr;
    T_ *ptr_array_delta_cells_states = nullptr;
    T_ *ptr_array_delta_cells_outputs = nullptr;
    T_ *ptr_delta_inputs_gates = nullptr;
    T_ *ptr_delta_input_inputs_gates = nullptr;
    T_ *ptr_delta_recurrent_inputs_gates = nullptr;
    T_ *ptr_delta_forgets_gates = nullptr;
    T_ *ptr_delta_input_forgets_gates = nullptr;
    T_ *ptr_delta_recurrent_forgets_gates = nullptr;
    T_ *ptr_delta_outputs_gates = nullptr;
    T_ *ptr_delta_input_outputs_gates = nullptr;
    T_ *ptr_delta_recurrent_outputs_gates = nullptr;
    
    struct Cell_unit *ptr_array_cell_units = nullptr;
    struct Cell_unit *ptr_last_cell_unit = nullptr;

    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION activation_function_gate = MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID;
    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION activation_function_io = MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_TANH;
    
    // Normalized unit.
    union Normalized_unit *ptr_array_normalized_units = nullptr; // size[6].
    union Normalized_unit *ptr_last_normalized_unit = nullptr; // size[1].
    // |END| Normalized unit. |END|
};

struct Bidirectional_Layer; // Forward declaration.

struct Layer
{
    Layer(void) { }
    ~Layer(void) { }

    // N: Number of threads.
    // B: Batch size.
    // T: Number of times to predict.
    // H: Number of neurons in layer.
    // R: Number of renormalizations units in layer.
    // K: Number of blocks in layer.
    // C: Number of cells in layer.
    
    bool use_bidirectional = false;
    bool use_tied_parameter = false;
    bool use_coded_dropout = false;
    bool use_layer_normalization_before_activation = true;
    bool *ptr_array__mask__dropout__bernoulli = nullptr; // size[H].
    bool *ptr_array__mask__dropout__shakedrop = nullptr; // size[T].
    bool Use__Bidirectional(void) const { return(this->use_bidirectional); }
    bool Use__Tied_Parameter(void) const;
    bool Use__Coded_Dropout(void) const;
    bool Use__K_Sparsity(void) const;
    bool Use__Regularization__Constraint_Recurrent_Weight(void) const;
    bool Compare__Dimensions(struct Layer const &ref_source_Layer_received) const;
    
    size_t Get__Number_Outputs(void) const;
    size_t Get__First_Connection_Index(void) const;
    size_t Get__Last_Connection_Index(void) const;
    size_t Get__K_Sparsity(void) const;
    size_t *ptr_number_outputs = nullptr; // size[1].
    size_t *ptr_first_connection_index = nullptr; // size[1].
    size_t *ptr_last_connection_index = nullptr; // size[1].
    size_t first_bias_connection_index = 0_zu; // size[1].
    size_t last_bias_connection_index = 0_zu; // size[1].
    size_t block_depth = 0_zu;
    size_t k_sparsity = 0_zu;
    
    /* pooling_values:
            [0]: Kernel size.
            [1]: Stride. 
            [2]: Padding.
            [3]: Dilation.
            [4]: Ceil mode. */
    size_t pooling_values[5u] = {0};

    std::pair<size_t, T_> *ptr_array_k_sparse_activities = nullptr;

    enum MyEA::Common::ENUM_TYPE_LAYER type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_NONE;
    enum MyEA::Common::ENUM_TYPE_GROUP type_group = MyEA::Common::ENUM_TYPE_GROUP::TYPE_GROUP_NONE;
    enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION type_activation = MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_NONE;
    enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;
    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION type_normalization = MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE;
    
    // Basic unit variable.
    struct Basic_unit *ptr_array_basic_units = nullptr; // size[H].
    struct Basic_unit *ptr_last_basic_unit = nullptr; // size[1].
    // |END| Basic unit variable. |END|
    
    // Basic indice unit variable.
    struct Basic_indice_unit *ptr_array_basic_indice_units = nullptr; // size[H].
    struct Basic_indice_unit *ptr_last_basic_indice_unit = nullptr; // size[1].
    // |END| Basic indice unit variable. |END|
    
    // FC layer variable.
    T_ *ptr_array_pre_summations = nullptr; // size[1].
    struct Neuron_unit *ptr_array_neuron_units = nullptr; // size[H].
    struct Neuron_unit *ptr_last_neuron_unit = nullptr; // size[1].
    // |END| FC layer variable. |END|
    
    // AF unit(s) variable.
    T_ *ptr_array_pre_activation_functions = nullptr; // size[1].
    struct AF_unit *ptr_array_AF_units = nullptr; // size[H].
    struct AF_unit *ptr_last_AF_unit = nullptr; // size[1].
    // |END| AF unit(s) variable. |END|
    
    // AF unit(s) variable.
    struct AF_Ind_recurrent_unit *ptr_array_AF_Ind_recurrent_units = nullptr; // size[H].
    struct AF_Ind_recurrent_unit *ptr_last_AF_Ind_recurrent_unit = nullptr; // size[1].
    // |END| AF unit(s) variable. |END|
    
    // LSTM layer variable.
    struct Block_unit *ptr_array_block_units = nullptr; // size[K].
    struct Block_unit *ptr_last_block_unit = nullptr; // size[1].
    
    struct Cell_unit *ptr_array_cell_units = nullptr; // size[C].
    struct Cell_unit *ptr_last_cell_unit = nullptr; // size[1].
    // |END| LSTM layer variable. |END|

    // Bidirectional layer variable.
    struct Bidirectional_Layer *ptr_Bidirectional_Layer = nullptr; // size[1].
    // |END| Bidirectional layer variable. |END|
    
    // Normalized unit.
    T_ *ptr_array_pre_normalization = nullptr; // size[1].
    union Normalized_unit *ptr_array_normalized_units = nullptr; // size[H || (H * 4 + C)].
    union Normalized_unit *ptr_last_normalized_unit = nullptr; // size[1].
    // |END| Normalized unit. |END|
    
    // Layer(s) connections.
    std::vector<struct Layer const *> previous_connected_layers;
    std::vector<struct Layer const *> next_connected_layers;
    // |END| Layer(s) connections |END|

    /* dropout_values:
        Bernoulli:
            [0]: Keep probability.
        Uout:
            [0]: Dropout probability.
        Zoneout:
            [0]: Cell zoneout probability. 
            [1]: Hidden zoneout probability.
        Alpha:
            [0]: Dropout probability.
            [1]: a. 
            [2]: b. */
    T_ dropout_values[3u] = {0};
    
    T_ const *Get__Array_Summations__Cell__Block_Input__Input__Activation(void) const;
    T_ const *Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(void) const;
    T_ const *Get__Array_Summations__Cell__Cell_State__Activation(void) const;
    T_ const *Get__Array_Summations__Block__Input_Gate__Input__Activation(void) const;
    T_ const *Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(void) const;
    T_ const *Get__Array_Summations__Block__Forget_Gate__Input__Activation(void) const;
    T_ const *Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(void) const;
    T_ const *Get__Array_Summations__Block__Output_Gate__Input__Activation(void) const;
    T_ const *Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(void) const;
    T_ const *Get__Array_Deltas__Cell__Block_Input__Input(void) const;
    T_ const *Get__Array_Deltas__Cell__Block_Input__Recurrent(void) const;
    T_ const *Get__Array_Deltas__Block__Input_Gate__Input(void) const;
    T_ const *Get__Array_Deltas__Block__Input_Gate__Recurrent(void) const;
    T_ const *Get__Array_Deltas__Block__Forget_Gate__Input(void) const;
    T_ const *Get__Array_Deltas__Block__Forget_Gate__Recurrent(void) const;
    T_ const *Get__Array_Deltas__Block__Output_Gate__Input(void) const;
    T_ const *Get__Array_Deltas__Block__Output_Gate__Recurrent(void) const;
    T_ Get__Alpha_Sparsity(void) const;
    T_ *ptr_array_derivative_outputs = nullptr;
    T_ *ptr_array_outputs = nullptr;
    T_ constraint_recurrent_weight_lower_bound = 0_T;
    T_ constraint_recurrent_weight_upper_bound = 0_T;
    T_ alpha_sparsity = 1_T;

    bool Use__Bias(void) const { return(this->Use__Normalization() == false); }
    bool Use__Dropout(void) const { return(this->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE); }
    bool Use__Dropout__Alpha(void) const { return(this->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA); }
    bool Use__Dropout__Bernoulli(void) const { return(this->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI); }
    bool Use__Dropout__Bernoulli__Inverted(void) const { return(this->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED); }
    bool Use__Dropout__Gaussian(void) const { return(this->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN); }
    bool Use__Dropout__ShakeDrop(void) const { return(this->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP); }
    bool Use__Dropout__Uout(void) const { return(this->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT); }
    bool Use__Dropout__Zoneout(void) const { return(this->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT); }
    bool Use__Normalization(void) const { return(this->type_normalization != MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_NONE); }
    bool Use__Batch_Normalization(void) const { return(this->type_normalization == MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_NORMALIZATION); }
    bool Use__Batch_Renormalization(void) const { return(this->type_normalization == MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION::TYPE_LAYER_NORMALIZATION_BATCH_RENORMALIZATION); }
};

struct Bidirectional_Layer
{
    Bidirectional_Layer(void) { }
    ~Bidirectional_Layer(void) { }

    struct Layer forward_layer;
    struct Layer backward_layer;
};

// TODO: Tied parameter(s)
class Neural_Network
{
    public:
        // N: Number of threads.
        // B: Batch size.
        // T: Number of times to predict.
        // L: Number of layers.
        // H: Number of neurons.
        // K: Number of blocks.
        // C: Number of cells.
        // P: Number of parameters.
        // W: Number of weights.
        
        Neural_Network(void) { }
        ~Neural_Network(void);
        
        class Neural_Network& operator = (class Neural_Network const &ref_source_Neural_Network_received);
        
        bool operator == (class Neural_Network const &ref_source_Neural_Network_received);
        bool operator != (class Neural_Network const &ref_source_Neural_Network_received);
        
        void Initialize__OpenMP(void);
        template<typename U> void Initialize_Connections__FC(struct Layer *const ptr_layer_it_received, U *const ptr_previous_layer_array_units_received);
        template<typename U> void Initialize_Connections__LSTM(struct Layer *const ptr_layer_it_received, U *const ptr_previous_layer_array_units_received);
        void Initialize_Connections__AF_Ind_Recurrent(struct Layer *const ptr_layer_it_received);
        void Initialize_Connections__Bias(struct Layer *const ptr_layer_it_received);
        void Initialize_Connections__LSTM__Bias(struct Layer *const ptr_layer_it_received);
        void Initialize_Connections__FC_to_FC(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received);
        void Initialize_Connections__FC_to_LSTM(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received);
        void Initialize_Connections__LSTM_to_FC(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received);
        void Initialize_Connections__LSTM_to_LSTM(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received);
        void Initialize_Connections__Basic_unit_to_FC(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received);
        void Initialize_Connections__Basic_unit_to_LSTM(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received);
        void Initialize_Connections__Basic_indice_unit_to_FC(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received);
        void Initialize_Connections__Basic_indice_unit_to_LSTM(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received);
        void Initialize__Constant__Bias(T_ const bias_received, struct Layer const *const ptr_layer_it_received);
        void Initialize__Constant__LSTM__Bias(T_ const bias_received, struct Layer const *const ptr_layer_it_received);
        void Initialize__Uniform(T_ *ptr_array_weights_received,
                                          T_ const *const ptr_last_weight_received,
                                          T_ const lower_bound_received,
                                          T_ const upper_bound_received);
        void Initialize__Uniform__LSTM(T_ const lower_bound_received[5u],
                                                      T_ const upper_bound_received[5u],
                                                      struct Layer const *const ptr_layer_it_received);
        void Initialize__Uniform__AF_Ind_Recurrent(struct Layer const *const ptr_layer_it_received);
        void Initialize__Uniform__AF_Ind_Recurrent__Long_Term_Memory(void);
        void Initialize__Gaussian(T_ *ptr_array_weights_received,
                                             T_ const *const ptr_last_weight_received,
                                             T_ const variance_received);
        void Initialize__Gaussian__LSTM(T_ const feedforward_cell_variance_received,
                                                         T_ const feedforward_gates_variance_received,
                                                         T_ const recurrent_cell_variance_received,
                                                         T_ const recurrent_gates_variance_received,
                                                         T_ const peephole_variance_received,
                                                         struct Layer *const ptr_layer_it_received);
        void Initialize__Identity(size_t const rows_received,
                                         size_t const columns_received,
                                         T_ *const ptr_array_weights_received);
        void Initialize__Orthogonal(size_t const rows_received,
                                               size_t const columns_received,
                                               T_ const scale_received,
                                               T_ *ptr_array_weights_received);
        void Reset__Global_Loss(void);
        void Reset__Loss(void);
        void Merge__Post__Training(void);
        void Merge__Accuracy__R(void);
        void Plot__Gradient(void);

        bool plot_gradient = false;

        void Set__Maximum_Allowable_Memory(size_t const maximum_allowable_memory_bytes_received);
        void Set__Loss_Function(enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS const loss_function_received);
        void Set__Accuracy_Function(enum MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS const type_accuracy_function_received);
        void Set__Optimizer_Function(enum MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS const optimizer_function_received);
        void Set__Bit_Fail_Limit(T_ const bit_fail_limiTreceived);
        void Set__Accuracy(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, T_ const accurancy_received);
        void Set__Loss(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, T_ const loss_received);
        void Set__Clip_Gradient(bool const use_clip_gradient_received);
        void Clip_Gradient__Loop(size_t const start_index_received, size_t const end_index_received);
        void Clip_Gradient__OpenMP(size_t const start_index_received, size_t const end_index_received);
        void Assign__Sparsity_Activities(size_t const number_threads_received);
        // TODO: Normalization tied.
        void Tied__Transpose(void);
        void Tied__Transpose(struct Layer *const ptr_layer_received);
        void Tied__Transpose__Weight(struct Layer *const ptr_layer_received);
        void Tied__Transpose__Weight__FC(struct Layer const *const ptr_coded_layer_it_received, struct Layer const *const ptr_mirror_layer_it_received);
        void Tied__Transpose__Weight__FC_Ind_RNN(struct Layer const *const ptr_encoded_layer_it_received, struct Layer const *const ptr_mirror_layer_it_received);
        void Tied__Transpose__Normalization(struct Layer *const ptr_layer_received);
        void Tied__Transpose__Normalization__Batch_Normalization(struct Layer const *const ptr_encoded_layer_it_received, struct Layer const *const ptr_mirror_layer_it_received);
        void Update_Parameter(size_t const batch_size_received, size_t const training_size_received);
        void Update_Parameter__Loop(size_t const batch_size_received, size_t const training_size_received);
        void Update_Parameter__OpenMP(size_t const batch_size_received, size_t const training_size_received);
        void Update_Parameter__Gradient_Descent(size_t const batch_size_received,
                                                                        size_t const training_size_received,
                                                                        size_t const start_index_received,
                                                                        size_t const end_index_received);
        void Update_Parameter__Gradient_Descent__Loop(size_t const batch_size_received,
                                                                                  size_t const training_size_received,
                                                                                  size_t const start_index_received,
                                                                                  size_t const end_index_received);
        void Update_Parameter__Gradient_Descent_Momentum__Loop(size_t const batch_size_received,
                                                                                                    size_t const training_size_received,
                                                                                                    size_t const start_index_received,
                                                                                                    size_t const end_index_received);
        void Update_Parameter_Nesterov_Accelerated_Gradient__Loop(size_t const batch_size_received,
                                                                                                    size_t const training_size_received,
                                                                                                    size_t const start_index_received,
                                                                                                    size_t const end_index_received);
        void Update_Parameter__Gradient_Descent__OpenMP(size_t const batch_size_received,
                                                                                        size_t const training_size_received,
                                                                                        size_t const start_index_received,
                                                                                        size_t const end_index_received);
        void Update_Parameter__Gradient_Descent_Momentum__OpenMP(size_t const batch_size_received,
                                                                                                          size_t const training_size_received,
                                                                                                          size_t const start_index_received,
                                                                                                          size_t const end_index_received);
        void Update_Parameter_Nesterov_Accelerated_Gradient__OpenMP(size_t const batch_size_received,
                                                                                                          size_t const training_size_received,
                                                                                                          size_t const start_index_received,
                                                                                                          size_t const end_index_received);
        void Update_Parameters__AdaBound(size_t const batch_size_received,
                                                                size_t const training_size_received,
                                                                size_t const start_index_received,
                                                                size_t const end_index_received);
        void Update_Parameters__AdaBound__Loop(size_t const batch_size_received,
                                                                          size_t const training_size_received,
                                                                          size_t const start_index_received,
                                                                          size_t const end_index_received);
        void Update_Parameters__AdaBound__OpenMP(size_t const batch_size_received,
                                                                                size_t const training_size_received,
                                                                                size_t const start_index_received,
                                                                                size_t const end_index_received);
        void Update_Parameters__Adam(size_t const batch_size_received,
                                                        size_t const training_size_received,
                                                        size_t const start_index_received,
                                                        size_t const end_index_received);
        void Update_Parameters__Adam__Loop(size_t const batch_size_received,
                                                                   size_t const training_size_received,
                                                                   size_t const start_index_received,
                                                                   size_t const end_index_received);
        void Update_Parameters__Adam__OpenMP(size_t const batch_size_received,
                                                                         size_t const training_size_received,
                                                                         size_t const start_index_received,
                                                                         size_t const end_index_received);
        void Update_Parameters__AMSBound(size_t const batch_size_received,
                                                              size_t const training_size_received,
                                                              size_t const start_index_received,
                                                              size_t const end_index_received);
        void Update_Parameters__AMSBound__Loop(size_t const batch_size_received,
                                                                           size_t const training_size_received,
                                                                           size_t const start_index_received,
                                                                           size_t const end_index_received);
        void Update_Parameters__AMSBound__OpenMP(size_t const batch_size_received,
                                                                                size_t const training_size_received,
                                                                                size_t const start_index_received,
                                                                                size_t const end_index_received);
        void Update_Parameters__AMSGrad(size_t const batch_size_received,
                                                              size_t const training_size_received,
                                                              size_t const start_index_received,
                                                              size_t const end_index_received);
        void Update_Parameters__AMSGrad__Loop(size_t const batch_size_received,
                                                                         size_t const training_size_received,
                                                                         size_t const start_index_received,
                                                                         size_t const end_index_received);
        void Update_Parameters__AMSGrad__OpenMP(size_t const batch_size_received,
                                                                              size_t const training_size_received,
                                                                              size_t const start_index_received,
                                                                              size_t const end_index_received);
        void Update_Parameters__NosAdam(size_t const batch_size_received,
                                                              size_t const training_size_received,
                                                              size_t const start_index_received,
                                                              size_t const end_index_received);
        void Update_Parameters__NosAdam__Loop(size_t const batch_size_received,
                                                                         size_t const training_size_received,
                                                                         size_t const start_index_received,
                                                                         size_t const end_index_received);
        void Update_Parameters__NosAdam__OpenMP(size_t const batch_size_received,
                                                                              size_t const training_size_received,
                                                                              size_t const start_index_received,
                                                                              size_t const end_index_received);
        void Update_Parameter__iRPROP_plus(size_t const start_index_received, size_t const end_index_received);
        void Update_Parameter__iRPROP_minus__Loop(size_t const start_index_received, size_t const end_index_received);
        void Update_Parameter__iRPROP_plus__Loop(size_t const start_index_received, size_t const end_index_received);
        void Update_Parameter__iRPROP_plus__OpenMP(size_t const start_index_received, size_t const end_index_received);
        void Dropout_Bernoulli(void);
        void Dropout_Bernoulli__Loop(void);
        void Dropout_Bernoulli__Layer__Loop(size_t const number_outputs_received, struct Layer *const ptr_layer_it_received);
        void Dropout_Bernoulli__OpenMP(void);
        void Dropout_Bernoulli__Layer__OpenMP(size_t const number_outputs_received, struct Layer *const ptr_layer_it_received);
        void Dropout_Zoneout(void);
        void Dropout_Zoneout__Loop(void);
        void Dropout_Zoneout__Block_Units__Loop(struct Layer *const ptr_layer_it_received);
        void Dropout_Zoneout__OpenMP(void);
        void Dropout_Zoneout__Block_Units__OpenMP(struct Layer *const ptr_layer_it_received);
        void Euclidean_Norm__Loop(size_t const start_index_received,
                                                  size_t const end_index_received,
                                                  T_ const max_norm_received,
                                                  T_ *const ptr_array_vector_received);
        void Euclidean_Norm__OpenMP(size_t const start_index_received,
                                                        size_t const end_index_received,
                                                        T_ const max_norm_received,
                                                        T_ *const ptr_array_vector_received);
        void Update_Weight_Regularization__Max_Norm_Constraints(size_t const start_index_received, size_t const end_index_received);
        void Update_Weight_Regularization__Max_Norm_Constraints__Loop(size_t const start_index_received, size_t const end_index_received);
        void Update_Weight_Regularization__Max_Norm_Constraints__Neurons__Loop(size_t const start_index_received,
                                                                                                                            size_t const end_index_received,
                                                                                                                            struct Layer const *const ptr_layer_it_received,
                                                                                                                            struct Layer const *const ptr_last_layer_received);
        void Update_Weight_Regularization__Max_Norm_Constraints__LSTMs__Loop(size_t const start_index_received,
                                                                                                                          size_t const end_index_received,
                                                                                                                          struct Layer const *const ptr_layer_it_received,
                                                                                                                          struct Layer const *const ptr_last_layer_received);
        void Update_Weight_Regularization__Max_Norm_Constraints__OpenMP(size_t const start_index_received, size_t const end_index_received);
        void Update_Weight_Regularization__Max_Norm_Constraints__Neurons__OpenMP(size_t const start_index_received,
                                                                                                                                 size_t const end_index_received,
                                                                                                                                 struct Layer const *const ptr_layer_it_received,
                                                                                                                                 struct Layer const *const ptr_last_layer_received);
        void Update_Weight_Regularization__Max_Norm_Constraints__LSTMs__OpenMP(size_t const start_index_received,
                                                                                                                                size_t const end_index_received,
                                                                                                                                struct Layer const *const ptr_layer_it_received,
                                                                                                                                struct Layer const *const ptr_last_layer_received);
        void Update_Weight_Regularization__Constraint_Recurrent_Weight(size_t const start_index_received, size_t const end_index_received);
        void Update_Weight_Regularization__Constraint_Recurrent_Weight__FC_Ind_RNN(struct Layer const *const ptr_layer_it_received);
        void Update_Derivative_Weight__Regularization__L1(size_t const start_index_received,
                                                                                  size_t const end_index_received,
                                                                                  size_t const batch_size_received);
        void Update_Derivative_Weight__Regularization__L1__Loop(size_t const start_index_received,
                                                                                             size_t const end_index_received,
                                                                                             size_t const batch_size_received);
        void Update_Derivative_Weight__Regularization__L1__OpenMP(size_t const start_index_received,
                                                                                                   size_t const end_index_received,
                                                                                                   size_t const batch_size_received);
        void Update_Derivative_Weight__Regularization__L2(size_t const start_index_received,
                                                                                  size_t const end_index_received,
                                                                                  size_t const batch_size_received);
        void Update_Derivative_Weight__Regularization__L2__Loop(size_t const start_index_received,
                                                                                             size_t const end_index_received,
                                                                                             size_t const batch_size_received);
        void Update_Derivative_Weight__Regularization__L2__OpenMP(size_t const start_index_received,
                                                                                                   size_t const end_index_received,
                                                                                                   size_t const batch_size_received);
        void Update_Derivative_Weight__Regularization__SRIP(size_t const start_index_received,
                                                                                        size_t const end_index_received,
                                                                                        size_t const batch_size_received);
        void Update_Derivative_Weight__Regularization__SRIP__Loop(size_t const start_index_received,
                                                                                                   size_t const end_index_received,
                                                                                                   size_t const batch_size_received);
        void Update_Derivative_Weight__Regularization__SRIP__OpenMP(size_t const start_index_received,
                                                                                                        size_t const end_index_received,
                                                                                                        size_t const batch_size_received);
        void Sparse_K_Filter(size_t const time_step_index_received,
                                       size_t const batch_size_received,
                                       size_t const input_unit_size_received,
                                       size_t const k_sparsity_received,
                                       std::pair<size_t, T_> *const ptr_array_k_sparses_received,
                                       T_ *const ptr_array_inputs_received);
        void Sparse_K_Filter__Loop(size_t const time_step_index_received,
                                                 size_t const batch_size_received,
                                                 size_t const input_unit_size_received,
                                                 size_t const k_sparsity_received,
                                                 std::pair<size_t, T_> *const ptr_array_k_sparses_received,
                                                 T_ *const ptr_array_inputs_received);
        void Sparse_K_Filter__OpenMP(size_t const time_step_index_received,
                                                       size_t const batch_size_received,
                                                       size_t const input_unit_size_received,
                                                       size_t const k_sparsity_received,
                                                       std::pair<size_t, T_> *const ptr_array_k_sparses_received,
                                                       T_ *const ptr_array_inputs_received);
        void Compute__Loss(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void RNN__Compute__Loss__Loop(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void RNN__Compute__Loss__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void FF__Compute__Loss__Loop(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void FF__Compute__Loss__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void Compute__Error(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void FF__Compute__Error__Loop(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void FF__Compute__Error__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void RNN__Compute__Error__Loop(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void RNN__Compute__Error__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void Compute__Accuracy__R(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void FF__Compute__Accuracy__R__Loop(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void FF__Compute__Accuracy__R__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void RNN__Compute__Accuracy__R__Loop(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void RNN__Compute__Accuracy__R__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_array_desireds_outputs_received);
        void FF__Forward_Pass_Batch__Loop(size_t const batch_size_received,
                                                                 T_ const *const *const ptr_array_inputs_received,
                                                                 struct Layer *const ptr_first_layer_received,
                                                                 struct Layer const *const ptr_last_layer_received);
        void FF__Forward_Pass_Batch__Pre_Training__Loop(size_t const batch_size_received, T_ const *const *const ptr_array_inputs_received);
        void Forward_Pass__Average_Pooling__Loop(size_t const time_step_index_received,
                                                                          size_t const batch_size_received,
                                                                          size_t const input_unit_size_received,
                                                                          T_ const *const ptr_array_inputs_received,
                                                                          struct Layer *const ptr_layer_it_received);
        void Forward_Pass__FC__Loop(size_t const time_step_index_received,
                                                       size_t const batch_size_received,
                                                       size_t const input_unit_size_received,
                                                       T_ const *const ptr_array_inputs_received,
                                                       struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Encode__FC__Loop(size_t const time_step_index_received,
                                                                     size_t const batch_size_received,
                                                                     size_t const input_unit_size_received,
                                                                     T_ const *const ptr_array_inputs_received,
                                                                     struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Code__FC__Loop(size_t const time_step_index_received,
                                                                  size_t const batch_size_received,
                                                                  size_t const input_unit_size_received,
                                                                  T_ const *const ptr_array_inputs_received,
                                                                  struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Decode__FC__Loop(size_t const time_step_index_received,
                                                                     size_t const batch_size_received,
                                                                     size_t const input_unit_size_received,
                                                                     T_ const *const ptr_array_inputs_received,
                                                                     struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__Loop(long long int const time_step_index_received,
                                                           long long int const tmp_time_step_reverse_direction,
                                                           long long int const tmp_time_step_start,
                                                           size_t const batch_size_received,
                                                           size_t const input_unit_size_received,
                                                           T_ const *const ptr_array_inputs_received,
                                                           struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Encode__LSTM__Loop(long long int const time_step_index_received,
                                                                         long long int const tmp_time_step_reverse_direction,
                                                                         long long int const tmp_time_step_start,
                                                                         size_t const batch_size_received,
                                                                         size_t const input_unit_size_received,
                                                                         T_ const *const ptr_array_inputs_received,
                                                                         struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Code__LSTM__Loop(long long int const time_step_index_received,
                                                                      long long int const tmp_time_step_reverse_direction,
                                                                      long long int const tmp_time_step_start,
                                                                      size_t const batch_size_received,
                                                                      size_t const input_unit_size_received,
                                                                      T_ const *const ptr_array_inputs_received,
                                                                              struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Decode__LSTM__Loop(long long int const time_step_index_received,
                                                                         long long int const tmp_time_step_reverse_direction,
                                                                         long long int const tmp_time_step_start,
                                                                         size_t const batch_size_received,
                                                                         size_t const input_unit_size_received,
                                                                         T_ const *const ptr_array_inputs_received,
                                                                         struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Max_Pooling__Loop(size_t const time_step_index_received,
                                                                     size_t const batch_size_received,
                                                                     size_t const input_unit_size_received,
                                                                     T_ const *const ptr_array_inputs_received,
                                                                     struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Residual__Loop(size_t const batch_size_received, struct Layer *&ptr_layer_it_received);
        void Forward_Pass__Residual__Layer__Loop(bool const is_block_input_layer_received,
                                                                          size_t const batch_size_received,
                                                                          struct Layer *&ptr_layer_it_received);
        void Forward_Pass__Residual__FC__Loop(bool const is_block_input_layer_received,
                                                                       size_t const time_step_index_received,
                                                                       size_t const batch_size_received,
                                                                       size_t const input_size_received,
                                                                       T_ const *const ptr_array_inputs_received,
                                                                       struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Average_Pooling__Loop(size_t const time_step_index_received,
                                                                          size_t const batch_size_received,
                                                                          size_t const input_unit_size_received,
                                                                          size_t const output_size_received,
                                                                          size_t const kernel_size_received,
                                                                          size_t const stride_received,
                                                                          size_t const padding_received,
                                                                          size_t const dilation_received,
                                                                          T_ const *const ptr_array_inputs_received,
                                                                          T_ *const ptr_array_outputs_received);
         void Forward_Pass__Bias__Loop(size_t const time_step_index_received,
                                                          size_t const batch_size_received,
                                                          size_t const output_size_received,
                                                          T_ const *const ptr_array_bias_received,
                                                          T_ *const ptr_array_outputs_received);
        void Forward_Pass__FC__Loop(size_t const time_step_index_received,
                                                       size_t const batch_size_received,
                                                       size_t const input_size_received,
                                                       size_t const output_size_received,
                                                       T_ const *const ptr_array_inputs_received,
                                                       T_ const *const ptr_array_parameters_received,
                                                       T_ *const ptr_array_outputs_received);
        void Forward_Pass__FC_Ind_RNN__Loop(size_t const time_step_index_received,
                                                                     size_t const batch_size_received,
                                                                     size_t const input_size_received,
                                                                     T_ const *const ptr_array_parameters_received,
                                                                     T_ const *const ptr_array_AFs_received,
                                                                     T_ const *const ptr_array_inputs_received,
                                                                     T_ *const ptr_array_outputs_received);
        void Forward_Pass__Batch_Normalization__Inference__Loop(size_t const time_step_index_received,
                                                                                                 size_t const batch_size_received,
                                                                                                 size_t const input_size_received,
                                                                                                 T_ const *const ptr_array_inputs_received,
                                                                                                 T_ const *const ptr_array_scales_received,
                                                                                                 T_ const *const ptr_array_shifts_received,
                                                                                                 T_ const *const ptr_array_means_averages_received,
                                                                                                 T_ const *const ptr_array_variances_averages_received,
                                                                                                 T_ *const ptr_array_output_normalizes_received);
        void Forward_Pass__Batch_Normalization__Training__Loop(size_t const time_step_index_received,
                                                                                               size_t const batch_size_received,
                                                                                               size_t const input_size_received,
                                                                                               T_ const *const ptr_array_inputs_received,
                                                                                               T_ const *const ptr_array_scales_received,
                                                                                               T_ const *const ptr_array_shifts_received,
                                                                                               T_ *const ptr_array_means_received,
                                                                                               T_ *const ptr_array_variances_received,
                                                                                               T_ *const ptr_array_means_averages_received,
                                                                                               T_ *const ptr_array_variances_averages_received,
                                                                                               T_ *const ptr_array_output_hats_received,
                                                                                               T_ *const ptr_array_output_normalizes_received);
        void Forward_Pass__Batch_Renormalization__Training__Loop(size_t const time_step_index_received,
                                                                                                  size_t const batch_size_received,
                                                                                                  size_t const input_size_received,
                                                                                                  T_ const *const ptr_array_inputs_received,
                                                                                                  T_ const *const ptr_array_scales_received,
                                                                                                  T_ const *const ptr_array_shifts_received,
                                                                                                  T_ *const ptr_array_means_received,
                                                                                                  T_ *const ptr_array_variances_received,
                                                                                                  T_ *const ptr_array_means_averages_received,
                                                                                                  T_ *const ptr_array_variances_averages_received,
                                                                                                  T_ *const ptr_array_r_corrections_received,
                                                                                                  T_ *const ptr_array_d_corrections_received,
                                                                                                  T_ *const ptr_array_output_hats_received,
                                                                                                  T_ *const ptr_array_output_normalizes_received);
        void Forward_Pass__FC_AF__Loop(size_t const time_step_index_received,
                                                             size_t const batch_size_received,
                                                             size_t const input_size_received,
                                                             T_ const *const ptr_array_inputs_received,
                                                             T_ *const ptr_array_outputs_received,
                                                             enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const *const ptr_array_type_activations_received);
        void Forward_Pass__FC_AF__Softmax__Loop(size_t const time_step_index_received,
                                                                            size_t const batch_size_received,
                                                                            size_t const input_size_received,
                                                                            T_ const *const ptr_array_inputs_received,
                                                                            T_ *const ptr_array_outputs_received);
        void Forward_Pass__Dropout__Bernoulli__Inverted__Loop(bool const *const ptr_array__mask__dropout__bernoulli_received,
                                                                                            size_t const time_step_index_received,
                                                                                            size_t const batch_size_received,
                                                                                            size_t const input_size_received,
                                                                                            T_ const inverse_retention_probability_divided_received,
                                                                                            T_ *const ptr_array_inputs_received);
        void Forward_Pass__Dropout__Bernoulli__Training__Loop(bool const *const ptr_array__mask__dropout__bernoulli_received,
                                                                                            size_t const time_step_index_received,
                                                                                            size_t const batch_size_received,
                                                                                            size_t const input_size_received,
                                                                                            T_ *const ptr_array_inputs_received);
        void Forward_Pass__Dropout__Bernoulli__Inference__Loop(size_t const time_step_index_received,
                                                                                              size_t const batch_size_received,
                                                                                              size_t const input_size_received,
                                                                                              T_ const retention_probability_received,
                                                                                              T_ *const ptr_array_inputs_received);
        void Forward_Pass__Dropout__Gaussian__Loop(size_t const time_step_index_received,
                                                                               size_t const batch_size_received,
                                                                               size_t const input_size_received,
                                                                               T_ const variance_received,
                                                                               T_ *const ptr_array_inputs_received);
        void Forward_Pass__Dropout__ShakeDrop__Loop(size_t const time_step_index_received,
                                                                                 size_t const batch_size_received,
                                                                                 size_t const input_size_received,
                                                                                 bool *const ptr_array_mask_dopout_shakedrop_received,
                                                                                 T_ const lower_bound_received,
                                                                                 T_ const upper_bound_received,
                                                                                 T_ const dropout_probability_received,
                                                                                 T_ *const ptr_array_inputs_received);
        void Forward_Pass__Dropout__Uout__Loop(size_t const time_step_index_received,
                                                                        size_t const batch_size_received,
                                                                        size_t const input_size_received,
                                                                        T_ const beta_received,
                                                                        T_ *const ptr_array_inputs_received);
        void Forward_Pass__Max_Pooling__Loop(size_t const time_step_index_received,
                                                                     size_t const batch_size_received,
                                                                     size_t const input_size_received,
                                                                     size_t const output_size_received,
                                                                     size_t const kernel_size_received,
                                                                     size_t const stride_received,
                                                                     size_t const padding_received,
                                                                     size_t const dilation_received,
                                                                     size_t *const ptr_array_indices_received,
                                                                     T_ const *const ptr_array_inputs_received,
                                                                     T_ *const ptr_array_outputs_received);
        void Forward_Pass__Zero_Padded_Identity__Loop(size_t const time_step_index_received,
                                                                                  size_t const batch_size_received,
                                                                                  size_t const A_unit_size_received,
                                                                                  size_t const B_unit_size_received,
                                                                                  size_t const padding_received,
                                                                                  T_ const *const ptr_array_A_received,
                                                                                  T_ const *const ptr_array_B_received,
                                                                                  T_ *const ptr_array_outputs_received);
        void FF__Forward_Pass_Batch__OpenMP(size_t const batch_size_received,
                                                                      T_ const *const *const ptr_array_inputs_received,
                                                                      struct Layer *const ptr_first_layer_received,
                                                                      struct Layer const *const ptr_last_layer_received);
        void FF__Forward_Pass_Batch__Pre_Training__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_array_inputs_received);
        void Forward_Pass__Average_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                size_t const batch_size_received,
                                                                                size_t const input_unit_size_received,
                                                                                T_ const *const ptr_array_inputs_received,
                                                                                struct Layer *const ptr_layer_it_received);
        void Forward_Pass__FC__OpenMP(size_t const time_step_index_received,
                                                            size_t const batch_size_received,
                                                            size_t const input_unit_size_received,
                                                            T_ const *const ptr_array_inputs_received,
                                                            struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Encode__FC__OpenMP(size_t const time_step_index_received,
                                                                           size_t const batch_size_received,
                                                                           size_t const input_unit_size_received,
                                                                           T_ const *const ptr_array_inputs_received,
                                                                           struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Code__FC__OpenMP(size_t const time_step_index_received,
                                                                       size_t const batch_size_received,
                                                                       size_t const input_unit_size_received,
                                                                       T_ const *const ptr_array_inputs_received,
                                                                       struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Decode__FC__OpenMP(size_t const time_step_index_received,
                                                                           size_t const batch_size_received,
                                                                           size_t const input_unit_size_received,
                                                                           T_ const *const ptr_array_inputs_received,
                                                                           struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__OpenMP(long long int const time_step_index_received,
                                                           long long int const tmp_time_step_reverse_direction,
                                                           long long int const tmp_time_step_start,
                                                           size_t const batch_size_received,
                                                           size_t const input_unit_size_received,
                                                           T_ const *const ptr_array_inputs_received,
                                                           struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Encode__LSTM__OpenMP(long long int const time_step_index_received,
                                                                         long long int const tmp_time_step_reverse_direction,
                                                                         long long int const tmp_time_step_start,
                                                                         size_t const batch_size_received,
                                                                         size_t const input_unit_size_received,
                                                                         T_ const *const ptr_array_inputs_received,
                                                                         struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Code__LSTM__OpenMP(long long int const time_step_index_received,
                                                                      long long int const tmp_time_step_reverse_direction,
                                                                      long long int const tmp_time_step_start,
                                                                      size_t const batch_size_received,
                                                                      size_t const input_unit_size_received,
                                                                      T_ const *const ptr_array_inputs_received,
                                                                              struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Decode__LSTM__OpenMP(long long int const time_step_index_received,
                                                                         long long int const tmp_time_step_reverse_direction,
                                                                         long long int const tmp_time_step_start,
                                                                         size_t const batch_size_received,
                                                                         size_t const input_unit_size_received,
                                                                         T_ const *const ptr_array_inputs_received,
                                                                         struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Max_Pooling__OpenMP(size_t const time_step_index_received,
                                                                           size_t const batch_size_received,
                                                                           size_t const input_unit_size_received,
                                                                           T_ const *const ptr_array_inputs_received,
                                                                           struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Residual__OpenMP(size_t const batch_size_received, struct Layer *&ptr_layer_it_received);
        void Forward_Pass__Residual__Layer__OpenMP(bool const is_block_input_layer_received,
                                                                                size_t const batch_size_received,
                                                                                struct Layer *&ptr_layer_it_received);
        void Forward_Pass__Residual__FC__OpenMP(bool const is_block_input_layer_received,
                                                                            size_t const time_step_index_received,
                                                                            size_t const batch_size_received,
                                                                            size_t const input_size_received,
                                                                            T_ const *const ptr_array_inputs_received,
                                                                            struct Layer *const ptr_layer_it_received);
        void Forward_Pass__Average_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                size_t const batch_size_received,
                                                                                size_t const input_unit_size_received,
                                                                                size_t const output_size_received,
                                                                                size_t const kernel_size_received,
                                                                                size_t const stride_received,
                                                                                size_t const padding_received,
                                                                                size_t const dilation_received,
                                                                                T_ const *const ptr_array_inputs_received,
                                                                                T_ *const ptr_array_outputs_received);
        void Forward_Pass__Bias__OpenMP(size_t const time_step_index_received,
                                                               size_t const batch_size_received,
                                                               size_t const output_size_received,
                                                               T_ const *const ptr_array_bias_received,
                                                               T_ *const ptr_array_outputs_received);
        void Forward_Pass__FC__OpenMP(size_t const time_step_index_received,
                                                            size_t const batch_size_received,
                                                            size_t const input_size_received,
                                                            size_t const output_size_received,
                                                            T_ const *const ptr_array_inputs_received,
                                                            T_ const *const ptr_array_parameters_received,
                                                            T_ *const ptr_array_outputs_received);
        void Forward_Pass__FC_Ind_RNN__OpenMP(size_t const time_step_index_received,
                                                                           size_t const batch_size_received,
                                                                           size_t const input_size_received,
                                                                           T_ const *const ptr_array_parameters_received,
                                                                           T_ const *const ptr_array_AFs_received,
                                                                           T_ const *const ptr_array_inputs_received,
                                                                           T_ *const ptr_array_outputs_received);
        void Forward_Pass__Batch_Normalization__Inference__OpenMP(size_t const time_step_index_received,
                                                                                                       size_t const batch_size_received,
                                                                                                       size_t const input_size_received,
                                                                                                       T_ const *const ptr_array_inputs_received,
                                                                                                       T_ const *const ptr_array_scales_received,
                                                                                                       T_ const *const ptr_array_shifts_received,
                                                                                                       T_ const *const ptr_array_means_averages_received,
                                                                                                       T_ const *const ptr_array_variances_averages_received,
                                                                                                       T_ *const ptr_array_output_normalizes_received);
        void Forward_Pass__Batch_Normalization__Training__OpenMP(size_t const time_step_index_received,
                                                                                                     size_t const batch_size_received,
                                                                                                     size_t const input_size_received,
                                                                                                     T_ const *const ptr_array_inputs_received,
                                                                                                     T_ const *const ptr_array_scales_received,
                                                                                                     T_ const *const ptr_array_shifts_received,
                                                                                                     T_ *const ptr_array_means_received,
                                                                                                     T_ *const ptr_array_variances_received,
                                                                                                     T_ *const ptr_array_means_averages_received,
                                                                                                     T_ *const ptr_array_variances_averages_received,
                                                                                                     T_ *const ptr_array_output_hats_received,
                                                                                                     T_ *const ptr_array_output_normalizes_received);
        void Forward_Pass__Batch_Renormalization__Training__OpenMP(size_t const time_step_index_received,
                                                                                                        size_t const batch_size_received,
                                                                                                        size_t const input_size_received,
                                                                                                        T_ const *const ptr_array_inputs_received,
                                                                                                        T_ const *const ptr_array_scales_received,
                                                                                                        T_ const *const ptr_array_shifts_received,
                                                                                                        T_ *const ptr_array_means_received,
                                                                                                        T_ *const ptr_array_variances_received,
                                                                                                        T_ *const ptr_array_means_averages_received,
                                                                                                        T_ *const ptr_array_variances_averages_received,
                                                                                                        T_ *const ptr_array_r_corrections_received,
                                                                                                        T_ *const ptr_array_d_corrections_received,
                                                                                                        T_ *const ptr_array_output_hats_received,
                                                                                                        T_ *const ptr_array_output_normalizes_received);
        void Forward_Pass__FC_AF__OpenMP(size_t const time_step_index_received,
                                                                  size_t const batch_size_received,
                                                                  size_t const input_size_received,
                                                                  T_ const *const ptr_array_inputs_received,
                                                                  T_ *const ptr_array_outputs_received,
                                                                  enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const *const ptr_array_type_activations_received);
        void Forward_Pass__FC_AF__Softmax__OpenMP(size_t const time_step_index_received,
                                                                                  size_t const batch_size_received,
                                                                                  size_t const input_size_received,
                                                                                  T_ const *const ptr_array_inputs_received,
                                                                                  T_ *const ptr_array_outputs_received);
        void Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(bool const *const ptr_array__mask__dropout__bernoulli_received,
                                                                                                  size_t const time_step_index_received,
                                                                                                  size_t const batch_size_received,
                                                                                                  size_t const input_size_received,
                                                                                                  T_ const inverse_retention_probability_divided_received,
                                                                                                  T_ *const ptr_array_inputs_received);
        void Forward_Pass__Dropout__Bernoulli__Training__OpenMP(bool const *const ptr_array__mask__dropout__bernoulli_received,
                                                                                                  size_t const time_step_index_received,
                                                                                                  size_t const batch_size_received,
                                                                                                  size_t const input_size_received,
                                                                                                  T_ *const ptr_array_inputs_received);
        void Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(size_t const time_step_index_received,
                                                                                                   size_t const batch_size_received,
                                                                                                   size_t const input_size_received,
                                                                                                   T_ const retention_probability_received,
                                                                                                   T_ *const ptr_array_inputs_received);
        void Forward_Pass__Dropout__Gaussian__OpenMP(size_t const time_step_index_received,
                                                                                     size_t const batch_size_received,
                                                                                     size_t const input_size_received,
                                                                                     T_ const variance_received,
                                                                                     T_ *const ptr_array_inputs_received);
        void Forward_Pass__Dropout__ShakeDrop__OpenMP(size_t const time_step_index_received,
                                                                                       size_t const batch_size_received,
                                                                                       size_t const input_size_received,
                                                                                       bool *const ptr_array_mask_dopout_shakedrop_received,
                                                                                       T_ const lower_bound_received,
                                                                                       T_ const upper_bound_received,
                                                                                       T_ const dropout_probability_received,
                                                                                       T_ *const ptr_array_inputs_received);
        void Forward_Pass__Dropout__Uout__OpenMP(size_t const time_step_index_received,
                                                                              size_t const batch_size_received,
                                                                              size_t const input_size_received,
                                                                              T_ const beta_received,
                                                                              T_ *const ptr_array_inputs_received);
        void Forward_Pass__Max_Pooling__OpenMP(size_t const time_step_index_received,
                                                                           size_t const batch_size_received,
                                                                           size_t const input_size_received,
                                                                           size_t const output_size_received,
                                                                           size_t const kernel_size_received,
                                                                           size_t const stride_received,
                                                                           size_t const padding_received,
                                                                           size_t const dilation_received,
                                                                           size_t *const ptr_array_indices_received,
                                                                           T_ const *const ptr_array_inputs_received,
                                                                           T_ *const ptr_array_outputs_received);
        void Forward_Pass__Zero_Padded_Identity__OpenMP(size_t const time_step_index_received,
                                                                                       size_t const batch_size_received,
                                                                                       size_t const A_unit_size_received,
                                                                                       size_t const B_unit_size_received,
                                                                                       size_t const padding_received,
                                                                                       T_ const *const ptr_array_A_received,
                                                                                       T_ const *const ptr_array_B_received,
                                                                                       T_ *const ptr_array_outputs_received);
        void RNN__Forward_Pass_Batch__Loop(size_t const batch_size_received,
                                                                   T_ const *const *const ptr_array_inputs_received,
                                                                   struct Layer *const ptr_first_layer_received,
                                                                   struct Layer const *const ptr_last_layer_received);
        void RNN__Forward_Pass_Batch__Pre_Training__Loop(size_t const batch_size_received, T_ const *const *const ptr_array_inputs_received);
        void Recurrent__Forward_Pass__Average_Pooling__Loop(size_t const batch_size_received,
                                                                                            size_t const input_unit_size_received,
                                                                                            T_ const *const ptr_array_inputs_received,
                                                                                            struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Dropout__Bernoulli__Inference__Loop(size_t const batch_size_received,
                                                                                                                size_t const input_unit_size_received,
                                                                                                                T_ const retention_probability_received,
                                                                                                                T_ *const ptr_array_inputs_received);
        void Recurrent__Forward_Pass__Dropout__ShakeDrop__Loop(size_t const batch_size_received,
                                                                                                   size_t const input_unit_size_received,
                                                                                                   bool *const ptr_array_mask_dopout_shakedrop_received,
                                                                                                   T_ const lower_bound_received,
                                                                                                   T_ const upper_bound_received,
                                                                                                   T_ const dropout_probability_received,
                                                                                                   T_ *const ptr_array_inputs_received);
        void Recurrent__Forward_Pass__FC__Loop(size_t const batch_size_received,
                                                                        size_t const input_unit_size_received,
                                                                        T_ const *const ptr_array_inputs_received,
                                                                        struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Encode__FC__Loop(size_t const batch_size_received,
                                                                                       size_t const input_unit_size_received,
                                                                                       T_ const *const ptr_array_inputs_received,
                                                                                       struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Code__FC__Loop(size_t const batch_size_received,
                                                                                   size_t const input_unit_size_received,
                                                                                   T_ const *const ptr_array_inputs_received,
                                                                                   struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Decode__FC__Loop(size_t const batch_size_received,
                                                                                       size_t const input_unit_size_received,
                                                                                       T_ const *const ptr_array_inputs_received,
                                                                                       struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__LSTM__Loop(bool const forward_layer_received,
                                                                            size_t const batch_size_received,
                                                                            size_t const input_unit_size_received,
                                                                            T_ const *const ptr_array_inputs_received,
                                                                            struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Encode__LSTM__Loop(bool const forward_layer_received,
                                                                                           size_t const batch_size_received,
                                                                                           size_t const input_unit_size_received,
                                                                                           T_ const *const ptr_array_inputs_received,
                                                                                           struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Code__LSTM__Loop(bool const forward_layer_received,
                                                                                       size_t const batch_size_received,
                                                                                       size_t const input_unit_size_received,
                                                                                       T_ const *const ptr_array_inputs_received,
                                                                                       struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Decode__LSTM__Loop(bool const forward_layer_received,
                                                                                           size_t const batch_size_received,
                                                                                           size_t const input_unit_size_received,
                                                                                           T_ const *const ptr_array_inputs_received,
                                                                                           struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Max_Pooling__Loop(size_t const batch_size_received,
                                                                                       size_t const input_unit_size_received,
                                                                                       T_ const *const ptr_array_inputs_received,
                                                                                       struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Residual__Loop(size_t const batch_size_received, struct Layer *&ptr_layer_it_received);
        void Recurrent__Forward_Pass__Residual__Layer__Loop(bool const is_block_input_layer_received,
                                                                                            size_t const batch_size_received,
                                                                                            struct Layer *&ptr_layer_it_received);
        void Recurrent__Forward_Pass__Residual__FC__Loop(bool const is_block_input_layer_received,
                                                                                        size_t const batch_size_received,
                                                                                        size_t const input_unit_size_received,
                                                                                        T_ const *const ptr_array_inputs_received,
                                                                                        struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Zero_Padded_Identity__Loop(size_t const batch_size_received,
                                                                                                   size_t const size_A_received,
                                                                                                   size_t const size_B_received,
                                                                                                   T_ const *const ptr_array_A_received,
                                                                                                   T_ const *const ptr_array_B_received,
                                                                                                   struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__Gates_CIFO__Loop(long long int const time_step_index_received,
                                                                                long long int const time_step_reverse_direction_received,
                                                                                long long int const time_step_prediction_start_received,
                                                                                size_t const batch_size_received,
                                                                                size_t const block_unit_size_received,
                                                                                size_t const cell_unit_size_received,
                                                                                size_t const input_unit_size_received,
                                                                                T_ const *const ptr_array_inputs_received,
                                                                                struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__Gates_CIF_AF_State__Loop(long long int const time_step_index_received,
                                                                                             long long int const time_step_reverse_direction_received,
                                                                                             long long int const time_step_prediction_start_received,
                                                                                             size_t const batch_size_received,
                                                                                             size_t const block_unit_size_received,
                                                                                             size_t const cell_unit_size_received,
                                                                                             T_ const *const ptr_array_summation_input_block_inputs_received,
                                                                                             T_ const *const ptr_array_summation_recurrent_block_inputs_received,
                                                                                             T_ const *const ptr_array_summation_input_inputs_gates_received,
                                                                                             T_ const *const ptr_array_summation_recurrent_inputs_gates_received,
                                                                                             T_ const *const ptr_array_summation_input_forgets_gates_received,
                                                                                             T_ const *const ptr_array_summation_recurrent_forgets_gates_received,
                                                                                             struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__Gates_CIF_AF_State__Zoneout__Loop(long long int const time_step_index_received,
                                                                                                             long long int const time_step_reverse_direction_received,
                                                                                                             long long int const time_step_prediction_start_received,
                                                                                                             size_t const batch_size_received,
                                                                                                             size_t const block_unit_size_received,
                                                                                                             size_t const cell_unit_size_received,
                                                                                                             T_ const *const ptr_array_summation_input_block_inputs_received,
                                                                                                             T_ const *const ptr_array_summation_recurrent_block_inputs_received,
                                                                                                             T_ const *const ptr_array_summation_input_inputs_gates_received,
                                                                                                             T_ const *const ptr_array_summation_recurrent_inputs_gates_received,
                                                                                                             T_ const *const ptr_array_summation_input_forgets_gates_received,
                                                                                                             T_ const *const ptr_array_summation_recurrent_forgets_gates_received,
                                                                                                             struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__Output__Loop(long long int const time_step_index_received,
                                                                        size_t const batch_size_received,
                                                                        size_t const block_unit_size_received,
                                                                        size_t const cell_unit_size_received,
                                                                        T_ const *const ptr_array_summation_input_outputs_gates_received,
                                                                        T_ const *const ptr_array_summation_recurrent_outputs_gates_received,
                                                                        struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__Output__Zoneout__Loop(long long int const time_step_index_received,
                                                                                       long long int const time_step_reverse_direction_received,
                                                                                       long long int const time_step_prediction_start_received,
                                                                                       size_t const batch_size_received,
                                                                                       size_t const block_unit_size_received,
                                                                                       size_t const cell_unit_size_received,
                                                                                       T_ const *const ptr_array_summation_input_outputs_gates_received,
                                                                                       T_ const *const ptr_array_summation_recurrent_outputs_gates_received,
                                                                                       struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__States_AF__Loop(long long int const time_step_index_received,
                                                                              size_t const batch_size_received,
                                                                              size_t const block_unit_size_received,
                                                                              size_t const cell_unit_size_received,
                                                                              T_ const *const ptr_array_summation_cell_states_received,
                                                                              struct Layer *const ptr_layer_it_received);
        void RNN__Forward_Pass_Batch__OpenMP(size_t const batch_size_received,
                                                                         T_ const *const *const ptr_array_inputs_received,
                                                                         struct Layer *const ptr_first_layer_received,
                                                                         struct Layer const *const ptr_last_layer_received);
        void RNN__Forward_Pass_Batch__Pre_Training__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_array_inputs_received);
        void Recurrent__Forward_Pass__Average_Pooling__OpenMP(size_t const batch_size_received,
                                                                                                  size_t const input_unit_size_received,
                                                                                                  T_ const *const ptr_array_inputs_received,
                                                                                                  struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(size_t const batch_size_received,
                                                                                                                      size_t const input_unit_size_received,
                                                                                                                      T_ const retention_probability_received,
                                                                                                                      T_ *const ptr_array_inputs_received);
        void Recurrent__Forward_Pass__Dropout__ShakeDrop__OpenMP(size_t const batch_size_received,
                                                                                                        size_t const input_unit_size_received,
                                                                                                        bool *const ptr_array_mask_dopout_shakedrop_received,
                                                                                                        T_ const lower_bound_received,
                                                                                                        T_ const upper_bound_received,
                                                                                                        T_ const dropout_probability_received,
                                                                                                        T_ *const ptr_array_inputs_received);
        void Recurrent__Forward_Pass__FC__OpenMP(size_t const batch_size_received,
                                                                                   size_t const input_unit_size_received,
                                                                                   T_ const *const ptr_array_inputs_received,
                                                                                   struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Encode__FC__OpenMP(size_t const batch_size_received,
                                                                                                 size_t const input_unit_size_received,
                                                                                                 T_ const *const ptr_array_inputs_received,
                                                                                                 struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Code__FC__OpenMP(size_t const batch_size_received,
                                                                                              size_t const input_unit_size_received,
                                                                                              T_ const *const ptr_array_inputs_received,
                                                                                              struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Decode__FC__OpenMP(size_t const batch_size_received,
                                                                                                 size_t const input_unit_size_received,
                                                                                                 T_ const *const ptr_array_inputs_received,
                                                                                                 struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__LSTM__OpenMP(bool const forward_layer_received,
                                                                                  size_t const batch_size_received,
                                                                                  size_t const input_unit_size_received,
                                                                                  T_ const *const ptr_array_inputs_received,
                                                                                  struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Encode__LSTM__OpenMP(bool const forward_layer_received,
                                                                                                 size_t const batch_size_received,
                                                                                                 size_t const input_unit_size_received,
                                                                                                 T_ const *const ptr_array_inputs_received,
                                                                                                 struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Code__LSTM__OpenMP(bool const forward_layer_received,
                                                                                             size_t const batch_size_received,
                                                                                             size_t const input_unit_size_received,
                                                                                             T_ const *const ptr_array_inputs_received,
                                                                                             struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Decode__LSTM__OpenMP(bool const forward_layer_received,
                                                                                                 size_t const batch_size_received,
                                                                                                 size_t const input_unit_size_received,
                                                                                                 T_ const *const ptr_array_inputs_received,
                                                                                                 struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Max_Pooling__OpenMP(size_t const batch_size_received,
                                                                                            size_t const input_unit_size_received,
                                                                                            T_ const *const ptr_array_inputs_received,
                                                                                            struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Residual__OpenMP(size_t const batch_size_received, struct Layer *&ptr_layer_it_received);
        void Recurrent__Forward_Pass__Residual__Layer__OpenMP(bool const is_block_input_layer_received,
                                                                                                  size_t const batch_size_received,
                                                                                                  struct Layer *&ptr_layer_it_received);
        void Recurrent__Forward_Pass__Residual__FC__OpenMP(bool const is_block_input_layer_received,
                                                                                              size_t const batch_size_received,
                                                                                              size_t const input_unit_size_received,
                                                                                              T_ const *const ptr_array_inputs_received,
                                                                                              struct Layer *const ptr_layer_it_received);
        void Recurrent__Forward_Pass__Zero_Padded_Identity__OpenMP(size_t const batch_size_received,
                                                                                                         size_t const size_A_received,
                                                                                                         size_t const size_B_received,
                                                                                                         T_ const *const ptr_array_A_received,
                                                                                                         T_ const *const ptr_array_B_received,
                                                                                                         struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__Gates_CIFO__OpenMP(long long int const time_step_index_received,
                                                                                      long long int const time_step_reverse_direction_received,
                                                                                      long long int const time_step_prediction_start_received,
                                                                                      size_t const batch_size_received,
                                                                                      size_t const block_unit_size_received,
                                                                                      size_t const cell_unit_size_received,
                                                                                      size_t const input_unit_size_received,
                                                                                      T_ const *const ptr_array_inputs_received,
                                                                                      struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__Gates_CIF_AF_State__OpenMP(long long int const time_step_index_received,
                                                                                                   long long int const time_step_reverse_direction_received,
                                                                                                   long long int const time_step_prediction_start_received,
                                                                                                   size_t const batch_size_received,
                                                                                                   size_t const block_unit_size_received,
                                                                                                   size_t const cell_unit_size_received,
                                                                                                   T_ const *const ptr_array_summation_input_block_inputs_received,
                                                                                                   T_ const *const ptr_array_summation_recurrent_block_inputs_received,
                                                                                                   T_ const *const ptr_array_summation_input_inputs_gates_received,
                                                                                                   T_ const *const ptr_array_summation_recurrent_inputs_gates_received,
                                                                                                   T_ const *const ptr_array_summation_input_forgets_gates_received,
                                                                                                   T_ const *const ptr_array_summation_recurrent_forgets_gates_received,
                                                                                                   struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__Gates_CIF_AF_State__Zoneout__OpenMP(long long int const time_step_index_received,
                                                                                                                  long long int const time_step_reverse_direction_received,
                                                                                                                  long long int const time_step_prediction_start_received,
                                                                                                                  size_t const batch_size_received,
                                                                                                                  size_t const block_unit_size_received,
                                                                                                                  size_t const cell_unit_size_received,
                                                                                                                  T_ const *const ptr_array_summation_input_block_inputs_received,
                                                                                                                  T_ const *const ptr_array_summation_recurrent_block_inputs_received,
                                                                                                                  T_ const *const ptr_array_summation_input_inputs_gates_received,
                                                                                                                  T_ const *const ptr_array_summation_recurrent_inputs_gates_received,
                                                                                                                  T_ const *const ptr_array_summation_input_forgets_gates_received,
                                                                                                                  T_ const *const ptr_array_summation_recurrent_forgets_gates_received,
                                                                                                                  struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__Output__OpenMP(long long int const time_step_index_received,
                                                                             size_t const batch_size_received,
                                                                             size_t const block_unit_size_received,
                                                                             size_t const cell_unit_size_received,
                                                                             T_ const *const ptr_array_summation_input_outputs_gates_received,
                                                                             T_ const *const ptr_array_summation_recurrent_outputs_gates_received,
                                                                             struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__Output__Zoneout__OpenMP(long long int const time_step_index_received,
                                                                                             long long int const time_step_reverse_direction_received,
                                                                                             long long int const time_step_prediction_start_received,
                                                                                             size_t const batch_size_received,
                                                                                             size_t const block_unit_size_received,
                                                                                             size_t const cell_unit_size_received,
                                                                                             T_ const *const ptr_array_summation_input_outputs_gates_received,
                                                                                             T_ const *const ptr_array_summation_recurrent_outputs_gates_received,
                                                                                             struct Layer *const ptr_layer_it_received);
        void Forward_Pass__LSTM__States_AF__OpenMP(long long int const time_step_index_received,
                                                                                    size_t const batch_size_received,
                                                                                    size_t const block_unit_size_received,
                                                                                    size_t const cell_unit_size_received,
                                                                                    T_ const *const ptr_array_summation_cell_states_received,
                                                                                    struct Layer *const ptr_layer_it_received);
        void Backward_Pass(size_t const batch_size_received);
        void Backward_Pass__Pre_Training(size_t const batch_size_received);
        void FF__Backward_Pass_Batch__Loop(size_t const batch_size_received);
        void FF__Backward_Pass_Batch__Pre_Training__Loop(size_t const batch_size_received);
        void Backward_Pass__FC__Loop(size_t const batch_size_received,
                                                         size_t const derivative_size_received,
                                                         T_ *const ptr_array_derivatives_received,
                                                         struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Average_Pooling__Loop(size_t const time_step_index_received,
                                                                             size_t const batch_size_received,
                                                                             size_t const derivative_size_received,
                                                                             T_ *const ptr_array_derivatives_received,
                                                                             struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__FC__Loop(size_t const time_step_index_received,
                                                         size_t const batch_size_received,
                                                         size_t const derivative_size_received,
                                                         T_ *const ptr_array_derivatives_received,
                                                         struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Max_Pooling__Loop(size_t const time_step_index_received,
                                                                        size_t const batch_size_received,
                                                                        size_t const derivative_size_received,
                                                                        T_ *const ptr_array_derivatives_received,
                                                                        struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Residual__Loop(size_t const time_step_index_received,
                                                                  size_t const batch_size_received,
                                                                  size_t const derivative_size_received,
                                                                  T_ *const ptr_array_derivatives_received,
                                                                  struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Residual__Block__Loop(size_t const time_step_index_received,
                                                                             size_t const batch_size_received,
                                                                             size_t const derivative_size_received,
                                                                             T_ *const ptr_array_derivatives_received,
                                                                             struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Residual__FC__Loop(size_t const time_step_index_received,
                                                                         size_t const batch_size_received,
                                                                         size_t const derivative_size_received,
                                                                         T_ *const ptr_array_derivatives_received,
                                                                         struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Gradient__FC__Loop(size_t const time_step_index_received,
                                                                         size_t const batch_size_received,
                                                                         struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Gradient__Residual__Loop(size_t const batch_size_received, struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Gradient__Residual__Layer__Loop(bool const is_block_input_layer_received,
                                                                                             size_t const batch_size_received,
                                                                                             struct Layer *&ptr_layer_it_received);
        void Backward_Pass__Gradient__Residual__FC__Loop(bool const is_block_input_layer_received,
                                                                                         size_t const time_step_index_received,
                                                                                         size_t const batch_size_received,
                                                                                         struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Average_Pooling__Loop(size_t const time_step_index_received,
                                                                             size_t const batch_size_received,
                                                                             size_t const input_size_received,
                                                                             size_t const derivative_size_received,
                                                                             size_t const kernel_size_received,
                                                                             size_t const stride_received,
                                                                             size_t const padding_received,
                                                                             size_t const dilation_received,
                                                                             T_ const *const ptr_array_derivative_inputs_received,
                                                                             T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Dropout__ShakeDrop__Loop(size_t const time_step_index_received,
                                                                                    size_t const batch_size_received,
                                                                                    size_t const derivative_size_received,
                                                                                    bool const *const ptr_array_mask_dopout_shakedrop_received,
                                                                                    T_ const lower_bound_received,
                                                                                    T_ const upper_bound_received,
                                                                                    T_ *const ptr_array_derivatives_received);
        void Backward_Pass__FC__Loop(size_t const time_step_index_received,
                                                         size_t const batch_size_received,
                                                         size_t const input_size_received,
                                                         size_t const derivative_size_received,
                                                         T_ const *const ptr_array_derivative_inputs_received,
                                                         T_ const *const ptr_array_parameters_received,
                                                         T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Identity__Loop(size_t const time_step_index_received,
                                                                size_t const batch_size_received,
                                                                size_t const input_size_received,
                                                                T_ const *const ptr_array_derivative_inputs_received,
                                                                T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Max_Pooling__Loop(size_t const time_step_index_received,
                                                                        size_t const batch_size_received,
                                                                        size_t const input_size_received,
                                                                        size_t const derivative_size_received,
                                                                        size_t const padding_received,
                                                                        size_t const *const ptr_array_indices_received,
                                                                        T_ const *const ptr_array_derivative_inputs_received,
                                                                        T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Residual__Loop(size_t const time_step_index_received,
                                                                  size_t const batch_size_received,
                                                                  size_t const input_size_received,
                                                                  size_t const derivative_size_received,
                                                                  size_t const padding_received,
                                                                  T_ const *const ptr_array_derivative_inputs_received,
                                                                  T_ *const ptr_array_derivatives_received);
        void Backward_Pass__FC__DF__Loop(size_t const time_step_index_received,
                                                                 size_t const batch_size_received,
                                                                 size_t const input_size_received,
                                                                 enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const *const ptr_array_type_activations_functions_received,
                                                                 T_ const *const ptr_array_activations_steepness_received,
                                                                 T_ const *const ptr_array_pre_AFs_received,
                                                                 T_ const *const ptr_array_AFs_received,
                                                                 T_ const *const ptr_array_derivative_inputs_received,
                                                                 T_ *const ptr_array_derivatives_received);
        void Backward_Pass__FC__DF_Ind_RNN__Loop(size_t const time_step_index_received,
                                                                                size_t const batch_size_received,
                                                                                size_t const input_size_received,
                                                                                T_ const *const ptr_array_parameters_received,
                                                                                enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const *const ptr_array_type_activations_functions_received,
                                                                                T_ const *const ptr_array_activations_steepness_received,
                                                                                T_ const *const ptr_array_pre_AFs_received,
                                                                                T_ const *const ptr_array_AFs_received,
                                                                                T_ const *const ptr_array_derivative_inputs_received,
                                                                                T_ *const ptr_array_dAFs_received,
                                                                                T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Batch_Normalization__Loop(size_t const time_step_index_received,
                                                                                   size_t const batch_size_received,
                                                                                   size_t const input_size_received,
                                                                                   T_ const *const ptr_array_means_received,
                                                                                   T_ const *const ptr_array_variances_received,
                                                                                   T_ const *const ptr_array_scales_received,
                                                                                   T_ const *const ptr_array_inputs_received,
                                                                                   T_ const *const ptr_array_inputs_hats_received,
                                                                                   T_ const *const ptr_array_derivative_inputs_received,
                                                                                   T_ *const ptr_array_derivatives_scales_received,
                                                                                   T_ *const ptr_array_derivatives_shifts_received,
                                                                                   T_ *const ptr_array_derivatives_means_received,
                                                                                   T_ *const ptr_array_derivatives_variances_received,
                                                                                   T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Batch_Normalization__Loop(size_t const time_step_index_received,
                                                                                   size_t const batch_size_received,
                                                                                   size_t const input_size_received,
                                                                                   T_ const *const ptr_array_means_received,
                                                                                   T_ const *const ptr_array_variances_received,
                                                                                   T_ const *const ptr_array_scales_received,
                                                                                   T_ const *const ptr_array_inputs_received,
                                                                                   T_ const *const ptr_array_inputs_hats_received,
                                                                                   T_ const *const ptr_array_derivative_inputs_received,
                                                                                   T_ *const ptr_array_derivatives_scales_received,
                                                                                   T_ *const ptr_array_derivatives_means_received,
                                                                                   T_ *const ptr_array_derivatives_variances_received,
                                                                                   T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Batch_Renormalization__Loop(size_t const time_step_index_received,
                                                                                      size_t const batch_size_received,
                                                                                      size_t const input_size_received,
                                                                                      T_ const *const ptr_array_means_received,
                                                                                      T_ const *const ptr_array_variances_received,
                                                                                      T_ const *const ptr_array_scales_received,
                                                                                      T_ const *const ptr_array_r_corrections_received,
                                                                                      T_ const *const ptr_array_inputs_received,
                                                                                      T_ const *const ptr_array_inputs_hats_received,
                                                                                      T_ const *const ptr_array_derivative_inputs_received,
                                                                                      T_ *const ptr_array_derivatives_scales_received,
                                                                                      T_ *const ptr_array_derivatives_shifts_received,
                                                                                      T_ *const ptr_array_derivatives_means_received,
                                                                                      T_ *const ptr_array_derivatives_variances_received,
                                                                                      T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Batch_Renormalization__Loop(size_t const time_step_index_received,
                                                                                      size_t const batch_size_received,
                                                                                      size_t const input_size_received,
                                                                                      T_ const *const ptr_array_means_received,
                                                                                      T_ const *const ptr_array_variances_received,
                                                                                      T_ const *const ptr_array_scales_received,
                                                                                      T_ const *const ptr_array_r_corrections_received,
                                                                                      T_ const *const ptr_array_inputs_received,
                                                                                      T_ const *const ptr_array_inputs_hats_received,
                                                                                      T_ const *const ptr_array_derivative_inputs_received,
                                                                                      T_ *const ptr_array_derivatives_scales_received,
                                                                                      T_ *const ptr_array_derivatives_means_received,
                                                                                      T_ *const ptr_array_derivatives_variances_received,
                                                                                      T_ *const ptr_array_derivatives_received);
        void FF__Backward_Pass_Batch__OpenMP(size_t const batch_size_received);
        void FF__Backward_Pass_Batch__Pre_Training__OpenMP(size_t const batch_size_received);
        void Backward_Pass__FC__OpenMP(size_t const batch_size_received,
                                                               size_t const derivative_size_received,
                                                               T_ *const ptr_array_derivatives_received,
                                                               struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Average_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                   size_t const batch_size_received,
                                                                                   size_t const derivative_size_received,
                                                                                   T_ *const ptr_array_derivatives_received,
                                                                                   struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__FC__OpenMP(size_t const time_step_index_received,
                                                               size_t const batch_size_received,
                                                               size_t const derivative_size_received,
                                                               T_ *const ptr_array_derivatives_received,
                                                               struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Max_Pooling__OpenMP(size_t const time_step_index_received,
                                                                             size_t const batch_size_received,
                                                                             size_t const derivative_size_received,
                                                                             T_ *const ptr_array_derivatives_received,
                                                                             struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Residual__OpenMP(size_t const time_step_index_received,
                                                                       size_t const batch_size_received,
                                                                       size_t const derivative_size_received,
                                                                       T_ *const ptr_array_derivatives_received,
                                                                       struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Residual__Block__OpenMP(size_t const time_step_index_received,
                                                                                   size_t const batch_size_received,
                                                                                   size_t const derivative_size_received,
                                                                                   T_ *const ptr_array_derivatives_received,
                                                                                   struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Residual__FC__OpenMP(size_t const time_step_index_received,
                                                                               size_t const batch_size_received,
                                                                               size_t const derivative_size_received,
                                                                               T_ *const ptr_array_derivatives_received,
                                                                               struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Gradient__FC__OpenMP(size_t const time_step_index_received,
                                                                               size_t const batch_size_received,
                                                                               struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Gradient__Residual__OpenMP(size_t const batch_size_received, struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Gradient__Residual__Layer__OpenMP(bool const is_block_input_layer_received,
                                                                                                   size_t const batch_size_received,
                                                                                                   struct Layer *&ptr_layer_it_received);
        void Backward_Pass__Gradient__Residual__FC__OpenMP(bool const is_block_input_layer_received,
                                                                                               size_t const time_step_index_received,
                                                                                               size_t const batch_size_received,
                                                                                               struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__Average_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                   size_t const batch_size_received,
                                                                                   size_t const input_size_received,
                                                                                   size_t const derivative_size_received,
                                                                                   size_t const kernel_size_received,
                                                                                   size_t const stride_received,
                                                                                   size_t const padding_received,
                                                                                   size_t const dilation_received,
                                                                                   T_ const *const ptr_array_derivative_inputs_received,
                                                                                   T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Dropout__ShakeDrop__OpenMP(size_t const time_step_index_received,
                                                                                          size_t const batch_size_received,
                                                                                          size_t const derivative_size_received,
                                                                                          bool const *const ptr_array_mask_dopout_shakedrop_received,
                                                                                          T_ const lower_bound_received,
                                                                                          T_ const upper_bound_received,
                                                                                          T_ *const ptr_array_derivatives_received);
        void Backward_Pass__FC__OpenMP(size_t const time_step_index_received,
                                                               size_t const batch_size_received,
                                                               size_t const input_size_received,
                                                               size_t const derivative_size_received,
                                                               T_ const *const ptr_array_derivative_inputs_received,
                                                               T_ const *const ptr_array_parameters_received,
                                                               T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Identity__OpenMP(size_t const time_step_index_received,
                                                                     size_t const batch_size_received,
                                                                     size_t const input_size_received,
                                                                     T_ const *const ptr_array_derivative_inputs_received,
                                                                     T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Max_Pooling__OpenMP(size_t const time_step_index_received,
                                                                             size_t const batch_size_received,
                                                                             size_t const input_size_received,
                                                                             size_t const derivative_size_received,
                                                                             size_t const padding_received,
                                                                             size_t const *const ptr_array_indices_received,
                                                                             T_ const *const ptr_array_derivative_inputs_received,
                                                                             T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Residual__OpenMP(size_t const time_step_index_received,
                                                                       size_t const batch_size_received,
                                                                       size_t const input_size_received,
                                                                       size_t const derivative_size_received,
                                                                       size_t const padding_received,
                                                                       T_ const *const ptr_array_derivative_inputs_received,
                                                                       T_ *const ptr_array_derivatives_received);
        void Backward_Pass__FC__DF__OpenMP(size_t const time_step_index_received,
                                                                       size_t const batch_size_received,
                                                                       size_t const input_size_received,
                                                                       enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const *const ptr_array_type_activations_functions_received,
                                                                       T_ const *const ptr_array_activations_steepness_received,
                                                                       T_ const *const ptr_array_pre_AFs_received,
                                                                       T_ const *const ptr_array_AFs_received,
                                                                       T_ const *const ptr_array_derivative_inputs_received,
                                                                       T_ *const ptr_array_derivatives_received);
        void Backward_Pass__FC__DF_Ind_RNN__OpenMP(size_t const time_step_index_received,
                                                                                     size_t const batch_size_received,
                                                                                     size_t const input_size_received,
                                                                                     T_ const *const ptr_array_parameters_received,
                                                                                     enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const *const ptr_array_type_activations_functions_received,
                                                                                     T_ const *const ptr_array_activations_steepness_received,
                                                                                     T_ const *const ptr_array_pre_AFs_received,
                                                                                     T_ const *const ptr_array_AFs_received,
                                                                                     T_ const *const ptr_array_derivative_inputs_received,
                                                                                     T_ *const ptr_array_dAFs_received,
                                                                                     T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Batch_Normalization__OpenMP(size_t const time_step_index_received,
                                                                                       size_t const batch_size_received,
                                                                                       size_t const input_size_received,
                                                                                       T_ const *const ptr_array_means_received,
                                                                                       T_ const *const ptr_array_variances_received,
                                                                                       T_ const *const ptr_array_scales_received,
                                                                                       T_ const *const ptr_array_inputs_received,
                                                                                       T_ const *const ptr_array_inputs_hats_received,
                                                                                       T_ const *const ptr_array_derivative_inputs_received,
                                                                                       T_ *const ptr_array_derivatives_scales_received,
                                                                                       T_ *const ptr_array_derivatives_shifts_received,
                                                                                       T_ *const ptr_array_derivatives_means_received,
                                                                                       T_ *const ptr_array_derivatives_variances_received,
                                                                                       T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Batch_Normalization__OpenMP(size_t const time_step_index_received,
                                                                                       size_t const batch_size_received,
                                                                                       size_t const input_size_received,
                                                                                       T_ const *const ptr_array_means_received,
                                                                                       T_ const *const ptr_array_variances_received,
                                                                                       T_ const *const ptr_array_scales_received,
                                                                                       T_ const *const ptr_array_inputs_received,
                                                                                       T_ const *const ptr_array_inputs_hats_received,
                                                                                       T_ const *const ptr_array_derivative_inputs_received,
                                                                                       T_ *const ptr_array_derivatives_scales_received,
                                                                                       T_ *const ptr_array_derivatives_means_received,
                                                                                       T_ *const ptr_array_derivatives_variances_received,
                                                                                       T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Batch_Renormalization__OpenMP(size_t const time_step_index_received,
                                                                                          size_t const batch_size_received,
                                                                                          size_t const input_size_received,
                                                                                          T_ const *const ptr_array_means_received,
                                                                                          T_ const *const ptr_array_variances_received,
                                                                                          T_ const *const ptr_array_scales_received,
                                                                                          T_ const *const ptr_array_r_corrections_received,
                                                                                          T_ const *const ptr_array_inputs_received,
                                                                                          T_ const *const ptr_array_inputs_hats_received,
                                                                                          T_ const *const ptr_array_derivative_inputs_received,
                                                                                          T_ *const ptr_array_derivatives_scales_received,
                                                                                          T_ *const ptr_array_derivatives_shifts_received,
                                                                                          T_ *const ptr_array_derivatives_means_received,
                                                                                          T_ *const ptr_array_derivatives_variances_received,
                                                                                          T_ *const ptr_array_derivatives_received);
        void Backward_Pass__Batch_Renormalization__OpenMP(size_t const time_step_index_received,
                                                                                          size_t const batch_size_received,
                                                                                          size_t const input_size_received,
                                                                                          T_ const *const ptr_array_means_received,
                                                                                          T_ const *const ptr_array_variances_received,
                                                                                          T_ const *const ptr_array_scales_received,
                                                                                          T_ const *const ptr_array_r_corrections_received,
                                                                                          T_ const *const ptr_array_inputs_received,
                                                                                          T_ const *const ptr_array_inputs_hats_received,
                                                                                          T_ const *const ptr_array_derivative_inputs_received,
                                                                                          T_ *const ptr_array_derivatives_scales_received,
                                                                                          T_ *const ptr_array_derivatives_means_received,
                                                                                          T_ *const ptr_array_derivatives_variances_received,
                                                                                          T_ *const ptr_array_derivatives_received);
        void RNN__Backward_Pass_Batch__Loop(size_t const batch_size_received);
        void RNN__Backward_Pass_Batch__Pre_Training__Loop(size_t const batch_size_received);
        void Recurrent__Backward_Pass__Average_Pooling__Loop(size_t const batch_size_received,
                                                                                               size_t const derivative_input_size_received,
                                                                                               T_ *const ptr_array_derivative_inputs_received,
                                                                                               struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__FC__Loop(size_t const batch_size_received,
                                                                           size_t const derivative_input_size_received,
                                                                           T_ *const ptr_array_derivative_inputs_received,
                                                                           struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__LSTM__Loop(size_t const batch_size_received,
                                                                               size_t const derivative_input_size_received,
                                                                               T_ *const ptr_array_derivative_inputs_received,
                                                                               struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Max_Pooling__Loop(size_t const batch_size_received,
                                                                                         size_t const derivative_input_size_received,
                                                                                         T_ *const ptr_array_derivative_inputs_received,
                                                                                         struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Residual__Loop(size_t const batch_size_received,
                                                                                   size_t const derivative_input_size_received,
                                                                                   T_ *const ptr_array_derivative_inputs_received,
                                                                                   struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Residual__Block__Loop(size_t const batch_size_received,
                                                                                               size_t const derivative_input_size_received,
                                                                                               T_ *const ptr_array_derivative_inputs_received,
                                                                                               struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Gradient__FC__Loop(size_t const batch_size_received, struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Gradient__LSTM__Loop(bool const forward_layer_received,
                                                                                               size_t const batch_size_received,
                                                                                               size_t const derivative_input_size_received,
                                                                                               T_ *const ptr_array_derivative_inputs_received,
                                                                                               struct Layer *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Gradient__Residual__Loop(size_t const batch_size_received, struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Gradient__Residual__Layer__Loop(bool const is_block_input_layer_received,
                                                                                                              size_t const batch_size_received,
                                                                                                              struct Layer *&ptr_layer_it_received);
        void Recurrent__Backward_Pass__Gradient__Residual__FC__Loop(bool const is_block_input_layer_received,
                                                                                                           size_t const batch_size_received,
                                                                                                           struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__LSTM__Loop(size_t const time_step_index_received,
                                                              size_t const batch_size_received,
                                                              size_t const derivative_input_size_received,
                                                              T_ const *const ptr_array_delta_input_block_inputs_received,
                                                              T_ const *const ptr_array_delta_input_input_gates_received,
                                                              T_ const *const ptr_array_delta_input_forget_gates_received,
                                                              T_ const *const ptr_array_delta_input_output_gates_received,
                                                              T_ *const ptr_array_derivative_inputs_received,
                                                              struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__LSTM_Derivative__Output__Loop(long long int const time_step_index_received,
                                                                                          long long int const time_step_direction_received,
                                                                                          long long int const time_step_prediction_end_received,
                                                                                          size_t const batch_size_received,
                                                                                          size_t const block_unit_size_received,
                                                                                          size_t const cell_unit_size_received,
                                                                                          T_ const *const ptr_array_delta_recurrent_block_inputs_received,
                                                                                          T_ const *const ptr_array_delta_recurrent_input_gates_received,
                                                                                          T_ const *const ptr_array_delta_recurrent_forget_gates_received,
                                                                                          T_ const *const ptr_array_delta_recurrent_output_gates_received,
                                                                                          struct Layer *const ptr_layer_it_received);
        void Backward_Pass__LSTM_Derivative__Cell_State_AF__Loop(long long int const time_step_index_received,
                                                                                                     long long int const time_step_direction_received,
                                                                                                     long long int const time_step_prediction_start_received,
                                                                                                     long long int const time_step_prediction_end_received,
                                                                                                     size_t const batch_size_received,
                                                                                                     size_t const block_unit_size_received,
                                                                                                     size_t const cell_unit_size_received,
                                                                                                     T_ const *const ptr_array_summation_cell_states_received,
                                                                                                     struct Layer *const ptr_layer_it_received);
        void Backward_Pass__LSTM_Derivative__Gates_CIF_AF_State__Loop(long long int const time_step_index_received,
                                                                                                               long long int const time_step_direction_received,
                                                                                                               long long int const time_step_reverse_direction_received,
                                                                                                               long long int const time_step_prediction_start_received,
                                                                                                               long long int const time_step_prediction_end_received,
                                                                                                               size_t const batch_size_received,
                                                                                                               size_t const block_unit_size_received,
                                                                                                               size_t const cell_unit_size_received,
                                                                                                               struct Layer *const ptr_layer_it_received);
        void RNN__Backward_Pass_Batch__OpenMP(size_t const batch_size_received);
        void RNN__Backward_Pass_Batch__Pre_Training__OpenMP(size_t const batch_size_received);
        void Recurrent__Backward_Pass__Average_Pooling__OpenMP(size_t const batch_size_received,
                                                                                                    size_t const derivative_input_size_received,
                                                                                                    T_ *const ptr_array_derivative_inputs_received,
                                                                                                    struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__FC__OpenMP(size_t const batch_size_received,
                                                                                size_t const derivative_input_size_received,
                                                                                T_ *const ptr_array_derivative_inputs_received,
                                                                                struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__LSTM__OpenMP(size_t const batch_size_received,
                                                                                     size_t const derivative_input_size_received,
                                                                                     T_ *const ptr_array_derivative_inputs_received,
                                                                                     struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Max_Pooling__OpenMP(size_t const batch_size_received,
                                                                                               size_t const derivative_input_size_received,
                                                                                               T_ *const ptr_array_derivative_inputs_received,
                                                                                               struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Residual__OpenMP(size_t const batch_size_received,
                                                                                         size_t const derivative_input_size_received,
                                                                                         T_ *const ptr_array_derivative_inputs_received,
                                                                                         struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Residual__Block__OpenMP(size_t const batch_size_received,
                                                                                                     size_t const derivative_input_size_received,
                                                                                                     T_ *const ptr_array_derivative_inputs_received,
                                                                                                     struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Gradient__FC__OpenMP(size_t const batch_size_received, struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Gradient__LSTM__OpenMP(bool const forward_layer_received,
                                                                                                    size_t const batch_size_received,
                                                                                                    size_t const derivative_input_size_received,
                                                                                                    T_ *const ptr_array_derivative_inputs_received,
                                                                                                    struct Layer *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Gradient__Residual__OpenMP(size_t const batch_size_received, struct Layer const *const ptr_layer_it_received);
        void Recurrent__Backward_Pass__Gradient__Residual__Layer__OpenMP(bool const is_block_input_layer_received,
                                                                                                                    size_t const batch_size_received,
                                                                                                                    struct Layer *&ptr_layer_it_received);
        void Recurrent__Backward_Pass__Gradient__Residual__FC__OpenMP(bool const is_block_input_layer_received,
                                                                                                                size_t const batch_size_received,
                                                                                                                struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__LSTM__OpenMP(size_t const time_step_index_received,
                                                                   size_t const batch_size_received,
                                                                   size_t const derivative_input_size_received,
                                                                   T_ const *const ptr_array_delta_input_block_inputs_received,
                                                                   T_ const *const ptr_array_delta_input_input_gates_received,
                                                                   T_ const *const ptr_array_delta_input_forget_gates_received,
                                                                   T_ const *const ptr_array_delta_input_output_gates_received,
                                                                   T_ *const ptr_array_derivative_inputs_received,
                                                                   struct Layer const *const ptr_layer_it_received);
        void Backward_Pass__LSTM_Derivative__Output__OpenMP(long long int const time_step_index_received,
                                                                                                long long int const time_step_direction_received,
                                                                                                long long int const time_step_prediction_end_received,
                                                                                                size_t const batch_size_received,
                                                                                                size_t const block_unit_size_received,
                                                                                                size_t const cell_unit_size_received,
                                                                                                T_ const *const ptr_array_delta_recurrent_block_inputs_received,
                                                                                                T_ const *const ptr_array_delta_recurrent_input_gates_received,
                                                                                                T_ const *const ptr_array_delta_recurrent_forget_gates_received,
                                                                                                T_ const *const ptr_array_delta_recurrent_output_gates_received,
                                                                                                struct Layer *const ptr_layer_it_received);
        void Backward_Pass__LSTM_Derivative__Cell_State_AF__OpenMP(long long int const time_step_index_received,
                                                                                                           long long int const time_step_direction_received,
                                                                                                           long long int const time_step_prediction_start_received,
                                                                                                           long long int const time_step_prediction_end_received,
                                                                                                           size_t const batch_size_received,
                                                                                                           size_t const block_unit_size_received,
                                                                                                           size_t const cell_unit_size_received,
                                                                                                           T_ const *const ptr_array_summation_cell_states_received,
                                                                                                           struct Layer *const ptr_layer_it_received);
        void Backward_Pass__LSTM_Derivative__Gates_CIF_AF_State__OpenMP(long long int const time_step_index_received,
                                                                                                                     long long int const time_step_direction_received,
                                                                                                                     long long int const time_step_reverse_direction_received,
                                                                                                                     long long int const time_step_prediction_start_received,
                                                                                                                     long long int const time_step_prediction_end_received,
                                                                                                                     size_t const batch_size_received,
                                                                                                                     size_t const block_unit_size_received,
                                                                                                                     size_t const cell_unit_size_received,
                                                                                                                     struct Layer *const ptr_layer_it_received);
        void Update_Derivative_Weight(size_t const batch_size_received,
                                                     struct Layer *const ptr_layer_it_received,
                                                     struct Layer const *const ptr_layer_end_received);
        void Update_Derivative_Weight__Pre_Training(size_t const batch_size_received);
        void FF__Update_Derivative_Weight_Batch__Loop(size_t const batch_size_received,
                                                                                 struct Layer *ptr_layer_it_received,
                                                                                 struct Layer const *const ptr_layer_end_received);
        void FF__Update_Derivative_Weight_Batch__Pre_Training__Loop(size_t const batch_size_received);
        void Update_Derivative_Weight__FC__Loop(size_t const time_step_index_received,
                                                                       size_t const batch_size_received,
                                                                       size_t const input_unit_size_received,
                                                                       T_ const *const ptr_array_inputs_received,
                                                                       struct Layer *const ptr_layer_it_received);
        void Update_Derivative_Weight__FC__Loop(size_t const time_step_index_received,
                                                                       size_t const batch_size_received,
                                                                       size_t const input_size_received,
                                                                       size_t const derivative_size_received,
                                                                       T_ const *const ptr_array_inputs_received,
                                                                       T_ const *const ptr_array_derivative_inputs_received,
                                                                       T_ *const ptr_array_derivatives_received);
        void Update_Derivative_Weight__Bias__Loop(size_t const time_step_index_received,
                                                                          size_t const batch_size_received,
                                                                          size_t const unit_size_received,
                                                                          T_ const *const ptr_array_derivative_inputs_received,
                                                                          T_ *const ptr_array_derivatives_received);
        void FF__Update_Derivative_Weight_Batch__OpenMP(size_t const batch_size_received,
                                                                                       struct Layer *ptr_layer_it_received,
                                                                                       struct Layer const *const ptr_layer_end_received);
        void FF__Update_Derivative_Weight_Batch__Pre_Training__OpenMP(size_t const batch_size_received);
        void Update_Derivative_Weight__FC__OpenMP(size_t const time_step_index_received,
                                                                             size_t const batch_size_received,
                                                                             size_t const input_unit_size_received,
                                                                             T_ const *const ptr_array_inputs_received,
                                                                             struct Layer *const ptr_layer_it_received);
        void Update_Derivative_Weight__FC__OpenMP(size_t const time_step_index_received,
                                                                             size_t const batch_size_received,
                                                                             size_t const input_size_received,
                                                                             size_t const derivative_size_received,
                                                                             T_ const *const ptr_array_inputs_received,
                                                                             T_ const *const ptr_array_derivative_inputs_received,
                                                                             T_ *const ptr_array_derivatives_received);
        void Update_Derivative_Weight__Bias__OpenMP(size_t const time_step_index_received,
                                                                               size_t const batch_size_received,
                                                                               size_t const unit_size_received,
                                                                               T_ const *const ptr_array_derivative_inputs_received,
                                                                               T_ *const ptr_array_derivatives_received);
        void RNN__Update_Derivative_Weight_Batch__Loop(size_t const batch_size_received,
                                                                                    struct Layer *ptr_layer_it_received,
                                                                                    struct Layer const *const ptr_layer_end_received);
        void RNN__Update_Derivative_Weight_Batch__Pre_Training__Loop(size_t const batch_size_received);
        void Recurrent__Update_Derivative_Weight__FC__Loop(size_t const batch_size_received,
                                                                                         size_t const input_unit_size_received,
                                                                                         T_ const *const ptr_array_inputs_received,
                                                                                         struct Layer *const ptr_layer_it_received);
        void Update_Derivative_Weight__FC_Ind_RNN__Loop(size_t const time_step_index_received,
                                                                                      size_t const batch_size_received,
                                                                                      size_t const derivative_size_received,
                                                                                      T_ const *const ptr_array_inputs_received,
                                                                                      T_ const *const ptr_array_derivative_inputs_received,
                                                                                      T_ *const ptr_array_derivatives_received);
        void Recurrent__Update_Derivative_Weight__LSTM__Loop(bool const forward_layer_received,
                                                                                             size_t const batch_size_received,
                                                                                             size_t const block_unit_size_received,
                                                                                             size_t const cell_unit_size_received,
                                                                                             size_t const input_size_received,
                                                                                             T_ const *const ptr_array_inputs_received,
                                                                                             T_ const *const ptr_array_delta_block_inputs_received,
                                                                                             T_ const *const ptr_array_delta_input_block_inputs_received,
                                                                                             T_ const *const ptr_array_delta_recurrent_block_inputs_received,
                                                                                             T_ const *const ptr_array_delta_input_gates_received,
                                                                                             T_ const *const ptr_array_delta_input_input_gates_received,
                                                                                             T_ const *const ptr_array_delta_recurrent_input_gates_received,
                                                                                             T_ const *const ptr_array_delta_forget_gates_received,
                                                                                             T_ const *const ptr_array_delta_input_forget_gates_received,
                                                                                             T_ const *const ptr_array_delta_recurrent_forget_gates_received,
                                                                                             T_ const *const ptr_array_delta_output_gates_received,
                                                                                             T_ const *const ptr_array_delta_input_output_gates_received,
                                                                                             T_ const *const ptr_array_delta_recurrent_output_gates_received,
                                                                                             struct Layer *const ptr_layer_it_received);
        void Recurrent__Update_Derivative_Weight__LSTM__Bias__Loop(size_t const batch_size_received,
                                                                                                       size_t const layer_block_unit_size_received,
                                                                                                       size_t const layer_cell_unit_size_received,
                                                                                                       T_ const *const ptr_array_delta_block_inputs_received,
                                                                                                       T_ const *const ptr_array_delta_input_gates_received,
                                                                                                       T_ const *const ptr_array_delta_forget_gates_received,
                                                                                                       T_ const *const ptr_array_delta_output_gates_received,
                                                                                                       T_ *const ptr_array_cell_input_derivatives_bias_received,
                                                                                                       T_ *const ptr_array_input_gate_derivatives_bias_received,
                                                                                                       T_ *const ptr_array_forget_gate_derivatives_bias_received,
                                                                                                       T_ *const ptr_array_output_gate_derivatives_bias_received);
        void RNN__Update_Derivative_Weight_Batch__OpenMP(size_t const batch_size_received,
                                                                                          struct Layer *ptr_layer_it_received,
                                                                                          struct Layer const *const ptr_layer_end_received);
        void RNN__Update_Derivative_Weight_Batch__Pre_Training__OpenMP(size_t const batch_size_received);
        void Recurrent__Update_Derivative_Weight__FC__OpenMP(size_t const batch_size_received,
                                                                                               size_t const input_unit_size_received,
                                                                                               T_ const *const ptr_array_inputs_received,
                                                                                               struct Layer *const ptr_layer_it_received);
        void Update_Derivative_Weight__FC_Ind_RNN__OpenMP(size_t const time_step_index_received,
                                                                                            size_t const batch_size_received,
                                                                                            size_t const derivative_size_received,
                                                                                            T_ const *const ptr_array_inputs_received,
                                                                                            T_ const *const ptr_array_derivative_inputs_received,
                                                                                            T_ *const ptr_array_derivatives_received);
        void Recurrent__Update_Derivative_Weight__LSTM__OpenMP(bool const forward_layer_received,
                                                                                                 size_t const batch_size_received,
                                                                                                 size_t const block_unit_size_received,
                                                                                                 size_t const cell_unit_size_received,
                                                                                                 size_t const input_size_received,
                                                                                                 T_ const *const ptr_array_inputs_received,
                                                                                                 T_ const *const ptr_array_delta_block_inputs_received,
                                                                                                 T_ const *const ptr_array_delta_input_block_inputs_received,
                                                                                                 T_ const *const ptr_array_delta_recurrent_block_inputs_received,
                                                                                                 T_ const *const ptr_array_delta_input_gates_received,
                                                                                                 T_ const *const ptr_array_delta_input_input_gates_received,
                                                                                                 T_ const *const ptr_array_delta_recurrent_input_gates_received,
                                                                                                 T_ const *const ptr_array_delta_forget_gates_received,
                                                                                                 T_ const *const ptr_array_delta_input_forget_gates_received,
                                                                                                 T_ const *const ptr_array_delta_recurrent_forget_gates_received,
                                                                                                 T_ const *const ptr_array_delta_output_gates_received,
                                                                                                 T_ const *const ptr_array_delta_input_output_gates_received,
                                                                                                 T_ const *const ptr_array_delta_recurrent_output_gates_received,
                                                                                                 struct Layer *const ptr_layer_it_received);
        void Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(size_t const batch_size_received,
                                                                                                             size_t const layer_block_unit_size_received,
                                                                                                             size_t const layer_cell_unit_size_received,
                                                                                                             T_ const *const ptr_array_delta_block_inputs_received,
                                                                                                             T_ const *const ptr_array_delta_input_gates_received,
                                                                                                             T_ const *const ptr_array_delta_forget_gates_received,
                                                                                                             T_ const *const ptr_array_delta_output_gates_received,
                                                                                                             T_ *const ptr_array_cell_input_derivatives_bias_received,
                                                                                                             T_ *const ptr_array_input_gate_derivatives_bias_received,
                                                                                                             T_ *const ptr_array_forget_gate_derivatives_bias_received,
                                                                                                             T_ *const ptr_array_output_gate_derivatives_bias_received);
        void Update_Error(struct AF_unit *const ptr_AF_received,
                                   T_ const observed_output_received,
                                   T_ const desired_output_received,
                                   T_ const error_received,
                                   size_t const thread_index_received = 0u);
        void Copy__Neuron_Unit(struct Neuron_unit const *const ptr_source_neuron_unit_received, struct Neuron_unit *const ptr_destination_neuron_unit_received);
        void Copy__Neuron_Units(size_t const start_index_received,
                                              size_t const end_index_received,
                                              struct Neuron_unit const *ptr_array_source_neuron_units_received);
        void Copy__AF_Units(size_t const start_index_received,
                                        size_t const end_index_received,
                                        struct AF_unit const *ptr_array_source_AF_units_received);
        void Copy__AF_Unit(struct AF_unit const *const ptr_source_AF_unit_received, struct AF_unit *const ptr_destination_AF_unit_received);
        void Copy__AF_Ind_Recurrent_Units(size_t const start_index_received,
                                                              size_t const end_index_received,
                                                              struct AF_Ind_recurrent_unit const *ptr_array_source_AF_Ind_recurrent_units_received,
                                                              bool const copy_connections_received = true);
        void Copy__AF_Ind_Recurrent_Unit(struct AF_Ind_recurrent_unit const *const ptr_source_AF_Ind_recurrent_unit_received,
                                                            struct AF_Ind_recurrent_unit *const ptr_destination_AF_Ind_recurrent_unit_received,
                                                            bool const copy_connections_received = true);
        void Copy__Normalized_Batch_Unit(size_t const number_units_received,
                                                             struct Normalized_batch_unit const &ref_source_normalized_batch_unit_received,
                                                             struct Normalized_batch_unit &ref_destination_normalized_batch_unit_received);
        void Copy__Block(struct Block_unit const *const ptr_source_block_unit_received, struct Block_unit *const ptr_destination_block_unit_received);
        void Copy__Block__AF(struct Block_unit const *const ptr_source_block_unit_received, struct Block_unit *const ptr_destination_block_unit_received);
        void Copy__Blocks(size_t const start_index_received,
                                     size_t const end_index_received,
                                     struct Block_unit const *ptr_array_source_block_units_received,
                                     bool const copy_connections_received = true);
        void Copy__Blocks__AF(size_t const start_index_received,
                                            size_t const end_index_received,
                                            struct Block_unit const *ptr_array_source_block_units_received);
        template<typename U> void Copy__Layer__FC(struct Layer const *const ptr_source_layer_received,
                                                                             struct Layer *const ptr_destination_layer_received,
                                                                             U *const ptr_source_first_U_received,
                                                                             U *const ptr_destination_first_U_received,
                                                                             U *const *ptr_source_array_ptr_connections_received,
                                                                             U **ptr_destination_array_ptr_connections_received);
        void Copy__Layer__AF_Ind_Recurrent(struct Layer const *const ptr_source_layer_received,
                                                         struct AF_Ind_recurrent_unit *const ptr_source_first_AF_Ind_recurrent_unit_received,
                                                         struct AF_Ind_recurrent_unit *const ptr_destination_first_AF_Ind_recurrent_unit_received,
                                                         struct AF_Ind_recurrent_unit *const *ptr_source_array_ptr_connections_received,
                                                         struct AF_Ind_recurrent_unit **ptr_destination_array_ptr_connections_received);
        template<typename U> void Copy__Layer__LSTM(struct Layer const *const ptr_source_layer_received,
                                                                                 struct Layer *const ptr_destination_layer_received,
                                                                                 struct Cell_unit *const ptr_source_first_cell_unit_received,
                                                                                 U *const ptr_source_first_U_received,
                                                                                 U *const ptr_destination_first_U_received,
                                                                                 void *const *ptr_source_array_ptr_connections_received,
                                                                                 void **ptr_destination_array_ptr_connections_received);
        void Indexing_Regularization_Parameters(void);
        void Indexing_Regularization_Parameters__Pre_training(void);
        void Indexing_Regularization__Weights__FC__Forward(T_ const mask_received, struct Layer const *const ptr_layer_it_received);
        void Indexing_Regularization__Weights__AF_Ind_Recurrent(T_ const mask_received, struct Layer const *const ptr_layer_it_received);
        void Indexing_Regularization__Weights__LSTM(T_ const mask_received, struct Layer const *const ptr_layer_it_received);
        void Indexing_Regularization__Bias(T_ const mask_received, struct Layer const *const ptr_layer_it_received);
        void Reset__Parameter__Mask_Dropout(bool *ptr_array_units_mask_dropout_bernoulli_received);
        void Reset__Parameters__Cell_Unit__Mask_Dropout(bool *ptr_array_cell_units_mask_dropout_received);
        void Clear(void);
        void Deallocate(void);
        void Deallocate__Sparse_K_Filter(void);
        void Deallocate__Parameter__Optimizer(void);
        void Deallocate__Parameter__Gradient_Descent(void);
        void Deallocate__Parameter__iRPROP_minus(void);
        void Deallocate__Parameter__iRPROP_plus(void);
        void Deallocate__Parameter__Adam(void);
        void Deallocate__Parameter__AMSGrad(void);
        void Deallocate__Parameter__Regularization(void);
        void Deallocate__Generator__Dropout_Bernoulli(void);
        void Deallocate__Generator__Dropout_Gaussian(void);
        void Deallocate__Generator__Dropout_ShakeDrop(void);
        void Deallocate__Generator__Dropout_Uout(void);
        void Deallocate__Generator__Dropout_Zoneout(void);
        void Deallocate__Neuron__Mask_Dropout_Bernoulli(void);
        void Deallocate__Layer__Mask_Dropout_ShakeDrop(void);
        void Deallocate__Cell_Unit__Mask_Dropout_Zoneout(void);
        void Deallocate__Parameter__Batch_Normalization(void);
        void Deallocate__Normalized_Unit(void);
        void Deallocate__Normalized_Unit__Batch_Normalization(void);
        void Deallocate__Normalized_Unit__Batch_Renormalization(void);
        void Clear_Optimizer(void);
        void Order__Layers__Connection(void);
        void Order__Layers__Output(void);
        void Order__Layer__Output(bool const is_sequentiel_received, struct Layer *const ptr_layer_received);
        void Order__Layer__Output__Pre_Training(bool const is_sequentiel_received, struct Layer *const ptr_layer_received);
        void Order__Layer__Basic(struct Layer *const ptr_layer_it_received);
        void Order__Layer__Basic_unit(struct Layer *const ptr_layer_it_received);
        void Order__Layer__Basic_indice(struct Layer *const ptr_layer_it_received);
        void Order__Layer__Basic_indice_unit(struct Layer *const ptr_layer_it_received);
        void Order__Layer__Neuron(struct Layer *const ptr_layer_it_received);
        void Order__Layer__Neuron_Unit(struct Layer *const ptr_layer_it_received);
        void Order__Layer__AF(struct Layer *const ptr_layer_it_received);
        void Order__Layer__AF_Unit(struct Layer *const ptr_layer_it_received);
        void Order__Layer__AF_Unit__Dropout_Bernoulli(struct Layer *const ptr_layer_it_received);
        void Order__Layer__AF_Ind_Recurrent(struct Layer *const ptr_layer_it_received);
        void Order__Layer__AF_Ind_Recurrent_Unit(struct Layer *const ptr_layer_it_received);
        void Order__Layer__AF_Ind_Recurrent_Unit__Dropout_Bernoulli(struct Layer *const ptr_layer_it_received);
        void Order__Layer__LSTM(struct Layer *const ptr_layer_it_received);
        void Order__Layer__Block_Unit(struct Layer *const ptr_layer_it_received);
        void Order__Layer__Block_Unit__Dropout_Zoneout(struct Layer *const ptr_layer_it_received);
        void Order__Layer__Normalization(struct Layer *const ptr_layer_it_received);
        void Order__Layer__Batch_Normalization(struct Layer *const ptr_layer_it_received);
        void Order__Layer__Batch_Renormalization(struct Layer *const ptr_layer_it_received);
        void Reset__Parameter__Normalized_Unit(void);
        void Reset__Derivative_Parameter__Normalized_Unit(void);
        void Clear_Training_Arrays(void);
        void Clear__Parameter__Normalized_Unit(void);
        bool Transfer_Learning(class Neural_Network *&ptr_destination_Neural_Network_received) const;
        bool User_Controls(void);
        bool User_Controls__Optimizer__Gradient_Descent(void);
        bool User_Controls__Optimizer__iRPROP(void);
        bool User_Controls__Optimizer__AdaBound(void);
        bool User_Controls__Optimizer__Adam(void);
        bool User_Controls__Optimizer__NosAdam(void);
        bool User_Controls__Regularization(void);
        bool User_Controls__Dropout(void);
        bool User_Controls__Normalization(void);
        bool User_Controls__Normalization_Layer(void);
        bool User_Controls__Tied__Parameter(void);
        bool User_Controls__K_Sparse(void);
        bool User_Controls__Weights_Initializer(void);
        bool User_Controls__Optimizer_Function_Initializer(void);
        bool User_Controls__Loss_Function_Initializer(void);
        bool User_Controls__Accuracy_Function_Initializer(void);
        bool User_Controls__Optimizer_Function(void);
        bool User_Controls__Warm_Restarts(void);
        bool User_Controls__Accuracy_Variance(void);
        bool User_Controls__Time_Delays(void);
        bool User_Controls__Clip_Gradient(void);
        bool User_Controls__Max_Norm_Constaints(void);
        bool User_Controls__L1_Regularization(void);
        bool User_Controls__L2_Regularization(void);
        bool User_Controls__SRIP_Regularization(void);
        bool User_Controls__Maximum__Batch_Size(void);
        bool User_Controls__OpenMP(void);
        bool Copy__Optimizer_Parameters(class Neural_Network const *const ptr_Neural_Network_received, bool const copy_delta_optimizer_received = false);
        bool Copy__Delta__Gradient_Descent(class Neural_Network const *const ptr_Neural_Network_received);
        bool Copy__Delta__iRPROP_minus(class Neural_Network const *const ptr_Neural_Network_received);
        bool Copy__Delta__iRPROP_plus(class Neural_Network const *const ptr_Neural_Network_received);
        bool Copy__Delta__Adam(class Neural_Network const *const ptr_Neural_Network_received);
        bool Copy__Delta__AMSGrad(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__Warm_Restarts_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__Gradient_Descent_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__QuickProp_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__RPROP_minus_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__RPROP_plus_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__SARProp_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__Adam_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__NosAdam_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__AdaBound_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__Training_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__Initializer__Weight_Parameter(class Neural_Network const &ref_source_Neural_Network_received);
        void Copy__Regularization(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__Tied_Weight(struct Layer const *ptr_array_source_layers_received,
                                             struct Layer const *const ptr_last_source_layer_received,
                                             struct Layer *ptr_array_destination_layers_received);
        void Copy__Sparse_K_Filters(struct Layer const *ptr_array_source_layers_received,
                                                    struct Layer const *const ptr_last_source_layer_received,
                                                    struct Layer *ptr_array_destination_layers_received);
        void Copy__Constraint_Recurrent_Weight(struct Layer const *ptr_array_source_layers_received,
                                                                     struct Layer const *const ptr_last_source_layer_received,
                                                                     struct Layer *ptr_array_destination_layers_received);
        void Copy__Loss(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__Accuracy(class Neural_Network const *const ptr_Neural_Network_received);
        void Copy__Dropout(struct Layer const *ptr_array_source_layers_received,
                                      struct Layer const *const ptr_last_source_layer_received,
                                      struct Layer *ptr_array_destination_layers_received);
        void Copy__Normalization(struct Layer const *ptr_array_source_layers_received,
                                              struct Layer const *const ptr_last_source_layer_received,
                                              struct Layer *ptr_array_destination_layers_received);
        void Copy__Normalization(class Neural_Network const *const ptr_source_Neural_Network_received);
        void Copy__Normalized_Units(size_t const start_index_received,
                                                    size_t const end_index_received,
                                                    union Normalized_unit const *ptr_array_source_normalized_units_received);
        void Merge_Derivatives_Parameters(size_t const start_index_received, size_t const end_index_received);
        template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> void Layer__Forward__Neuron_Information__Connection(std::string &ref_output_received,
                                                                                                                                                                                                                struct Neuron_unit const *const ptr_neuron_it_received,
                                                                                                                                                                                                                U const *const ptr_first_U_unit_received);
        template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> void Layer__LSTM_Information__Connection(std::string &ref_output_received,
                                                                                                                                                                                               struct Block_unit const *const ptr_block_unit_it_received,
                                                                                                                                                                                               U const *const ptr_first_U_unit_received);
        
        bool Allocate__Structure(size_t const number_layers_received, size_t const maximum_allowable_memory_received = 32_zu * KILOBYTE * KILOBYTE);
        bool Copy(class Neural_Network const &ref_source_Neural_Network_received,
                       bool const initialize_parallel_computation_received = true,
                       bool const copy_delta_optimizer_received = false,
                       size_t const maximum_allowable_memory_received = 0_zu);
        bool Update(class Neural_Network const &ref_source_Neural_Network_received,
                          bool const initialize_parallel_computation_received = false,
                          bool const update_delta_optimizer_received = false);
        bool Assign__Layers(struct Layer_Parameters const *const ptr_array_layers_received);
        bool Assign__Layer(struct Layer *&ptr_layer_it_received, struct Layer_Parameters const *const ptr_array_layers_received);
        bool Assign__Residual_Block(struct Layer *&ptr_layer_it_received, struct Layer_Parameters const *const ptr_array_layers_received);
        bool Assign__Residual_Layer(struct Layer *&ptr_layer_it_received, struct Layer_Parameters const *const ptr_array_layers_received);
        bool Assign__Post__Layers(void);
        bool Assign__Post__Layer(struct Layer *&ptr_layer_it_received);
        bool Assign__Post__Residual_Block(struct Layer *&ptr_layer_it_received);
        bool Assign__Post__Residual_Layer(bool const is_block_input_layer_received, struct Layer *&ptr_layer_it_received);
        bool Compile(size_t const number_layers_received,
                            size_t const number_recurrent_depth_received,
                            MyEA::Common::ENUM_TYPE_NETWORKS const type_network_received,
                            struct Layer_Parameters const *const ptr_array_layers_received,
                            size_t const maximum_allowable_memory_received = 32_zu * KILOBYTE * KILOBYTE);
        bool Allouable__Batch_Size(size_t const desired_batch_size_received,
                                                size_t &ref_batch_size_allouable_received,
                                                size_t &ref_number_threads_allouable_received);
        bool Information__Output_Layer(std::string &ref_output_received,
                                                      struct Layer const *const ptr_layer_it_received,
                                                      struct Layer const *const ptr_previous_layer_received);
        bool Information__Layer__AF(std::string &ref_output_received, struct Layer const *const ptr_layer_it_received);
        bool Information__Layer__AF_Ind_Recurrent(std::string &ref_output_received, struct Layer const *const ptr_layer_it_received);
        bool Information__Layer__Bias(std::string &ref_output_received, struct Layer const *const ptr_layer_it_received);
        bool Information__Layer__Normalization(std::string &ref_output_received, struct Layer const *const ptr_layer_it_received);
        bool Information__Normalized_Unit(size_t const number_units_received,
                                                           enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const type_normalization_received,
                                                           union Normalized_unit const *const ptr_normalized_unit_received,
                                                           std::string &ref_output_received);
        bool Information__Layer__FC(std::string &ref_output_received,
                                                   struct Layer const *const ptr_layer_it_received,
                                                   struct Layer const *const ptr_previous_layer_received);
        bool Information__Layer__LSTM(std::string &ref_output_received,
                                                       struct Layer const *const ptr_layer_it_received,
                                                       struct Layer const *const ptr_previous_layer_received);
        bool Deinitialize__OpenMP(void);
        bool Multi_Class_Classification(void) const;
        bool Update__Thread_Size(size_t const desired_number_threads_received);
        bool Update__Batch_Size(size_t const desired_batch_size_received, bool const force_update_received = false);
        bool Reallocate__Thread(size_t const number_threads_received);
        bool Reallocate__Thread__Sparse_K_Filter(size_t const number_threads_received);
        bool Reallocate__Thread__Cost(size_t const number_threads_received);
        bool Reallocate__Thread__Normalized_Unit__Batch_Normalization(size_t const number_threads_received);
        bool Reallocate__Thread__Parameter(size_t const number_threads_received);
        bool Reallocate__Thread__Generator__Dropout__Bernoulli(size_t const number_threads_received);
        bool Reallocate__Thread__Generator__Dropout__Gaussian(size_t const number_threads_received);
        bool Reallocate__Thread__Generator__Dropout__ShakeDrop(size_t const number_threads_received);
        bool Reallocate__Thread__Generator__Dropout__Uout(size_t const number_threads_received);
        bool Reallocate__Thread__Generator__Dropout__Zoneout(size_t const number_threads_received);
        bool Reallocate__Batch(size_t const batch_size_received);
        bool Reallocate__Batch__Basic_Unit(size_t const batch_size_received);
        bool Reallocate__Batch__Basic_Indice_Unit(size_t const batch_size_received);
        bool Reallocate__Batch__Neuron_Unit(size_t const batch_size_received);
        bool Reallocate__Batch__AF_Unit(size_t const batch_size_received);
        bool Reallocate__Batch__AF_Ind_Recurrent_Unit(size_t const batch_size_received);
        bool Reallocate__Batch__LSTM(size_t const batch_size_received);
        bool Reallocate__Batch__Dropout__ShakeDrop(size_t const batch_size_received);
        bool Reallocate__Normalized_Unit__Batch_Normalization(size_t const batch_size_received);
        bool Reallocate__Parameter(size_t const number_parameters_received);
        bool Reallocate__Parameter__Regularization(size_t const number_parameters_received);
        bool Reallocate__Parameter__Optimizer(size_t const number_parameters_received);
        bool Reallocate__Parameter__Gradient_Descent(size_t const number_parameters_received);
        bool Reallocate__Parameter__iRPROP_minus(size_t const number_parameters_received);
        bool Reallocate__Parameter__iRPROP_plus(size_t const number_parameters_received);
        bool Reallocate__Parameter__Adam(size_t const number_parameters_received);
        bool Reallocate__Parameter__AMSGrad(size_t const number_parameters_received);
        bool Load(std::string const &ref_path_dimension_received,
                       std::string const &ref_path_parameters_received,
                       size_t const maximum_allowable_memory_received);
        bool Load_Dimension__Neuron(struct Neuron_unit *const ptr_neuron_received, std::ifstream &ref_ifstream_received);
        bool Load_Dimension__AF(struct AF_unit *const ptr_AF_received, std::ifstream &ref_ifstream_received);
        bool Load_Dimension__AF_Ind_Recurrent(struct AF_Ind_recurrent_unit *const ptr_AF_Ind_received, std::ifstream &ref_ifstream_received);
        bool Load_Dimension__Normalized_Unit(size_t const number_units_received,
                                                                   enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const type_normalization_received,
                                                                   union Normalized_unit *const ptr_normalized_unit_received,
                                                                   std::ifstream &ref_ifstream_received);
        bool Load_Dimension__Block(size_t const layer_number_block_units_received,
                                                    size_t const layer_number_cell_units_received,
                                                    enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const type_normalization_received,
                                                    struct Block_unit *const ptr_block_received,
                                                    std::ifstream &ref_ifstream_received);
        bool Load_Dimension__Cell_Units(struct Layer *const ptr_layer_it_received,
                                                          struct Cell_unit *&ptr_reference_array_cells_received,
                                                          std::ifstream &ref_ifstream_received);
        template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> bool Load_Dimension__Connection(size_t index_received,
                                                                                                                                                                                T_ *const ptr_array_parameters_received,
                                                                                                                                                                                U *const ptr_first_U_unit_received,
                                                                                                                                                                                U **ptr_array_ptr_U_unit_connection_received,
                                                                                                                                                                                std::ifstream &ref_ifstream_received);
        template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> bool Load_Dimension__Neuron__Forward__Connection(struct Neuron_unit *const ptr_neuron_received,
                                                                                                                                                                                                             U *const ptr_first_U_unit_received,
                                                                                                                                                                                                             std::ifstream &ref_ifstream_received);
        template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> bool Load_Dimension__Block__Connection(struct Block_unit *const ptr_block_unit_it_received,
                                                                                                                                                                                            U *const ptr_first_U_unit_received,
                                                                                                                                                                                            std::ifstream &ref_ifstream_received);
        template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> bool Load_Dimension__FC(struct Layer *const ptr_layer_it_received,
                                                                                                                                                                    U *const ptr_first_U_unit_received,
                                                                                                                                                                    std::ifstream &ref_ifstream_received);
        bool Load_Dimension__AF(struct Layer *const ptr_layer_it_received, std::ifstream &ref_ifstream_received);
        bool Load_Dimension__AF_Ind_Recurrent(struct Layer *const ptr_layer_it_received, std::ifstream &ref_ifstream_received);
        template<typename U, enum MyEA::Common::ENUM_TYPE_LAYER const E> bool Load_Dimension__LSTM(struct Layer *const ptr_layer_it_received,
                                                                                                                                                                        U *const ptr_first_U_unit_received,
                                                                                                                                                                        std::ifstream &ref_ifstream_received);
        bool Load_Dimension__Normalization(struct Layer *const ptr_layer_it_received, std::ifstream &ref_ifstream_received);
        bool Load_Dimension__Bias(struct Layer *const ptr_layer_it_received, std::ifstream &ref_ifstream_received);
        bool Load_Parameters(std::string const &ref_path_received);
        bool Save_General_Parameters(std::string const &ref_path_received);
        bool Save_Dimension_Parameters(std::string const &ref_path_received);
        bool Allocate__Sparse_K_Filter(void);
        bool Allocate__Parameter(void);
        bool Allocate__Parameter__Optimizer(void);
        bool Allocate__Parameter__Gradient_Descent(void);
        bool Allocate__Parameter__iRPROP_minus(void);
        bool Allocate__Parameter__iRPROP_plus(void);
        bool Allocate__Parameter__Adam(void);
        bool Allocate__Parameter__AMSGrad(void);
        bool Allocate__Parameter__Regularization(void);
        /* arguments:
            reallocate_received: Use in the load from a file function. */
        bool Allocate__Parameter__Normalization(void);
        bool Allocate__Basic_Units(void);
        bool Allocate__Basic_Indice_Units(void);
        bool Allocate__Neuron_Units(void);
        bool Allocate__Neuron__Mask_Dropout_Bernoulli(void);
        bool Allocate__Layer__Mask__Dropout__ShakeDrop(void);
        bool Allocate__AF_Units(void);
        bool Allocate__AF_Ind_Recurrent_Units(void);
        bool Allocate__Normalized_Unit(bool const organize_pointers_received);
        bool Allocate__Normalized_Unit__Batch_Normalization(void);
        bool Allocate__Normalized_Unit__Batch_Renormalization(void);
        bool Allocate__Block_Unit__Mask_Dropout_Zoneout(void);
        bool Allocate__LSTM_Layers(void);
        bool Allocate__Bidirectional__Layers(void);
        bool Allocate__Generator__Dropout_Bernoulli(void);
        bool Allocate__Generator__Dropout_Gaussian(void);
        bool Allocate__Generator__Dropout_ShakeDrop(void);
        bool Allocate__Generator__Dropout_Uout(void);
        bool Allocate__Generator__Dropout_Zoneout(void);
        bool Initialized__Weight(void) const;
        bool Initialize__Weight(class Dataset<T_> const *const ptr_Dataset_received);
        bool Set__Pre_Training_Level(size_t const pre_training_level_received);
        bool Set__Maximum__Batch_Size(size_t const maximum_batch_size_received);
        bool Set__OpenMP(bool const use_openmp_received);
        bool Set__Maximum_Thread_Usage(double const percentage_maximum_thread_usage_received);
        bool Set__Accurancy_Variance(T_ const accurancy_variance_received);
        bool Set__Number_Time_Delays(size_t const time_delays_received);
        bool Set__Dropout(size_t const index_layer_received,
                                    enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT const type_layer_dropout_received,
                                    T_ const value_dropout_received[],
                                    bool const scale_weights_received = true);
        bool Set__Dropout(struct Layer *const ptr_layer_received,
                                    enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT const type_layer_dropout_received,
                                    T_ const value_dropout_received[],
                                    bool const scale_weights_received = true);
        bool Set__Dropout_None(struct Layer *const ptr_layer_received);
        bool Set__Dropout_Alpha(struct Layer *const ptr_layer_received, T_ const dropout_probability_received);
        bool Set__Dropout_Bernoulli(struct Layer *const ptr_layer_received,
                                                  T_ const retention_probability_received,
                                                  bool const scale_weights_received = true);
        bool Set__Dropout_Bernoulli_Inverted(struct Layer *const ptr_layer_received, T_ const retention_probability_received);
        bool Set__Dropout_Gaussian(struct Layer *const ptr_layer_received, T_ const dropout_probability_received);
        bool Set__Dropout_ShakeDrop(struct Layer *const ptr_layer_received, T_ const dropout_probability_received);
        bool Set__Dropout_Uout(struct Layer *const ptr_layer_received, T_ const dropout_probability_received);
        bool Set__Dropout_Zoneout(struct Layer *const ptr_layer_received,
                                                 T_ const zoneout_cell_received,
                                                 T_ const zoneout_hidden_received);
        void Scale_Weight__Dropout(T_ const scale_factor_received, struct Layer const *const ptr_layer_it_received);
        void Scale_Weight__FC__Forward__Dropout(T_ const scale_factor_received, struct Layer const *const ptr_layer_it_received);
        void Scale_Weight__FC__Recurrent__Dropout(T_ const scale_factor_received, struct Layer const *const ptr_layer_it_received);
        bool Prepare__Normalized__Layers(void);
        bool Prepare__Normalized__Layer(struct Layer *&ptr_layer_it_received);
        bool Prepare__Normalized__Residual_Block(struct Layer *&ptr_layer_it_received);
        bool Prepare__Normalized__Residual_Layer(struct Layer *&ptr_layer_it_received);
        /* arguments:
            reallocate: When loading from a file this value should be set to false.
            organize_pointers: When loading from a file this value should be set to false. */
        bool Set__Layer_Normalization(size_t const index_layer_received,
                                                      enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const type_layer_normalization_received,
                                                      bool const reallocate_dimension_parameters_received = true,
                                                      bool const organize_pointers_received = true);
        bool Set__Layer_Normalization(struct Layer *const ptr_layer_received,
                                                      enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const type_layer_normalization_received,
                                                      bool const reallocate_dimension_parameters_received = true,
                                                      bool const organize_pointers_received = true);
        bool Set__Normalization_None(struct Layer *const ptr_layer_received, bool const organize_pointers_received);
        bool Set__Batch_Normalization(struct Layer *const ptr_layer_received,
                                                      bool const use_batch_normalization_received = true,
                                                      bool const reallocate_dimension_parameters_received = true,
                                                      bool const organize_pointers_received = true);
        bool Set__Batch_Renormalization(struct Layer *const ptr_layer_received,
                                                          bool const use_batch_renormalization_received = true,
                                                          bool const reallocate_dimension_parameters_received = true,
                                                          bool const organize_pointers_received = true);
        bool Set__Ghost_Batch_Normalization(struct Layer *const ptr_layer_received,
                                                                 bool const use_ghost_batch_normalization_received = true,
                                                                 bool const reallocate_dimension_parameters_received = true,
                                                                 bool const organize_pointers_received = true);
        bool Set__Clip_Gradient(T_ const clip_gradient_received);
        bool Check__Use__Regularization__Constraint_Recurrent_Weight__Default(size_t const index_layer_received) const;
        bool Check__Use__Regularization__Constraint_Recurrent_Weight__Default(struct Layer *const ptr_layer_received) const;
        bool Set__Regularization__Constraint_Recurrent_Weight__Default(size_t const index_layer_received);
        bool Set__Regularization__Constraint_Recurrent_Weight__Default(struct Layer *const ptr_layer_received);
        bool Set__Regularization__Constraint_Recurrent_Weight(size_t const index_layer_received,
                                                                                           T_ const constraint_recurrent_weight_lower_bound_received,
                                                                                           T_ const constraint_recurrent_weight_upper_bound_received);
        bool Set__Regularization__Constraint_Recurrent_Weight(struct Layer *const ptr_layer_received,
                                                                                           T_ const constraint_recurrent_weight_lower_bound_received,
                                                                                           T_ const constraint_recurrent_weight_upper_bound_received);
        bool Set__Tied_Parameter(size_t const index_layer_received,
                                           bool const use_tied_parameter_received,
                                           bool const transpose_received = true);
        bool Set__Tied_Parameter(struct Layer *const ptr_layer_received,
                                           bool const use_tied_parameter_received,
                                           bool const transpose_received = true);
        // TODO: Backpropagate toward the K largest activation function (basic indice unit).
        bool Set__K_Sparsity(size_t const index_layer_received, size_t const k_sparsity_received);
        bool Set__K_Sparsity(struct Layer *const ptr_layer_received, size_t const k_sparsity_received);
        bool Set__Alpha_Sparsity(size_t const index_layer_received, T_ const alpha_sparsity_received);
        bool Set__Alpha_Sparsity(struct Layer *const ptr_layer_received,T_ const alpha_sparsity_received);
        bool Set__Regularization__L1(T_ const regularization__l1_received);
        bool Set__Regularization__L2(T_ const regularization__l2_received);
        bool Set__Regularization__SRIP(T_ const regularization__l2_received);
        bool Set__Regularization__Weight_Decay(T_ const regularization__weight_decay_received);
        bool Set__Regularization__Max_Norm_Constraints(T_ const regularization__max_norm_constraints_received);
        bool Set__Normalization_Momentum_Average(T_ const momentum_average_received);
        bool Set__Normalization_Epsilon(T_ const epsilon_received);
        bool Set__Batch_Renormalization_r_Correction_Maximum(T_ const r_correction_maximum_received);
        bool Set__Batch_Renormalization_d_Correction_Maximum(T_ const d_correction_maximum_received);
        bool Set__Layer_Activation_Function(size_t const index_layer_received, enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received);
        bool Set__Layer_Activation_Function(struct Layer *const ptr_layer_it_received, enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received);
        bool Set__Layer_Activation_Steepness(size_t const index_layer_received, T_ const activation_steepness_received);
        bool Set__Layer_Activation_Steepness(struct Layer *const ptr_layer_it_received, T_ const activation_steepness_received);
        bool Set__Multi_Label(bool const use_multi_label_received);
        bool Set__Input_Mode(bool const use_first_layer_as_input_received);
        bool Set__Output_Mode(bool const use_last_layer_as_output_received);
        bool Use__Clip_Gradient(void) const;
        bool Use__Regularization_Parameter(void) const;
        bool Use__Normalization(void) const;
        bool Use__Batch_Normalization(void) const;
        bool Use__Batch_Renormalization(void) const;
        bool Use__Ghost_Batch_Normalization(void) const;
        bool Use__Streaming_Normalization(void) const;
        bool Use__Dropout__Alpha(void) const;
        bool Use__Dropout__Bernoulli(void) const;
        bool Use__Dropout__Bernoulli__Inverted(void) const;
        bool Use__Dropout__Gaussian(void) const;
        bool Use__Dropout__ShakeDrop(void) const;
        bool Use__Dropout__Uout(void) const;
        bool Use__Dropout__Zoneout(void) const;
        bool Use__K_Sparse(void) const;
        bool Use__Tied_Parameter(void) const;
        bool Use__Regularization__Constraint_Recurrent_Weight(void) const;
        bool Use__Multi_Label(void) const;
        bool Usable_Warm_Restarts(void) const;
        bool Compare(bool const use_metric_loss_received,
                             bool const dataset_in_equal_less_dataset_out_accepted_received,
                             enum MyEA::Common::ENUM_TYPE_DATASET const type_holdout_dataset_received,
                             T_ const minimum_loss_holdout_dataset_accepted_received,
                             class Neural_Network const *const ptr_source_Neural_Network_received) const;
        bool *ptr_array_units_mask_dropout_bernoulli = nullptr; // size[H].
        bool *ptr_array_layers_mask_dropout_shakedrop = nullptr; // size[L, T, B].
        bool *ptr_array_cell_units_mask_dropout_zoneout = nullptr;
        bool use_OpenMP = false;
        bool is_OpenMP_initialized = false;
        bool use_Warm_Restarts = false;
        bool use_Nesterov = false;
        bool use_normalized_weight_decay = true;
        bool use_adam_bias_correction = true;
        bool use_multi_label = false;
        bool use_clip_gradient = false;
        /* Use the first layer as input.
            Default:
            - Always true.
            Autoencoder:
            - true: Feed inputs into the input layer. 
            - false: Feed inputs into the decoded layer. */
        bool use_first_layer_as_input = true;
        /* Use the last layer as output.
            Default:
            - Always true.
            Autoencoder:
            - true: Reconstruct the inputs as output(s). 
            - false: Compress the inputs as output(s). */
        bool use_last_layer_as_output = true;
        
        std::pair<size_t, T_> *ptr_array_k_sparse_activities = nullptr;

        size_t Prepare__Connections__FC(size_t const input_size_received, struct Layer *const ptr_layer_it_received);
        size_t Prepare__Connections__FC_Ind_RNN(size_t const input_size_received, struct Layer *const ptr_layer_it_received);
        size_t Prepare__Connections__LSTM(size_t const input_size_received, struct Layer *const ptr_layer_it_received);
        size_t Prepare__Bias__FC(size_t const shift_index_received, struct Layer *const ptr_layer_it_received);
        size_t Prepare__Bias__LSTM(size_t const shift_index_received, struct Layer *const ptr_layer_it_received);
        size_t Get__Total_Layers(void) const;
        size_t *ptr_array_number_loss = nullptr; // size[N].
        size_t *ptr_array_number_bit_fail = nullptr; // size[N].
        size_t number_accuracy_trial = 0_zu;
        size_t number_threads = 1_zu;
        size_t cache_number_threads = 1_zu;
        size_t batch_size = 1_zu;
        size_t cache_batch_size = 1_zu;
        size_t maximum_batch_size = (std::numeric_limits<size_t>::max)();
        size_t total_basic_units = 0_zu;
        size_t total_basic_units_allocated = 0_zu;
        size_t total_basic_indice_units = 0_zu;
        size_t total_basic_indice_units_allocated = 0_zu;
        size_t total_neuron_units = 0_zu;
        size_t total_neuron_units_allocated = 0_zu;
        size_t total_AF_units = 0_zu;
        size_t total_AF_units_allocated = 0_zu;
        size_t total_AF_Ind_recurrent_units = 0_zu;
        size_t total_AF_Ind_recurrent_units_allocated = 0_zu;
        size_t total_block_units = 0_zu;
        size_t total_block_units_allocated = 0_zu;
        size_t total_cell_units = 0_zu;
        size_t total_cell_units_allocated = 0_zu;
        size_t total_normalized_units = 0_zu;
        size_t total_normalized_units_allocated = 0_zu;
        size_t total_parameters = 0_zu;
        size_t total_parameters_allocated = 0_zu;
        size_t total_weights = 0_zu;
        size_t total_weights_allocated = 0_zu;
        size_t total_bias = 0_zu;
        size_t total_bias_allocated = 0_zu;
        size_t number_inputs = 0_zu;
        size_t number_outputs = 0_zu;
        size_t number_recurrent_depth = 1_zu;
        size_t number_time_delays = 0_zu;
        size_t pre_training_level = 0_zu;
        size_t *ptr_array_basic_indice_units_indices = nullptr; // size[B, T, H].
        size_t *ptr_array_number_neurons_by_layer = nullptr; // size[L].
        size_t *ptr_array_number_connections_by_layer = nullptr; // size[L].
        size_t *ptr_array_neuron_units_first_forward_connection_index = nullptr; // size[H].
        size_t *ptr_array_neuron_units_last_forward_connection_index = nullptr; // size[H].
        size_t *ptr_array_neuron_units_number_forward_connections = nullptr; // size[H].
        size_t *ptr_array_AF_Ind_recurrent_units_recurrent_connection_index = nullptr; // size[H].
        size_t *ptr_array_layers_number_outputs = nullptr; // size[L].
        size_t *ptr_array_layers_first_connection_index = nullptr; // size[L].
        size_t *ptr_array_layers_last_connection_index = nullptr; // size[L].
        size_t total_layers = 0_zu;
        size_t total_batch_normalization_layers = 0_zu;
        size_t total_batch_renormalization_layers = 0_zu;
        size_t total_ghost_batch_normalization_layers = 0_zu;
        size_t total_streaming_normalization_layers = 0_zu;
        size_t total_dropout_alpha_layers = 0_zu;
        size_t total_dropout_bernoulli_layers = 0_zu;
        size_t total_dropout_bernoulli_inverted_layers = 0_zu;
        size_t total_dropout_gaussian_layers = 0_zu;
        size_t total_dropout_shakedrop_layers = 0_zu;
        size_t total_dropout_uout_layers = 0_zu;
        size_t total_dropout_zoneout_layers = 0_zu;
        size_t total_k_sparse_layers = 0_zu;
        size_t total_tied_parameter_layers = 0_zu;
        size_t total_constraint_recurrent_weight_layers = 0_zu;

        double percentage_maximum_thread_usage = 100.0;
        double cache_maximum_threads_percent = 0.0;
        
        T_ Warm_Restarts_Decay(void);
        T_ Normalized_Weight_Decay(size_t const batch_size_received, size_t const training_size_received);
        T_ Get__Regularization__Max_Norm_Constraints(void) const;
        T_ Get__Regularization__L1(void) const;
        T_ Get__Regularization__L2(void) const;
        T_ Get__Regularization__SRIP(void) const;
        T_ Activation_Function(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received,T_ summation_received);
        T_ Activation_Function_Derive(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received,
                                                    T_ summation_received,
                                                    T_ steepness_received,
                                                    T_ value_received);
        T_ Initialization__Gain__Scale(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received);
        T_ Initialization__Gaussian__Variance(size_t const fan_in_received,
                                                                size_t const fan_out_received,
                                                                enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION const type_layer_activation_received);
        T_ Initialization__Uniform__Variance(size_t const fan_in_received,
                                                             size_t const fan_out_received,
                                                             enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION const type_layer_activation_received);
        void Forward_Pass(size_t const batch_size_received,
                                    T_ const *const *const ptr_array_inputs_received,
                                    long long int input_layer_index_received = -1ll,
                                    long long int output_layer_index_received = -1ll);
        void Forward_Pass__Pre_Training(size_t const batch_size_received, T_ const *const *const ptr_array_inputs_received);
        void FF__Assign_Inputs__Loop(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received);
        void FF__Assign_Inputs__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received);
        void RNN__Assign_Inputs__Loop(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received);
        void RNN__Assign_Inputs__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received);
        void FF__Assign_Inputs__Pre_Training__Loop(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received);
        void FF__Assign_Inputs__Pre_Training__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received);
        void RNN__Assign_Inputs__Pre_Training__Loop(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received);
        void RNN__Assign_Inputs__Pre_Training__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received);
        void Clear_Outputs(void);
        std::pair<T_, T_> Compute__Regularization__Constraint_Recurrent_Weight__Default(size_t const index_layer_received) const;
        std::pair<T_, T_> Compute__Regularization__Constraint_Recurrent_Weight__Default(struct Layer *const ptr_layer_received) const;
        T_ Get__Accuracy(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received) const;
        T_ Get__Loss(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received) const;
        T_ Get__ME(void) const;
        T_ Get__Loss_L1(void) const;
        T_ Get__MAE(void) const;
        T_ Get__Loss_L2(void) const;
        T_ Get__MSE(void) const;
        T_ Get__RMSE(void) const;
        T_ Get__MAPE(void) const;
        T_ Get__SMAPE(void) const;
        T_ Get__MASE(void) const;
        T_ Get__ACE(void) const;
        T_ Get__BITFAIL(void) const;
        T_ const *Get__Outputs(size_t const data_index_received, size_t const time_step_index_received = 0_zu) const;
        T_ const *Get__Outputs(struct Layer const *const ptr_layer_it_received, size_t const data_index_received, size_t const time_step_index_received = 0_zu) const;
        T_ Get__Outputs__Variance(size_t const layer_index_received, size_t const maximum_batch_size_received) const;
        T_ Get__Outputs__Variance(struct Layer const *const ptr_layer_received, size_t const maximum_batch_size_received) const;
        T_ *ptr_array_loss_values = nullptr; // size[N].
        T_ *ptr_array_accuracy_values[5u] = {nullptr}; // size[N].
        T_ loss_training = (std::numeric_limits<ST_>::max)();
        T_ loss_validating = (std::numeric_limits<ST_>::max)();
        T_ loss_testing = (std::numeric_limits<ST_>::max)();
        T_ loss_rprop = (std::numeric_limits<ST_>::max)();
        T_ previous_loss_rprop = (std::numeric_limits<ST_>::max)();
        T_ accuracy_variance = 0.49_T;
        T_ accuracy_training = 0_T;
        T_ accuracy_validating = 0_T;
        T_ accuracy_testing = 0_T;
        T_ learning_rate = 0.01_T;
        T_ learning_rate_final = 0.1_T;
        T_ learning_momentum = 0.9_T;
        T_ learning_gamma = 1e-3_T;
        T_ bit_fail_limit = 0.35_T;
        T_ regularization__max_norm_constraints = 0_T;
        T_ regularization__l1 = 0_T;
        T_ regularization__l2 = 0_T;
        T_ regularization__srip = 0_T;
        T_ regularization__weight_decay = 0_T;
        T_ adam_learning_rate = 0.001_T;
        T_ adam_beta1 = 0.9_T;
        T_ adam_beta2 = 0.999_T; // {0.99, 0.999}
        T_ adam_previous_beta2 = 0_T;
        T_ adam_epsilon = 1.0e-8_T;
        T_ adam_gamma = 0.1_T; // {0.05, 0.1}
        T_ optimizer_time_step = 0_T;
        T_ epoch_time_step = 1_T;
        T_ warm_restarts_decay_learning_rate = 1_T;
        T_ warm_restarts_initial_maximum_learning_rate = 1_T;
        T_ warm_restarts_maximum_learning_rate = 1_T;
        T_ warm_restarts_minimum_learning_rate = 1.0e-7_T;
        T_ warm_restarts_initial_T_i = 1_T;
        T_ warm_restarts_T_i = 1_T;
        T_ warm_restarts_multiplier = 2_T;
        T_ clip_gradient = 1_T;
        T_ normalization_momentum_average = 0.999_T;
        T_ normalization_epsilon = 1.0e-5_T;
        T_ batch_renormalization_r_correction_maximum = 1_T;
        T_ batch_renormalization_d_correction_maximum = 0_T;
        T_ *ptr_array_basic_indice_units_values = nullptr; // size[B, T, H].
        T_ *ptr_array_basic_indice_units_errors = nullptr; // size[B, T, H].
        T_ *ptr_array_basic_units_values = nullptr; // size[B, T, H].
        T_ *ptr_array_basic_units_errors = nullptr; // size[B, T, H].
        T_ *ptr_array_neuron_units_summations = nullptr; // size[B, T, H].
        T_ *ptr_array_neuron_units_errors = nullptr; // size[B, T, H].
        T_ *ptr_array_AF_units_activation_steepness = nullptr; // size[H].
        T_ *ptr_array_AF_units_values = nullptr; // size[B, T, H].
        T_ *ptr_array_AF_units_errors = nullptr; // size[B, T, H].
        T_ *ptr_array_AF_Ind_recurrent_units_activation_steepness = nullptr; // size[H].
        T_ *ptr_array_AF_Ind_recurrent_units_pre_AFs = nullptr; // size[B, T, H].
        T_ *ptr_array_AF_Ind_recurrent_units_AFs = nullptr; // size[B, T, H].
        T_ *ptr_array_AF_Ind_recurrent_units_errors = nullptr; // size[B, T, H].
        T_ *ptr_array_AF_Ind_recurrent_units_dAFs = nullptr; // size[B, T, H].
        T_ *ptr_array_cells_summations_cells_inputs = nullptr; // size[B, T, H].
        T_ *ptr_array_cells_summations_input_cells_inputs = nullptr; // size[B, T, H].
        T_ *ptr_array_cells_summations_recurrent_cells_inputs = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_summations_inputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_summations_input_inputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_summations_recurrent_inputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_summations_forgets_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_summations_input_forgets_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_summations_recurrent_forgets_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_summations_outputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_summations_input_outputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_summations_recurrent_outputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_cells_inputs = nullptr; // size[B, T, H].
        T_ *ptr_array_cells_states = nullptr; // size[B, T, H].
        T_ *ptr_array_cells_states_activates = nullptr; // size[B, T, H].
        T_ *ptr_array_cells_outputs = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_inputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_forgets_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_outputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_cells_delta_inputs = nullptr; // size[B, T, H].
        T_ *ptr_array_cells_delta_input_inputs = nullptr; // size[B, T, H].
        T_ *ptr_array_cells_delta_recurrent_inputs = nullptr; // size[B, T, H].
        T_ *ptr_array_cells_delta_states = nullptr; // size[B, T, H].
        T_ *ptr_array_cells_delta_outputs = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_delta_inputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_delta_input_inputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_delta_recurrent_inputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_delta_forgets_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_delta_input_forgets_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_delta_recurrent_forgets_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_delta_outputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_delta_input_outputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_blocks_delta_recurrent_outputs_gates = nullptr; // size[B, T, H].
        T_ *ptr_array_derivatives_parameters = nullptr; // size[N, P].
        T_ *ptr_array_previous_steps = nullptr; // size[P].
        T_ *ptr_array_previous_delta_parameters = nullptr; // size[P].
        T_ *ptr_array_previous_derivatives_parameters = nullptr; // size[P].
        T_ *ptr_array_parameters = nullptr; // size[P].
        T_ *ptr_array_mask_regularized_parameters = nullptr; // size[P].
        T_ *ptr_array_previous_biased_first_moment = nullptr; // size[P].
        T_ *ptr_array_previous_biased_second_moment = nullptr; // size[P].
        T_ *ptr_array_previous_biased_second_moment_hat = nullptr; // size[P].
        T_ *ptr_array_normalized_batch_units_values_hats = nullptr; // size[B, T, H]. Batch normalization variable.
        T_ *ptr_array_normalized_batch_units_values_normalizes = nullptr; // size[B, T, H]. Batch normalization variable.
        T_ *ptr_array_normalized_batch_units_scales = nullptr; // size[H]. Batch normalization variable.
        T_ *ptr_array_normalized_batch_units_shifts = nullptr; // size[H]. Batch normalization variable.
        T_ *ptr_array_normalized_batch_units_derivatives_scales = nullptr; // size[N, H]. Batch normalization variable.
        T_ *ptr_array_normalized_batch_units_derivatives_shifts = nullptr; // size[N, H]. Batch normalization variable.
        T_ *ptr_array_normalized_batch_units_means = nullptr; // size[N, T, H]. Batch normalization variable.
        T_ *ptr_array_normalized_batch_units_variances = nullptr; // size[N, T, H]. Batch normalization variable.
        T_ *ptr_array_normalized_batch_units_derivatives_means = nullptr; // size[N, T, H]. Batch normalization variable.
        T_ *ptr_array_normalized_batch_units_derivatives_variances = nullptr; // size[N, T, H]. Batch normalization variable.
        T_ *ptr_array_normalized_batch_units_r_corrections = nullptr; // size[T, H]. Batch renormalization variable.
        T_ *ptr_array_normalized_batch_units_d_corrections = nullptr; // size[T, H]. Batch renormalization variable.
        T_ *ptr_array_normalized_batch_units_means_averages = nullptr; // size[T, H]. Batch normalization variable.
        T_ *ptr_array_normalized_batch_units_variances_averages = nullptr; // size[T, H]. Batch normalization variable.
        T_ *ptr_array_normalized_batch_units_errors = nullptr; // size[B, T, H]. Batch normalization variable.

        enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION *ptr_array_AF_units_type_activation_function = nullptr; // size[H].
        enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION *ptr_array_AF_Ind_recurrent_units_type_activation_function = nullptr; // size[H].
        enum MyEA::Common::ENUM_TYPE_NETWORKS type_network = MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_NONE;
        enum MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS type_optimizer_function = MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NONE;
        enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS type_loss_function = MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_NONE;
        enum MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS type_accuracy_function = MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_DISTANCE;
        enum MyEA::Common::ENUM_TYPE_STATE_PROPAGATION type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_INFERENCE; // Dropout && Batch normalization variable
        enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION Activation_Function__To__Class_Activation_Function(enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received) const;
        
        class MyEA::Common::Class_Generator_Random_Bernoulli<T_> *ptr_array_Class_Generator_Bernoulli = nullptr;
        class MyEA::Common::Class_Generator_Random_Bernoulli<T_> *ptr_array_Class_Generator_Bernoulli_ShakeDrop = nullptr;
        class MyEA::Common::Class_Generator_Random_Bernoulli<T_> *ptr_array_Class_Generator_Bernoulli_Zoneout_State = nullptr;
        class MyEA::Common::Class_Generator_Random_Bernoulli<T_> *ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden = nullptr;
        class MyEA::Common::Class_Generator_Random_Real<T_> *ptr_array_Class_Generator_Real_ShakeDrop = nullptr;
        class MyEA::Common::Class_Generator_Random_Real<T_> *ptr_array_Class_Generator_Real_Uout = nullptr;
        class MyEA::Common::Class_Generator_Random_Real<T_> Class_Generator_Real;
        class MyEA::Common::Class_Generator_Random_Gaussian<T_> *ptr_array_Class_Generator_Real_Gaussian = nullptr;
        class MyEA::Common::Class_Generator_Random_Gaussian<T_> Class_Generator_Gaussian;
        
        bool Initialize__LSUV(size_t const maximum_number_trials_received = 10_zu,
                                        size_t const maximum_batch_size_received = 32_zu,
                                        T_ const bias_received = 0_T,
                                        T_ const variance_target_received = 1_T,
                                        T_ const variance_tolerance_received = 0.01_T);

        void Initialization__Glorot__Gaussian(T_ const bias_received = 0_T);
        void Initialization__Glorot__Uniform(T_ const bias_received = 0_T);
        void Initialization__Identity(T_ const bias_received = 0_T);
        bool Initialization__LSUV(class Dataset<T_> const *const ptr_Dataset_received);
        bool Initialization__LSUV__Loop(class Dataset<T_> const *const ptr_Dataset_received);
        bool Initialization__LSUV__OpenMP(class Dataset<T_> const *const ptr_Dataset_received);
        void Initialization__Orthogonal(bool const pre_initialize_received = false, T_ const bias_received = 0_T);
        void Initialization__Uniform(T_ const bias_received = 0_T,
                                               T_ const lower_bound_received = -1_T,
                                               T_ const upper_bound_received = 1_T);
        void **ptr_array_ptr_connections = nullptr;
        
        struct Layer const *Get__Layer(size_t const index_received) const;
        struct Layer const *Get__End_Layer__Active(void) const;
        struct Layer *ptr_array_layers = nullptr;
        struct Layer *ptr_last_layer = nullptr;

        struct Bidirectional_Layer *ptr_array_bidirectional_layers = nullptr;
        struct Bidirectional_Layer *ptr_last_bidirectional_layer = nullptr;
        
        struct Basic_unit *ptr_array_basic_units = nullptr;
        struct Basic_unit *ptr_last_basic_unit = nullptr;
        
        struct Basic_indice_unit *ptr_array_basic_indice_units = nullptr;
        struct Basic_indice_unit *ptr_last_basic_indice_unit = nullptr;
        
        struct Neuron_unit *ptr_array_neuron_units = nullptr;
        struct Neuron_unit *ptr_last_neuron_unit = nullptr;
        
        struct AF_unit *ptr_array_AF_units = nullptr;
        struct AF_unit *ptr_last_AF_unit = nullptr;
        
        struct AF_Ind_recurrent_unit *ptr_array_AF_Ind_recurrent_units = nullptr;
        struct AF_Ind_recurrent_unit *ptr_last_AF_Ind_recurrent_unit = nullptr;
        
        union Normalized_unit *ptr_array_normalized_units = nullptr;
        union Normalized_unit *ptr_last_normalized_unit = nullptr;
        
        struct Block_unit *ptr_array_block_units = nullptr;
        struct Block_unit *ptr_last_block_unit = nullptr;

        struct Cell_unit *ptr_array_cell_units = nullptr;
        struct Cell_unit *ptr_last_cell_unit = nullptr;
        
    #if defined(COMPILE_CUDA)
        void Clear_Training_Arrays__CUDA(void);
        void Copy__Parameters__Host_To_Device(void);

        bool Use__CUDA(void) const;
        bool Set__CUDA(bool const use_cuda_received, size_t const maximum_allowable_memory_received);
        bool Initialize__CUDA(size_t const maximum_allowable_memory_received);
        bool Initialize__CUDA__Thread(class Dataset_Manager<T_> const *const ptr_Dataset_Manager_received);
        bool Copy_Device_To_Host(bool const refresh_from_genetic_algorithm_received);
        bool Copy__Parameters__Device_To_Host(void);
        bool Copy__Optimizer_Paramaters__Device_To_Host(void);
        bool Copy__Optimizer_Gradient_Descent__Device_To_Host(void);
        bool Copy__Optimizer_RPROP_minus__Device_To_Host(void);
        bool Copy__Optimizer_RPROP_plus__Device_To_Host(void);
        bool Copy__Optimizer_Adam__Device_To_Host(void);
        bool Copy__Optimizer_AMSGrad__Device_To_Host(void);
        bool Copy__Batch_Normalization_Neurons__Device_To_Host(void);
        template<typename T>
        bool Copy__Optimizer_Gradient_Descent__Device_To_Host(T &ref_optimizer_time_step_received,
                                                                                               T &ref_warm_maximum_learning_rate_received,
                                                                                               T &ref_warm_T_i_received,
                                                                                               T *const ptr_array_previous_delta_parameters_received) const;
        template<typename T>
        bool Copy__Optimizer_RPROP_minus__Device_To_Host(T *const ptr_array_previous_steps_received, T *const ptr_array_previous_derivates_parameters_received) const;
        template<typename T>
        bool Copy__Optimizer_RPROP_plus__Device_To_Host(T &ref_loss_received,
                                                                                        T &ref_previous_loss_received,
                                                                                        T *const ptr_array_previous_steps_received,
                                                                                        T *const ptr_array_previous_derivates_parameters_received,
                                                                                        T *const ptr_array_previous_delta_parameters_received) const;
        template<typename T>
        bool Copy__Optimizer_Adam__Device_To_Host(T &ref_optimizer_time_step_received,
                                                                              T &ref_warm_maximum_learning_rate_received,
                                                                              T &ref_warm_T_i_received,
                                                                              T *const ptr_array_previous_biased_first_moment_received,
                                                                              T *const ptr_array_previous_biased_second_moment_received) const;
        template<typename T>
        bool Copy__Optimizer_AMSGrad__Device_To_Host(T &ref_optimizer_time_step_received,
                                                                                   T &ref_warm_maximum_learning_rate_received,
                                                                                   T &ref_warm_T_i_received,
                                                                                   T *const ptr_array_previous_biased_first_moment_received,
                                                                                   T *const ptr_array_previous_biased_second_moment_received,
                                                                                   T *const ptr_array_previous_biased_second_moment_hat_received) const;
        template<typename T>
        bool Copy__Batch_Normalization_Neurons__Device_To_Host(T *const ptr_array_neuron_units_scale_received,
                                                                                                T *const ptr_array_neuron_units_shift_received,
                                                                                                T *const ptr_array_neuron_units_mean_average_received,
                                                                                                T *const ptr_array_neuron_units_variance_average_received) const;
        template<typename T>
        bool Copy__Optimizer_Gradient_Descent__Host_To_Device(T const optimizer_time_step_received,
                                                                                               T const warm_restarts_maximum_learning_rate_received,
                                                                                               T const warm_restarts_T_i_received,
                                                                                               T const *const ptr_array_previous_delta_parameters_received);
        template<typename T>
        bool Copy__Optimizer_RPROP_minus__Host_To_Device(T const *const ptr_array_previous_steps_received, T const *const ptr_array_previous_derivates_parameters_received);
        template<typename T>
        bool Copy__Optimizer_RPROP_plus__Host_To_Device(T const loss_received,
                                                                                        T const previous_loss_received,
                                                                                        T const *const ptr_array_previous_steps_received,
                                                                                        T const *const ptr_array_previous_derivates_parameters_received,
                                                                                        T const *const ptr_array_previous_delta_parameters_received);
        template<typename T>
        bool Copy__Optimizer_Adam__Host_To_Device(T const optimizer_time_step_received,
                                                                              T const warm_restarts_maximum_learning_rate_received,
                                                                              T const warm_restarts_T_i_received,
                                                                              T const *const ptr_array_previous_biased_first_moment_received,
                                                                              T const *const ptr_array_previous_biased_second_moment_received);
        template<typename T>
        bool Copy__Optimizer_AMSGrad__Host_To_Device(T const optimizer_time_step_received,
                                                                                    T const warm_restarts_maximum_learning_rate_received,
                                                                                    T const warm_restarts_T_i_received,
                                                                                    T const *const ptr_array_previous_biased_first_moment_received,
                                                                                    T const *const ptr_array_previous_biased_second_moment_received,
                                                                                    T const *const ptr_array_previous_biased_second_moment_hat_received);
        template<typename T>
        bool Copy__Batch_Normalization_Neurons__Host_To_Device(T const *const ptr_array_neuron_units_scale_received,
                                                                                                T const *const ptr_array_neuron_units_shift_received,
                                                                                                T const *const ptr_array_neuron_units_mean_average_received,
                                                                                                T const *const ptr_array_neuron_units_variance_average_received) const;
        bool Deinitialize__CUDA(void);
        bool is_update_from_device = true;
        bool is_device_initialized = false;
        bool use_CUDA = false;

        class CUDA_Neural_Network *ptr_device_Neural_Network = NULL;
    #endif

        std::string Get__Parameters(bool const full_description_received = false);
    
        size_t Get__Sizeof(size_t number_threads_received = 0u, size_t batch_size_received = 0u) const;
        size_t Get__Batch_Sizeof(size_t batch_size_received = 0u) const;
        size_t Get__Threads_Sizeof(size_t number_threads_received = 0u) const;
        size_t Get__Input_Size(void) const;
        size_t Get__Output_Size(void) const;
        size_t maximum_allowable_memory_bytes = 0u; // Bytes.

    #ifdef USE_FIXED
        /* the decimal_point, used for shifting the fix point
         * in fixed point integer operatons.
         */
        unsigned int decimal_point;

        /* the multiplier, used for multiplying the fix point
         * in fixed point integer operatons.
         * Only used in special cases, since the decimal_point is much faster.
         */
        unsigned int multiplier;

        /* When in choosen (or in fixed point), the sigmoid function is
         * calculated as a stepwise linear function. In the
         * activation_results array, the result is saved, and in the
         * two values arrays, the values that gives the results are saved.
         */
        T_ sigmoid_results[6];
        T_ sigmoid_values[6];
        T_ sigmoid_symmetric_results[6];
        T_ sigmoid_symmetric_values[6];
    #endif
        
        /* Variables for use with Quickprop training */
        /* Decay is used to make the ptr_array_parameters not go so high */
        T_ quickprop_decay = -0.0001_T;
        /* Mu is a factor used to increase and decrease the stepsize */
        T_ quickprop_mu = 1.75_T;

        /* Variables for use with with RPROP training */
        /* Tells how much the stepsize should increase during learning */
        T_ rprop_increase_factor = 1.2_T;
        /* Tells how much the stepsize should decrease during learning */
        T_ rprop_decrease_factor = 0.5_T;
        /* The minimum stepsize */
        T_ rprop_delta_min = 1.0e-6_T;
        /* The maximum stepsize */
        T_ rprop_delta_max = 50.0_T;
        /* The initial stepsize */
        T_ rprop_delta_zero = 0.1_T;
        
        /* Defines how much the ptr_array_parameters are constrained to smaller values at the beginning */
        T_ sarprop_weight_decay_shift = -6.644_T;
        /* Decides if the stepsize is too big with regard to the error */
        T_ sarprop_step_error_threshold_factor = 0.1_T;
        /* Defines how much the stepsize is influenced by the error */
        T_ sarprop_step_error_shift = 1.385_T;
        /* Defines how much the epoch influences weight decay and noise */
        T_ sarprop_temperature = 0.015_T;
        /* Current training epoch */
        size_t sarprop_epoch = 0u;

    private:
        bool _initialized__weight = true;
        
        enum MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS _type_weights_initializer = MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_NONE;

        struct LSUV_Parameters _LSUV_Parameters;

        // Need to be call in a sequential layer [0, ..., L - 1].
        void Order__Layer__Normalization_Iterator(struct Layer *const ptr_layer_it_received);

        bool Strategy_Comparison__Loss(unsigned int const strategy_index_received,
                                                          enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_in_received,
                                                          enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_out_received,
                                                          class Neural_Network const *const ptr_source_Neural_Network_received) const;
        bool Strategy_Comparison__Accuracy(unsigned int const strategy_index_received,
                                                                enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_in_received,
                                                                enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_out_received,
                                                                class Neural_Network const *const ptr_source_Neural_Network_received) const;
        bool Set__Layer_Activation_Function__AF(struct Layer *const ptr_layer_it_received, enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received);
        bool Set__Layer_Activation_Function__AF_Ind_Recurrent(struct Layer *const ptr_layer_it_received, enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received);
        bool Set__Layer_Activation_Function__LSTM(struct Layer *const ptr_layer_it_received, enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received);
        bool Set__Layer_Activation_Steepness__AF(struct Layer *const ptr_layer_it_received, T_ const activation_steepness_received);
        bool Set__Layer_Activation_Steepness__AF_Ind_Recurrent(struct Layer *const ptr_layer_it_received, T_ const activation_steepness_received);

        struct Layer *Get__Input_Layer(void) const;
        struct Layer *Get__Output_Layer(void) const;
        void Organize__Previous_Layers_Connected(size_t &ref_state_layer_index_received,
                                                                         struct Layer *const ptr_layer_received,
                                                                         struct Layer const *&ptr_layer_state_received) const;
        void Organize__Next_Layers_Connected(size_t &ref_state_layer_index_received,
                                                                    struct Layer *const ptr_layer_received,
                                                                    struct Layer const *&ptr_layer_state_received) const;
        void Organize__Layer__Group(size_t &ref_state_layer_index_received,
                                                    struct Layer *const ptr_layer_received,
                                                    struct Layer const *&ptr_layer_state_received) const;
};