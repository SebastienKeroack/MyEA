#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

Neural_Network::~Neural_Network(void) { this->Deallocate(); }

void Neural_Network::Clear(void)
{
    if(this->type_network != MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_NONE)
    {
        this->type_network = MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_NONE;
        this->type_optimizer_function = MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NONE;

        this->Deallocate();
    }
}

void Neural_Network::Deallocate(void)
{
    // Delete basic unit variable.
    SAFE_DELETE_ARRAY(this->ptr_array_basic_units);
    SAFE_DELETE_ARRAY(this->ptr_array_basic_units_values); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_basic_units_errors); // delete[] array T_.
    
    // Delete basic indice unit variable.
    SAFE_DELETE_ARRAY(this->ptr_array_basic_indice_units);
    SAFE_DELETE_ARRAY(this->ptr_array_basic_indice_units_indices); // delete[] array size_t
    SAFE_DELETE_ARRAY(this->ptr_array_basic_indice_units_values); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_basic_indice_units_errors); // delete[] array T_.

    // Delete block(s)/cell(s) variable.
    SAFE_DELETE_ARRAY(this->ptr_array_cells_summations_cells_inputs);
    SAFE_DELETE_ARRAY(this->ptr_array_cells_summations_input_cells_inputs);
    SAFE_DELETE_ARRAY(this->ptr_array_cells_summations_recurrent_cells_inputs);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_inputs_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_input_inputs_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_recurrent_inputs_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_forgets_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_input_forgets_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_recurrent_forgets_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_outputs_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_input_outputs_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_recurrent_outputs_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_cells_inputs);
    SAFE_DELETE_ARRAY(this->ptr_array_cells_states);
    SAFE_DELETE_ARRAY(this->ptr_array_cells_states_activates);
    SAFE_DELETE_ARRAY(this->ptr_array_cells_outputs);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_inputs_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_forgets_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_outputs_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_cells_delta_inputs);
    SAFE_DELETE_ARRAY(this->ptr_array_cells_delta_input_inputs);
    SAFE_DELETE_ARRAY(this->ptr_array_cells_delta_recurrent_inputs);
    SAFE_DELETE_ARRAY(this->ptr_array_cells_delta_states);
    SAFE_DELETE_ARRAY(this->ptr_array_cells_delta_outputs);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_inputs_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_input_inputs_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_recurrent_inputs_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_forgets_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_input_forgets_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_recurrent_forgets_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_outputs_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_input_outputs_gates);
    SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_recurrent_outputs_gates);
        
    SAFE_DELETE_ARRAY(this->ptr_array_block_units);
    SAFE_DELETE_ARRAY(this->ptr_array_cell_units);
    // |END| Delete block(s)/cell(s) variable. |END|
    
    // Delete neuron unit(s) variable.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_first_forward_connection_index); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_last_forward_connection_index); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_number_forward_connections); // delete[] array size_t.
    
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_summations); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_errors); // delete[] array T_.

    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units);
    // |END| Delete neuron unit(s) variable. |END|
    
    // Delete AF unit(s) variable.
    SAFE_DELETE_ARRAY(this->ptr_array_AF_units_activation_steepness); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_AF_units_values); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_AF_units_errors); // delete[] array T_.

    SAFE_DELETE_ARRAY(this->ptr_array_AF_units_type_activation_function); // delete[] array enum.
    SAFE_DELETE_ARRAY(this->ptr_array_AF_units);
    // |END| Delete AF unit(s) variable. |END|
    
    // Delete AF Ind unit(s) variable.
    SAFE_DELETE_ARRAY(this->ptr_array_AF_Ind_recurrent_units_recurrent_connection_index); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_AF_Ind_recurrent_units_activation_steepness); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_AF_Ind_recurrent_units_pre_AFs); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_AF_Ind_recurrent_units_AFs); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_AF_Ind_recurrent_units_errors); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_AF_Ind_recurrent_units_dAFs); // delete[] array T_.

    SAFE_DELETE_ARRAY(this->ptr_array_AF_Ind_recurrent_units_type_activation_function); // delete[] array enum.
    SAFE_DELETE_ARRAY(this->ptr_array_AF_Ind_recurrent_units);
    // |END| Delete AF Ind unit(s) variable. |END|
    
    SAFE_DELETE_ARRAY(this->ptr_array_layers);
    SAFE_DELETE_ARRAY(this->ptr_array_layers_number_outputs); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_layers_first_connection_index); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_layers_last_connection_index); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_bidirectional_layers);
    
    if(this->Use__Dropout__Bernoulli() || this->Use__Dropout__Bernoulli__Inverted())
    {
        this->Deallocate__Generator__Dropout_Bernoulli();

        this->Deallocate__Neuron__Mask_Dropout_Bernoulli();
    }

    if(this->Use__Dropout__Gaussian()) { this->Deallocate__Generator__Dropout_Gaussian(); }

    if(this->Use__Dropout__ShakeDrop())
    {
        this->Deallocate__Generator__Dropout_ShakeDrop();

        this->Deallocate__Layer__Mask_Dropout_ShakeDrop();
    }

    if(this->Use__Dropout__Uout()) { this->Deallocate__Generator__Dropout_Uout(); }

    if(this->Use__Dropout__Zoneout())
    {
        this->Deallocate__Generator__Dropout_Zoneout();

        this->Deallocate__Cell_Unit__Mask_Dropout_Zoneout();
    }
    
    if(this->Use__Normalization())
    {
        this->Deallocate__Normalized_Unit();

        this->Deallocate__Normalized_Unit__Batch_Normalization();
    }

    if(this->Use__Batch_Renormalization()) { this->Deallocate__Normalized_Unit__Batch_Renormalization(); }

    SAFE_DELETE_ARRAY(this->ptr_array_parameters);

    this->Deallocate__Parameter__Regularization();

#if defined(COMPILE_CUDA)
    if(this->Deinitialize__CUDA() == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Deinitialize__CUDA()\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);
    }
#endif
    
    this->Deallocate__Sparse_K_Filter();
    
    // Deallocate optimizer array.
    this->Deallocate__Parameter__Optimizer();
    // |END| Deallocate optimizer array. |END|
    
    SAFE_DELETE_ARRAY(this->ptr_array_ptr_connections);
    SAFE_DELETE_ARRAY(this->ptr_array_derivatives_parameters);
    
    // Loss parameters.
    SAFE_DELETE_ARRAY(this->ptr_array_number_loss);
    SAFE_DELETE_ARRAY(this->ptr_array_number_bit_fail);
    SAFE_DELETE_ARRAY(this->ptr_array_loss_values);
    // |END| Loss parameters. |END|
        
    // Accuracy parameters.
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[0u]);
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[1u]);
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[2u]);
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[3u]);
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[4u]);
    // |END| Accuracy parameters. |END|
}

void Neural_Network::Deallocate__Neuron__Mask_Dropout_Bernoulli(void) { SAFE_DELETE_ARRAY(this->ptr_array_units_mask_dropout_bernoulli); }

void Neural_Network::Deallocate__Layer__Mask_Dropout_ShakeDrop(void) { SAFE_DELETE_ARRAY(this->ptr_array_layers_mask_dropout_shakedrop); }

void Neural_Network::Deallocate__Cell_Unit__Mask_Dropout_Zoneout(void) { SAFE_DELETE_ARRAY(this->ptr_array_cell_units_mask_dropout_zoneout); }

void Neural_Network::Deallocate__Sparse_K_Filter(void) { SAFE_DELETE_ARRAY(this->ptr_array_k_sparse_activities); }

void Neural_Network::Deallocate__Parameter__Optimizer(void)
{
    switch(this->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NONE: break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD: this->Deallocate__Parameter__Gradient_Descent(); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus: this->Deallocate__Parameter__iRPROP_minus(); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus: this->Deallocate__Parameter__iRPROP_plus(); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADABOUND:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: this->Deallocate__Parameter__Adam(); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSBOUND:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: this->Deallocate__Parameter__AMSGrad(); break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Optimizer function type (%u | %s) is not managed in the switch." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     this->type_optimizer_function,
                                     MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS_NAMES[this->type_optimizer_function].c_str());
                break;
    }
}

void Neural_Network::Deallocate__Parameter__Gradient_Descent(void) { SAFE_DELETE_ARRAY(this->ptr_array_previous_delta_parameters); }

void Neural_Network::Deallocate__Parameter__iRPROP_minus(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_previous_steps);
    SAFE_DELETE_ARRAY(this->ptr_array_previous_derivatives_parameters);
}

void Neural_Network::Deallocate__Parameter__iRPROP_plus(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_previous_steps);
    SAFE_DELETE_ARRAY(this->ptr_array_previous_delta_parameters);
    SAFE_DELETE_ARRAY(this->ptr_array_previous_derivatives_parameters);
}

void Neural_Network::Deallocate__Parameter__Adam(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_previous_biased_first_moment);
    SAFE_DELETE_ARRAY(this->ptr_array_previous_biased_second_moment);
}

void Neural_Network::Deallocate__Parameter__AMSGrad(void)
{
    this->Deallocate__Parameter__Adam();

    SAFE_DELETE_ARRAY(this->ptr_array_previous_biased_second_moment_hat);
}

void Neural_Network::Deallocate__Generator__Dropout_Bernoulli(void) { SAFE_DELETE_ARRAY(this->ptr_array_Class_Generator_Bernoulli); }

void Neural_Network::Deallocate__Generator__Dropout_Gaussian(void) { SAFE_DELETE_ARRAY(this->ptr_array_Class_Generator_Real_Gaussian); }

void Neural_Network::Deallocate__Generator__Dropout_ShakeDrop(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_Class_Generator_Bernoulli_ShakeDrop);
    SAFE_DELETE_ARRAY(this->ptr_array_Class_Generator_Real_ShakeDrop);
}

void Neural_Network::Deallocate__Generator__Dropout_Uout(void) { SAFE_DELETE_ARRAY(this->ptr_array_Class_Generator_Real_Uout); }

void Neural_Network::Deallocate__Generator__Dropout_Zoneout(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_Class_Generator_Bernoulli_Zoneout_State);
    SAFE_DELETE_ARRAY(this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden);
}

void Neural_Network::Deallocate__Parameter__Batch_Normalization(void)
{
    if(this->total_normalized_units_allocated != 0_zu)
    {
        size_t const tmp_new_dimension_parameters(this->total_parameters_allocated - 2_zu * this->total_normalized_units_allocated);
        
        if(this->Reallocate__Parameter(tmp_new_dimension_parameters) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Reallocate__Parameter(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     tmp_new_dimension_parameters,
                                     __LINE__);

            return;
        }

        this->total_normalized_units_allocated = this->total_normalized_units = 0_zu;
    }
}

void Neural_Network::Deallocate__Normalized_Unit(void) { SAFE_DELETE_ARRAY(this->ptr_array_normalized_units); }

void Neural_Network::Deallocate__Normalized_Unit__Batch_Normalization(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_values_hats); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_values_normalizes); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_means); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_variances); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_derivatives_means); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_derivatives_variances); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_means_averages); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_variances_averages); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_errors); // delete[] array T_.
}

void Neural_Network::Deallocate__Normalized_Unit__Batch_Renormalization(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_r_corrections); // delete[] array T_.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_d_corrections); // delete[] array T_.
}

void Neural_Network::Deallocate__Parameter__Regularization(void) { SAFE_DELETE_ARRAY(this->ptr_array_mask_regularized_parameters); }
