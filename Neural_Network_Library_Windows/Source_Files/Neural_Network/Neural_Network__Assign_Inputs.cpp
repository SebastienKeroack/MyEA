#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

void Neural_Network::FF__Assign_Inputs__Loop(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received)
{
    struct Layer *const tmp_ptr_input_layer(this->ptr_array_layers);

    struct AF_unit *const tmp_ptr_input_layer_ptr_first_AF_unit(tmp_ptr_input_layer->ptr_array_AF_units);
    
    size_t const tmp_number_inputs(static_cast<size_t>(tmp_ptr_input_layer->ptr_last_AF_unit - tmp_ptr_input_layer_ptr_first_AF_unit));
    size_t tmp_example_index;

    T_ const *tmp_ptr_array_inputs_it;
    T_ *tmp_ptr_array_values_it;

    // Loop through each sample data.
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_values_it = tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values + tmp_example_index * tmp_number_inputs;
        tmp_ptr_array_inputs_it = ptr_matrix_inputs_received[tmp_example_index];
        
        MEMCPY(tmp_ptr_array_values_it,
                       tmp_ptr_array_inputs_it,
                       tmp_number_inputs * sizeof(T_));
    }

    if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Training__Loop(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                    0_zu,
                                                                                                    batch_size_received,
                                                                                                    tmp_number_inputs,
                                                                                                    tmp_ptr_input_layer->ptr_array_AF_units->ptr_array_values);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                this->Forward_Pass__Dropout__Bernoulli__Inverted__Loop(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                  0_zu,
                                                                                                  batch_size_received,
                                                                                                  tmp_number_inputs,
                                                                                                  tmp_ptr_input_layer->dropout_values[0u] == 0_T ? 0_T : 1_T / tmp_ptr_input_layer->dropout_values[0u],
                                                                                                  tmp_ptr_input_layer->ptr_array_AF_units->ptr_array_values);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                this->Forward_Pass__Dropout__Gaussian__Loop(0_zu,
                                                                                        batch_size_received,
                                                                                        tmp_number_inputs,
                                                                                        tmp_ptr_input_layer->dropout_values[0u],
                                                                                        tmp_ptr_input_layer->ptr_array_AF_units->ptr_array_values);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                this->Forward_Pass__Dropout__Uout__Loop(0_zu,
                                                                                batch_size_received,
                                                                                tmp_number_inputs,
                                                                                tmp_ptr_input_layer->dropout_values[0u],
                                                                                tmp_ptr_input_layer->ptr_array_AF_units->ptr_array_values);
                    break;
            // TODO: Alpha dropout forward pass.
            default: break;
        }
    }
    else
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(0_zu,
                                                                                      batch_size_received,
                                                                                      tmp_number_inputs,
                                                                                      tmp_ptr_input_layer->dropout_values[0u],
                                                                                      tmp_ptr_input_layer->ptr_array_AF_units->ptr_array_values);
                    break;
            default: break;
        }
    }
}

void Neural_Network::FF__Assign_Inputs__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    struct Layer *const tmp_ptr_input_layer(this->ptr_array_layers);

    struct AF_unit *const tmp_ptr_input_layer_ptr_first_AF_unit(tmp_ptr_input_layer->ptr_array_AF_units);
    
    size_t const tmp_number_inputs(static_cast<size_t>(tmp_ptr_input_layer->ptr_last_AF_unit - tmp_ptr_input_layer_ptr_first_AF_unit));

    T_ *tmp_ptr_array_values_it;
    T_ const *tmp_ptr_array_inputs_it,
                  *tmp_ptr_array_values_end;
    
    // Loop through each sample data.
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_inputs_it = ptr_matrix_inputs_received[tmp_example_index__int];
        tmp_ptr_array_values_it = tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values + static_cast<size_t>(tmp_example_index__int) * tmp_number_inputs;
        tmp_ptr_array_values_end = tmp_ptr_array_values_it + tmp_number_inputs;

        for(; tmp_ptr_array_values_it != tmp_ptr_array_values_end; ++tmp_ptr_array_values_it,
                                                                                             ++tmp_ptr_array_inputs_it)
        { *tmp_ptr_array_values_it = *tmp_ptr_array_inputs_it; }
    }
    
    if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Training__OpenMP(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                          0_zu,
                                                                                                          batch_size_received,
                                                                                                          tmp_number_inputs,
                                                                                                          tmp_ptr_input_layer->ptr_array_AF_units->ptr_array_values);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                this->Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                       0_zu,
                                                                                                       batch_size_received,
                                                                                                       tmp_number_inputs,
                                                                                                       tmp_ptr_input_layer->dropout_values[0u] == 0_T ? 0_T : 1_T / tmp_ptr_input_layer->dropout_values[0u],
                                                                                                       tmp_ptr_input_layer->ptr_array_AF_units->ptr_array_values);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                this->Forward_Pass__Dropout__Gaussian__OpenMP(0_zu,
                                                                                            batch_size_received,
                                                                                            tmp_number_inputs,
                                                                                            tmp_ptr_input_layer->dropout_values[0u],
                                                                                            tmp_ptr_input_layer->ptr_array_AF_units->ptr_array_values);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                this->Forward_Pass__Dropout__Uout__OpenMP(0_zu,
                                                                                    batch_size_received,
                                                                                    tmp_number_inputs,
                                                                                    tmp_ptr_input_layer->dropout_values[0u],
                                                                                    tmp_ptr_input_layer->ptr_array_AF_units->ptr_array_values);
                    break;
            default: break;
        }
    }
    else
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(0_zu,
                                                                                            batch_size_received,
                                                                                            tmp_number_inputs,
                                                                                            tmp_ptr_input_layer->dropout_values[0u],
                                                                                            tmp_ptr_input_layer->ptr_array_AF_units->ptr_array_values);
                    break;
            default: break;
        }
    }
}

void Neural_Network::RNN__Assign_Inputs__Loop(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received)
{
    struct Layer *const tmp_ptr_input_layer(this->ptr_array_layers);

    struct AF_unit *const tmp_ptr_input_layer_ptr_first_AF_unit(tmp_ptr_input_layer->ptr_array_AF_units);
    
    size_t const tmp_number_inputs(static_cast<size_t>(tmp_ptr_input_layer->ptr_last_AF_unit - tmp_ptr_input_layer_ptr_first_AF_unit));
    size_t tmp_example_index,
              tmp_time_step_index;

    T_ const *tmp_ptr_array_inputs_it;
    T_ *tmp_ptr_array_values_it;
    
    // Loop through each sample data.
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_inputs_it = ptr_matrix_inputs_received[tmp_example_index];

        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_ptr_array_values_it = tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values + tmp_example_index * tmp_number_inputs + this->batch_size * tmp_number_inputs * tmp_time_step_index;

            MEMCPY(tmp_ptr_array_values_it,
                           tmp_ptr_array_inputs_it,
                           tmp_number_inputs * sizeof(T_));

            tmp_ptr_array_inputs_it += tmp_number_inputs;
        }
    }
    
    if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Bernoulli__Training__Loop(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                        tmp_time_step_index,
                                                                                                        batch_size_received,
                                                                                                        tmp_number_inputs,
                                                                                                        tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Bernoulli__Inverted__Loop(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                      tmp_time_step_index,
                                                                                                      batch_size_received,
                                                                                                      tmp_number_inputs,
                                                                                                      tmp_ptr_input_layer->dropout_values[0u] == 0_T ? 0_T : 1_T / tmp_ptr_input_layer->dropout_values[0u],
                                                                                                      tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                this->Forward_Pass__Dropout__Gaussian__Loop(tmp_time_step_index,
                                                                                       batch_size_received,
                                                                                       tmp_number_inputs,
                                                                                       tmp_ptr_input_layer->dropout_values[0u],
                                                                                       tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Uout__Loop(tmp_time_step_index,
                                                                                    batch_size_received,
                                                                                    tmp_number_inputs,
                                                                                    tmp_ptr_input_layer->dropout_values[0u],
                                                                                    tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            default: break;
        }
    }
    else
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(tmp_time_step_index,
                                                                                                          batch_size_received,
                                                                                                          tmp_number_inputs,
                                                                                                          tmp_ptr_input_layer->dropout_values[0u],
                                                                                                          tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            default: break;
        }
    }
}

void Neural_Network::RNN__Assign_Inputs__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_data_index__int__int;
    
    struct Layer *const tmp_ptr_input_layer(this->ptr_array_layers);

    struct AF_unit *const tmp_ptr_input_layer_ptr_first_AF_unit(tmp_ptr_input_layer->ptr_array_AF_units);
    
    size_t const tmp_number_inputs(static_cast<size_t>(tmp_ptr_input_layer->ptr_last_AF_unit - tmp_ptr_input_layer_ptr_first_AF_unit));
    size_t tmp_time_step_index;

    T_ *tmp_ptr_array_values_it;
    T_ const *tmp_ptr_array_inputs_it,
                  *tmp_ptr_array_values_end;
    
    // Loop through each sample data.
    #pragma omp for schedule(static)
    for(tmp_data_index__int__int = 0; tmp_data_index__int__int < tmp_batch_size__int; ++tmp_data_index__int__int)
    {
        tmp_ptr_array_inputs_it = ptr_matrix_inputs_received[tmp_data_index__int__int];

        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_ptr_array_values_it = tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values + static_cast<size_t>(tmp_data_index__int__int) * tmp_number_inputs + this->batch_size * tmp_number_inputs * tmp_time_step_index;
            tmp_ptr_array_values_end = tmp_ptr_array_values_it + tmp_number_inputs;

            for(; tmp_ptr_array_values_it != tmp_ptr_array_values_end; ++tmp_ptr_array_values_it,
                                                                                                 ++tmp_ptr_array_inputs_it)
            { *tmp_ptr_array_values_it = *tmp_ptr_array_inputs_it; }
        }
    }
    
    if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING)
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Bernoulli__Training__OpenMP(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                            tmp_time_step_index,
                                                                                                            batch_size_received,
                                                                                                            tmp_number_inputs,
                                                                                                            tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                          tmp_time_step_index,
                                                                                                          batch_size_received,
                                                                                                          tmp_number_inputs,
                                                                                                          tmp_ptr_input_layer->dropout_values[0u] == 0_T ? 0_T : 1_T / tmp_ptr_input_layer->dropout_values[0u],
                                                                                                          tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                this->Forward_Pass__Dropout__Gaussian__OpenMP(tmp_time_step_index,
                                                                                           batch_size_received,
                                                                                           tmp_number_inputs,
                                                                                           tmp_ptr_input_layer->dropout_values[0u],
                                                                                           tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Uout__OpenMP(tmp_time_step_index,
                                                                                        batch_size_received,
                                                                                        tmp_number_inputs,
                                                                                        tmp_ptr_input_layer->dropout_values[0u],
                                                                                        tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            default: break;
        }
    }
    else
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(tmp_time_step_index,
                                                                                                                batch_size_received,
                                                                                                                tmp_number_inputs,
                                                                                                                tmp_ptr_input_layer->dropout_values[0u],
                                                                                                                tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            default: break;
        }
    }
}

void Neural_Network::FF__Assign_Inputs__Pre_Training__Loop(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received)
{
    struct Layer *const tmp_ptr_input_layer(this->ptr_array_layers);

    struct AF_unit *const tmp_ptr_input_layer_ptr_first_AF_unit(tmp_ptr_input_layer->ptr_array_AF_units);
    
    size_t const tmp_number_inputs(static_cast<size_t>(tmp_ptr_input_layer->ptr_last_AF_unit - tmp_ptr_input_layer_ptr_first_AF_unit));
    size_t tmp_example_index;

    T_ const *tmp_ptr_array_inputs_it;
    T_ *tmp_ptr_array_values_it;
    
    // Loop through each sample data.
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_values_it = tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values + tmp_example_index * tmp_number_inputs;
        tmp_ptr_array_inputs_it = ptr_matrix_inputs_received[tmp_example_index];
        
        MEMCPY(tmp_ptr_array_values_it,
                       tmp_ptr_array_inputs_it,
                       tmp_number_inputs * sizeof(T_));
    }

    if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING
      &&
      this->pre_training_level == 1_zu)
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Training__Loop(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                    0_zu,
                                                                                                    batch_size_received,
                                                                                                    tmp_number_inputs,
                                                                                                    tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                this->Forward_Pass__Dropout__Bernoulli__Inverted__Loop(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                    0_zu,
                                                                                                    batch_size_received,
                                                                                                    tmp_number_inputs,
                                                                                                    tmp_ptr_input_layer->dropout_values[0u] == 0_T ? 0_T : 1_T / tmp_ptr_input_layer->dropout_values[0u],
                                                                                                    tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                this->Forward_Pass__Dropout__Gaussian__Loop(0_zu,
                                                                                        batch_size_received,
                                                                                        tmp_number_inputs,
                                                                                        tmp_ptr_input_layer->dropout_values[0u],
                                                                                        tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                this->Forward_Pass__Dropout__Uout__Loop(0_zu,
                                                                                batch_size_received,
                                                                                tmp_number_inputs,
                                                                                tmp_ptr_input_layer->dropout_values[0u],
                                                                                tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                    break;
            default: break;
        }
    }
    else
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(0_zu,
                                                                                        batch_size_received,
                                                                                        tmp_number_inputs,
                                                                                        tmp_ptr_input_layer->dropout_values[0u],
                                                                                        tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                    break;
            default: break;
        }
    }
}

void Neural_Network::FF__Assign_Inputs__Pre_Training__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    struct Layer *const tmp_ptr_input_layer(this->ptr_array_layers);

    struct AF_unit *const tmp_ptr_input_layer_ptr_first_AF_unit(tmp_ptr_input_layer->ptr_array_AF_units);
    
    size_t const tmp_number_inputs(static_cast<size_t>(tmp_ptr_input_layer->ptr_last_AF_unit - tmp_ptr_input_layer_ptr_first_AF_unit));

    T_ *tmp_ptr_array_values_it;
    T_ const *tmp_ptr_array_inputs_it,
                  *tmp_ptr_array_values_end;
    
    // Loop through each sample data.
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_inputs_it = ptr_matrix_inputs_received[tmp_example_index__int];
        tmp_ptr_array_values_it = tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values + static_cast<size_t>(tmp_example_index__int) * tmp_number_inputs;
        tmp_ptr_array_values_end = tmp_ptr_array_values_it + tmp_number_inputs;

        for(; tmp_ptr_array_values_it != tmp_ptr_array_values_end; ++tmp_ptr_array_values_it,
                                                                                             ++tmp_ptr_array_inputs_it)
        { *tmp_ptr_array_values_it = *tmp_ptr_array_inputs_it; }
    }
    
    if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING && this->pre_training_level == 1_zu)
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Training__OpenMP(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                          0_zu,
                                                                                                          batch_size_received,
                                                                                                          tmp_number_inputs,
                                                                                                          tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                this->Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                        0_zu,
                                                                                                        batch_size_received,
                                                                                                        tmp_number_inputs,
                                                                                                        tmp_ptr_input_layer->dropout_values[0u] == 0_T ? 0_T : 1_T / tmp_ptr_input_layer->dropout_values[0u],
                                                                                                        tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                this->Forward_Pass__Dropout__Gaussian__OpenMP(0_zu,
                                                                                            batch_size_received,
                                                                                            tmp_number_inputs,
                                                                                            tmp_ptr_input_layer->dropout_values[0u],
                                                                                            tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                this->Forward_Pass__Dropout__Uout__OpenMP(0_zu,
                                                                                     batch_size_received,
                                                                                     tmp_number_inputs,
                                                                                     tmp_ptr_input_layer->dropout_values[0u],
                                                                                     tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                    break;
            default: break;
        }
    }
    else
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(0_zu,
                                                                                           batch_size_received,
                                                                                           tmp_number_inputs,
                                                                                           tmp_ptr_input_layer->dropout_values[0u],
                                                                                           tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                    break;
            default: break;
        }
    }
}

void Neural_Network::RNN__Assign_Inputs__Pre_Training__Loop(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received)
{
    struct Layer *const tmp_ptr_input_layer(this->ptr_array_layers);

    struct AF_unit *const tmp_ptr_input_layer_ptr_first_AF_unit(tmp_ptr_input_layer->ptr_array_AF_units);
    
    size_t const tmp_number_inputs(static_cast<size_t>(tmp_ptr_input_layer->ptr_last_AF_unit - tmp_ptr_input_layer_ptr_first_AF_unit));
    size_t tmp_example_index,
              tmp_time_step_index;

    T_ const *tmp_ptr_array_inputs_it;
    T_ *tmp_ptr_array_values_it;
    
    // Loop through each sample data.
    for(tmp_example_index = 0_zu; tmp_example_index != batch_size_received; ++tmp_example_index)
    {
        tmp_ptr_array_inputs_it = ptr_matrix_inputs_received[tmp_example_index];

        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_ptr_array_values_it = tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values + tmp_example_index * tmp_number_inputs + this->batch_size * tmp_number_inputs * tmp_time_step_index;
            
            MEMCPY(tmp_ptr_array_values_it,
                           tmp_ptr_array_inputs_it,
                           tmp_number_inputs * sizeof(T_));

            tmp_ptr_array_inputs_it += tmp_number_inputs;
        }
    }
    
    if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING
      &&
      this->pre_training_level == 1_zu)
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Bernoulli__Training__Loop(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                        tmp_time_step_index,
                                                                                                        batch_size_received,
                                                                                                        tmp_number_inputs,
                                                                                                        tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Bernoulli__Inverted__Loop(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                      tmp_time_step_index,
                                                                                                      batch_size_received,
                                                                                                      tmp_number_inputs,
                                                                                                      tmp_ptr_input_layer->dropout_values[0u] == 0_T ? 0_T : 1_T / tmp_ptr_input_layer->dropout_values[0u],
                                                                                                      tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Gaussian__Loop(tmp_time_step_index,
                                                                                           batch_size_received,
                                                                                           tmp_number_inputs,
                                                                                           tmp_ptr_input_layer->dropout_values[0u],
                                                                                           tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Uout__Loop(tmp_time_step_index,
                                                                                    batch_size_received,
                                                                                    tmp_number_inputs,
                                                                                    tmp_ptr_input_layer->dropout_values[0u],
                                                                                    tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            default: break;
        }
    }
    else
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(tmp_time_step_index,
                                                                                          batch_size_received,
                                                                                          tmp_number_inputs,
                                                                                          tmp_ptr_input_layer->dropout_values[0u],
                                                                                          tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            default: break;
        }
    }
}

void Neural_Network::RNN__Assign_Inputs__Pre_Training__OpenMP(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size_received));
    int tmp_example_index__int;
    
    struct Layer *const tmp_ptr_input_layer(this->ptr_array_layers);

    struct AF_unit *const tmp_ptr_input_layer_ptr_first_AF_unit(tmp_ptr_input_layer->ptr_array_AF_units);
    
    size_t const tmp_number_inputs(static_cast<size_t>(tmp_ptr_input_layer->ptr_last_AF_unit - tmp_ptr_input_layer_ptr_first_AF_unit));
    size_t tmp_time_step_index;

    T_ *tmp_ptr_array_values_it;
    T_ const *tmp_ptr_array_inputs_it,
                  *tmp_ptr_array_values_end;
    
    // Loop through each sample data.
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_inputs_it = ptr_matrix_inputs_received[tmp_example_index__int];

        for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
        {
            tmp_ptr_array_values_it = tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values + static_cast<size_t>(tmp_example_index__int) * tmp_number_inputs + this->batch_size * tmp_number_inputs * tmp_time_step_index;
            tmp_ptr_array_values_end = tmp_ptr_array_values_it + tmp_number_inputs;

            for(; tmp_ptr_array_values_it != tmp_ptr_array_values_end; ++tmp_ptr_array_values_it,
                                                                                                 ++tmp_ptr_array_inputs_it)
            { *tmp_ptr_array_values_it = *tmp_ptr_array_inputs_it; }
        }
    }
    
    if(this->type_state_propagation == MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING && this->pre_training_level == 1_zu)
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Bernoulli__Training__OpenMP(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                            tmp_time_step_index,
                                                                                                            batch_size_received,
                                                                                                            tmp_number_inputs,
                                                                                                            tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(tmp_ptr_input_layer->ptr_array__mask__dropout__bernoulli,
                                                                                                          tmp_time_step_index,
                                                                                                          batch_size_received,
                                                                                                          tmp_number_inputs,
                                                                                                          tmp_ptr_input_layer->dropout_values[0u] == 0_T ? 0_T : 1_T / tmp_ptr_input_layer->dropout_values[0u],
                                                                                                          tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Gaussian__OpenMP(tmp_time_step_index,
                                                                                               batch_size_received,
                                                                                               tmp_number_inputs,
                                                                                               tmp_ptr_input_layer->dropout_values[0u],
                                                                                               tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Uout__OpenMP(tmp_time_step_index,
                                                                                        batch_size_received,
                                                                                        tmp_number_inputs,
                                                                                        tmp_ptr_input_layer->dropout_values[0u],
                                                                                        tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            default: break;
        }
    }
    else
    {
        switch(tmp_ptr_input_layer->type_dropout)
        {
            case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
                for(tmp_time_step_index = 0_zu; tmp_time_step_index != this->number_recurrent_depth; ++tmp_time_step_index)
                {
                    this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(tmp_time_step_index,
                                                                                              batch_size_received,
                                                                                              tmp_number_inputs,
                                                                                              tmp_ptr_input_layer->dropout_values[0u],
                                                                                              tmp_ptr_input_layer_ptr_first_AF_unit->ptr_array_values);
                }
                    break;
            default: break;
        }
    }
}