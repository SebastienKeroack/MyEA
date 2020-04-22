#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

bool Layer::Use__Regularization__Constraint_Recurrent_Weight(void) const { return(this->constraint_recurrent_weight_lower_bound != 0_T || this->constraint_recurrent_weight_upper_bound != 0_T); }

bool Neural_Network::Check__Use__Regularization__Constraint_Recurrent_Weight__Default(size_t const index_layer_received) const { return(this->Check__Use__Regularization__Constraint_Recurrent_Weight__Default(this->ptr_array_layers + index_layer_received)); }

bool Neural_Network::Check__Use__Regularization__Constraint_Recurrent_Weight__Default(struct Layer *const ptr_layer_received) const
{
    std::pair<T_, T_> const tmp_pair_constraint_bound(this->Compute__Regularization__Constraint_Recurrent_Weight__Default(ptr_layer_received));
    T_ const tmp_constraint_recurrent_weight_lower_bound(tmp_pair_constraint_bound.first),
                 tmp_constraint_recurrent_weight_upper_bound(tmp_pair_constraint_bound.second);
    
    return(ptr_layer_received->constraint_recurrent_weight_lower_bound <= tmp_constraint_recurrent_weight_lower_bound + 1e-8
             &&
             ptr_layer_received->constraint_recurrent_weight_lower_bound >= tmp_constraint_recurrent_weight_lower_bound - 1e-8
             &&
             ptr_layer_received->constraint_recurrent_weight_upper_bound <= tmp_constraint_recurrent_weight_upper_bound + 1e-8
             &&
             ptr_layer_received->constraint_recurrent_weight_upper_bound >= tmp_constraint_recurrent_weight_upper_bound - 1e-8);
}

std::pair<T_, T_> Neural_Network::Compute__Regularization__Constraint_Recurrent_Weight__Default(size_t const index_layer_received) const { return(this->Compute__Regularization__Constraint_Recurrent_Weight__Default(this->ptr_array_layers + index_layer_received)); }

std::pair<T_, T_> Neural_Network::Compute__Regularization__Constraint_Recurrent_Weight__Default(struct Layer *const ptr_layer_received) const
{
    T_ const tmp_MAG(MyEA::Math::Clip<T_>(this->clip_gradient, 2_T, 10_T));
    T_ tmp_constraint_recurrent_weight_lower_bound,
         tmp_constraint_recurrent_weight_upper_bound;
    
    switch(ptr_layer_received->type_activation)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_NONE:
            tmp_constraint_recurrent_weight_lower_bound = -1_T;
            tmp_constraint_recurrent_weight_upper_bound = 1_T;
                break;
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC:
            tmp_constraint_recurrent_weight_lower_bound = -pow(tmp_MAG / pow(0.9_T, static_cast<T_>(this->number_recurrent_depth) / 10_T), 1_T / static_cast<T_>(this->number_recurrent_depth));
            tmp_constraint_recurrent_weight_upper_bound = pow(tmp_MAG / pow(0.9_T, static_cast<T_>(this->number_recurrent_depth) / 10_T), 1_T / static_cast<T_>(this->number_recurrent_depth));
                break;
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_ASYMMETRIC:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SOFTMAX:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_RECTIFIER:
        case MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SELF_NORMALIZATION:
            tmp_constraint_recurrent_weight_lower_bound = -pow(tmp_MAG, 1_T / static_cast<T_>(this->number_recurrent_depth));
            tmp_constraint_recurrent_weight_upper_bound = pow(tmp_MAG, 1_T / static_cast<T_>(this->number_recurrent_depth));
                break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Layer activation type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_received->type_activation,
                                     MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION_NAME[ptr_layer_received->type_activation].c_str(),
                                     __LINE__);
                return(std::make_pair(0_T, 0_T));
    }

    return(std::make_pair(tmp_constraint_recurrent_weight_lower_bound, tmp_constraint_recurrent_weight_upper_bound));
}

bool Neural_Network::Set__Regularization__Constraint_Recurrent_Weight__Default(size_t const index_layer_received)
{
    if(index_layer_received >= this->total_layers)
    {
        PRINT_FORMAT("%s: ERROR: Layer received (%zu) overflow the number of layers (%zu) in the neural network." NEW_LINE,
                                 __FUNCTION__,
                                 index_layer_received,
                                 this->total_layers);

        return(false);
    }
    else if(this->ptr_array_layers == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_array_layers\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(this->Set__Regularization__Constraint_Recurrent_Weight__Default(this->ptr_array_layers + index_layer_received));
}

bool Neural_Network::Set__Regularization__Constraint_Recurrent_Weight__Default(struct Layer *const ptr_layer_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the input layer. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the output layer. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    switch(ptr_layer_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Can not constraining the recurrent weight in the layer %zu with (%u | %s) as the type layer. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     static_cast<size_t>(ptr_layer_received - this->ptr_array_layers),
                                     ptr_layer_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_received->type_layer].c_str(),
                                     __LINE__);
                return(false);
    }

    std::pair<T_, T_> const tmp_pair_constraint_bound(this->Compute__Regularization__Constraint_Recurrent_Weight__Default(ptr_layer_received));
    T_ const tmp_constraint_recurrent_weight_lower_bound(tmp_pair_constraint_bound.first),
                 tmp_constraint_recurrent_weight_upper_bound(tmp_pair_constraint_bound.second);
    
    return(this->Set__Regularization__Constraint_Recurrent_Weight(ptr_layer_received,
                                                                                                  tmp_constraint_recurrent_weight_lower_bound,
                                                                                                  tmp_constraint_recurrent_weight_upper_bound));
}

bool Neural_Network::Set__Regularization__Constraint_Recurrent_Weight(size_t const index_layer_received,
                                                                                                            T_ const constraint_recurrent_weight_lower_bound_received,
                                                                                                            T_ const constraint_recurrent_weight_upper_bound_received)
{
    if(index_layer_received >= this->total_layers)
    {
        PRINT_FORMAT("%s: ERROR: Layer received (%zu) overflow the number of layers (%zu) in the neural network." NEW_LINE,
                                 __FUNCTION__,
                                 index_layer_received,
                                 this->total_layers);

        return(false);
    }
    else if(this->ptr_array_layers == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_array_layers\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(this->Set__Regularization__Constraint_Recurrent_Weight(this->ptr_array_layers + index_layer_received,
                                                                                                  constraint_recurrent_weight_lower_bound_received,
                                                                                                  constraint_recurrent_weight_upper_bound_received));
}

bool Neural_Network::Set__Regularization__Constraint_Recurrent_Weight(struct Layer *const ptr_layer_received,
                                                                                                            T_ const constraint_recurrent_weight_lower_bound_received,
                                                                                                            T_ const constraint_recurrent_weight_upper_bound_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the input layer. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is the output layer. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(constraint_recurrent_weight_lower_bound_received > constraint_recurrent_weight_upper_bound_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Lower bound (%f) can not be greater than upper bound (%f). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(constraint_recurrent_weight_lower_bound_received),
                                 Cast_T(constraint_recurrent_weight_upper_bound_received),
                                 __LINE__);
        
        return(false);
    }
    else if(ptr_layer_received->constraint_recurrent_weight_lower_bound == constraint_recurrent_weight_lower_bound_received
             &&
             ptr_layer_received->constraint_recurrent_weight_upper_bound == constraint_recurrent_weight_upper_bound_received) { return(true); }
    
    switch(ptr_layer_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM: break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Can not constraining the recurrent weight in the layer %zu with (%u | %s) as the type layer. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     static_cast<size_t>(ptr_layer_received - this->ptr_array_layers),
                                     ptr_layer_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_received->type_layer].c_str(),
                                     __LINE__);
                return(false);
    }
    
    if(ptr_layer_received->constraint_recurrent_weight_lower_bound == 0_T
      &&
      ptr_layer_received->constraint_recurrent_weight_upper_bound == 0_T
      &&
      (constraint_recurrent_weight_lower_bound_received != 0_T
       ||
       constraint_recurrent_weight_upper_bound_received != 0_T)) { ++this->total_constraint_recurrent_weight_layers; }
    else if((ptr_layer_received->constraint_recurrent_weight_lower_bound != 0_T
               ||
               ptr_layer_received->constraint_recurrent_weight_upper_bound != 0_T)
              &&
              constraint_recurrent_weight_lower_bound_received == 0_T
              &&
              constraint_recurrent_weight_upper_bound_received == 0_T) { --this->total_constraint_recurrent_weight_layers; }

    ptr_layer_received->constraint_recurrent_weight_lower_bound = constraint_recurrent_weight_lower_bound_received;
    ptr_layer_received->constraint_recurrent_weight_upper_bound = constraint_recurrent_weight_upper_bound_received;

    // Mirror layer.
    if(this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER
      &&
      ptr_layer_received < this->Get__End_Layer__Active() - 1 // Get last active layer.
      &&
      this->Set__Regularization__Constraint_Recurrent_Weight(this->ptr_last_layer - static_cast<size_t>(ptr_layer_received - this->ptr_array_layers) - 1,
                                                                                          constraint_recurrent_weight_lower_bound_received,
                                                                                          constraint_recurrent_weight_upper_bound_received))
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Regularization__Constraint_Recurrent_Weight(ptr, %f, %f)\" function. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(constraint_recurrent_weight_lower_bound_received),
                                 Cast_T(constraint_recurrent_weight_upper_bound_received),
                                 __LINE__);

        return(false);
    }
    // |END| Mirror layer. |END|

    return(true);
}

void Neural_Network::Update_Weight_Regularization__Constraint_Recurrent_Weight(size_t const start_index_received, size_t const end_index_received)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer - 1);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + 1);
    
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        if(*tmp_ptr_layer_it->ptr_first_connection_index < start_index_received) { continue; }
        else if(*tmp_ptr_layer_it->ptr_last_connection_index > end_index_received) { break; }

        if(tmp_ptr_layer_it->Use__Regularization__Constraint_Recurrent_Weight())
        {
            switch(tmp_ptr_layer_it->type_layer)
            {
                case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Update_Weight_Regularization__Constraint_Recurrent_Weight__FC_Ind_RNN(tmp_ptr_layer_it); break;
                default:
                    PRINT_FORMAT("%s: %s: ERROR: Type layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             tmp_ptr_layer_it->type_layer,
                                             MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                             __LINE__);
                        return;
            }
        }
    }
}

void Neural_Network::Update_Weight_Regularization__Constraint_Recurrent_Weight__FC_Ind_RNN(struct Layer const *const ptr_layer_it_received)
{
    struct AF_Ind_recurrent_unit const *const tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);

    size_t const tmp_number_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit));
    
    T_ *tmp_ptr_parameter_it(this->ptr_array_parameters + *tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index);
    T_ const *tmp_ptr_last_parameter(tmp_ptr_parameter_it + tmp_number_units),
                  tmp_constraint_recurrent_weight_lower_bound(ptr_layer_it_received->constraint_recurrent_weight_lower_bound),
                  tmp_constraint_recurrent_weight_upper_bound(ptr_layer_it_received->constraint_recurrent_weight_upper_bound);

    for(; tmp_ptr_parameter_it != tmp_ptr_last_parameter; ++tmp_ptr_parameter_it)
    {
        *tmp_ptr_parameter_it = MyEA::Math::Clip<T_>(*tmp_ptr_parameter_it,
                                                                              tmp_constraint_recurrent_weight_lower_bound,
                                                                              tmp_constraint_recurrent_weight_upper_bound);
    }
}
