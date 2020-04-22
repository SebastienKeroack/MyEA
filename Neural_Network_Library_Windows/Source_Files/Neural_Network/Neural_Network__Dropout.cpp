#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

bool Layer::Use__Coded_Dropout(void) const { return(this->use_coded_dropout); }

bool Neural_Network::Set__Dropout(size_t const index_layer_received,
                                                    enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT const type_layer_dropout_received,
                                                    T_ const value_dropout_received[],
                                                    bool const scale_weights_received)
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

    return(this->Set__Dropout(this->ptr_array_layers + index_layer_received,
                                          type_layer_dropout_received,
                                          value_dropout_received,
                                          scale_weights_received));
}

bool Neural_Network::Set__Dropout(struct Layer *const ptr_layer_received,
                                                    enum MyEA::Common::ENUM_TYPE_LAYER_DROPOUT const type_layer_dropout_received,
                                                    T_ const value_dropout_received[],
                                                    bool const scale_weights_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_received\" is a nullptr. At line %d." NEW_LINE,
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
    
    if(this->type_network == MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_AUTOENCODER)
    {
        if(ptr_layer_received == this->ptr_last_layer - (this->total_layers - 3_zu) / 2_zu + 2_zu)
        {
            PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is a coded layer. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
        else if(ptr_layer_received >= this->ptr_last_layer - (this->total_layers - 3_zu) / 2_zu + 1_zu)
        {
            PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is a decoded layer. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }

    switch(type_layer_dropout_received)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE: return(this->Set__Dropout_None(ptr_layer_received));
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA: return(this->Set__Dropout_Alpha(ptr_layer_received, value_dropout_received[0u]));
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI:
            return(this->Set__Dropout_Bernoulli(ptr_layer_received,
                                                                 value_dropout_received[0u],
                                                                 scale_weights_received));
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED: return(this->Set__Dropout_Bernoulli_Inverted(ptr_layer_received, value_dropout_received[0u]));
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN: return(this->Set__Dropout_Gaussian(ptr_layer_received, value_dropout_received[0u]));
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP: return(this->Set__Dropout_ShakeDrop(ptr_layer_received, value_dropout_received[0u]));
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT: return(this->Set__Dropout_Uout(ptr_layer_received, value_dropout_received[0u]));
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT:
            return(this->Set__Dropout_Zoneout(ptr_layer_received,
                                                                value_dropout_received[0u],
                                                                value_dropout_received[1u]));
        default: return(false);
    }
}

bool Neural_Network::Set__Dropout_None(struct Layer *const ptr_layer_received)
{
    switch(ptr_layer_received->type_dropout)
    {
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE: return(true);
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA: return(this->Set__Dropout_Alpha(ptr_layer_received, 0_T));
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI: return(this->Set__Dropout_Bernoulli(ptr_layer_received, 1_T));
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED: return(this->Set__Dropout_Bernoulli_Inverted(ptr_layer_received, 1_T));
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN: return(this->Set__Dropout_Gaussian(ptr_layer_received, 0_T));
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP: return(this->Set__Dropout_ShakeDrop(ptr_layer_received, 0_T));
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT: return(this->Set__Dropout_Uout(ptr_layer_received, 0_T));
        case MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT:
            return(this->Set__Dropout_Zoneout(ptr_layer_received,
                                                                0_T,
                                                                0_T));
        default:
            PRINT_FORMAT("%s: %s: ERROR: Dropout layer type (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_received->type_dropout,
                                     MyEA::Common::ENUM_TYPE_LAYER_DROPOUT_NAMES[ptr_layer_received->type_dropout].c_str(),
                                     __LINE__);
                return(false);
    }
}

bool Neural_Network::Set__Dropout_Alpha(struct Layer *const ptr_layer_received, T_ const dropout_probability_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_received\" is a nullptr. At line %d." NEW_LINE,
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
    else if(dropout_probability_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of dropout (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(dropout_probability_received),
                                 __LINE__);
        return(false);
    }
    else if(dropout_probability_received > 1_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of retention (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(dropout_probability_received),
                                 __LINE__);

        return(false);
    }
    
    if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA || ptr_layer_received->dropout_values[0u] != dropout_probability_received)
    {
        if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout_None(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }
        
        T_ const tmp_keep_probability(1_T - dropout_probability_received);

        ptr_layer_received->dropout_values[0u] = tmp_keep_probability;

        if(dropout_probability_received != 0_T && ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE)
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA;
            
            T_ const tmp_alpha(-SELU_Scale * SELU_Alpha);

            ptr_layer_received->dropout_values[1u] = pow(tmp_keep_probability + pow(tmp_alpha, 2_T) * tmp_keep_probability * dropout_probability_received, -0.5_T);
            ptr_layer_received->dropout_values[2u] = -ptr_layer_received->dropout_values[1u] * dropout_probability_received * tmp_alpha;

            if(++this->total_dropout_alpha_layers == 1_zu)
            {
                if(this->Allocate__Generator__Dropout_Bernoulli() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Generator__Dropout_Bernoulli()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

                    --this->total_dropout_alpha_layers;

                    return(false);
                }
                else if(this->Allocate__Neuron__Mask_Dropout_Bernoulli() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Neuron__Mask_Dropout_Bernoulli()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

                    --this->total_dropout_alpha_layers;

                    return(false);
                }
            }
        }
        else if(dropout_probability_received == 0_T && ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ALPHA)
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;
            
            ptr_layer_received->dropout_values[1u] = 0_T;
            ptr_layer_received->dropout_values[2u] = 0_T;

            if(this->total_dropout_alpha_layers != 0_zu
              &&
              --this->total_dropout_alpha_layers == 0_zu
              &&
                    (this->Use__Dropout__Bernoulli() == false
                     &&
                     this->Use__Dropout__Bernoulli__Inverted() == false
                     &&
                     this->Use__Dropout__Alpha() == false))
            {
                this->Deallocate__Generator__Dropout_Bernoulli();

                this->Deallocate__Neuron__Mask_Dropout_Bernoulli();
            }
        }
    }

    return(true);
}

bool Neural_Network::Set__Dropout_Bernoulli(struct Layer *const ptr_layer_received,
                                                                  T_ const retention_probability_received,
                                                                  bool const scale_weights_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_received\" is a nullptr. At line %d." NEW_LINE,
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
    else if(retention_probability_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of retention (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(retention_probability_received),
                                 __LINE__);

        return(false);
    }
    else if(retention_probability_received > 1_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of retention (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(retention_probability_received),
                                 __LINE__);

        return(false);
    }

    if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI || ptr_layer_received->dropout_values[0u] != retention_probability_received)
    {
        if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout_None(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }
        
        if(scale_weights_received
          &&
          ptr_layer_received != this->ptr_array_layers
          &&
          ptr_layer_received->dropout_values[0u] != retention_probability_received) { this->Scale_Weight__Dropout(ptr_layer_received->dropout_values[0u] / retention_probability_received, ptr_layer_received); }
        
        ptr_layer_received->dropout_values[0u] = retention_probability_received;

        if(retention_probability_received != 1_T && ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE)
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI;

            if(++this->total_dropout_bernoulli_layers == 1_zu)
            {
                if(this->Allocate__Generator__Dropout_Bernoulli() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Generator__Dropout_Bernoulli()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

                    --this->total_dropout_bernoulli_layers;

                    return(false);
                }
                else if(this->Allocate__Neuron__Mask_Dropout_Bernoulli() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Neuron__Mask_Dropout_Bernoulli()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

                    --this->total_dropout_bernoulli_layers;

                    return(false);
                }
            }
        }
        else if(retention_probability_received == 1_T && ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI)
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;
            
            if(this->total_dropout_bernoulli_layers != 0_zu
              &&
              --this->total_dropout_bernoulli_layers == 0_zu
              &&
                    (this->Use__Dropout__Bernoulli() == false
                     &&
                     this->Use__Dropout__Bernoulli__Inverted() == false
                     &&
                     this->Use__Dropout__Alpha() == false))
            {
                this->Deallocate__Generator__Dropout_Bernoulli();

                this->Deallocate__Neuron__Mask_Dropout_Bernoulli();
            }
        }
    }

    return(true);
}

bool Neural_Network::Set__Dropout_Bernoulli_Inverted(struct Layer *const ptr_layer_received, T_ const retention_probability_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_received\" is a nullptr. At line %d." NEW_LINE,
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
    else if(retention_probability_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of retention (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(retention_probability_received),
                                 __LINE__);

        return(false);
    }
    else if(retention_probability_received > 1_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of retention (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(retention_probability_received),
                                 __LINE__);

        return(false);
    }

    if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED || ptr_layer_received->dropout_values[0u] != retention_probability_received)
    {
        if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout_None(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }

        ptr_layer_received->dropout_values[0u] = retention_probability_received;

        if(retention_probability_received != 1_T && ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE)
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED;

            if(++this->total_dropout_bernoulli_inverted_layers == 1_zu)
            {
                if(this->Allocate__Generator__Dropout_Bernoulli() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Generator__Dropout_Bernoulli()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

                    --this->total_dropout_bernoulli_inverted_layers;

                    return(false);
                }
                else if(this->Allocate__Neuron__Mask_Dropout_Bernoulli() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Neuron__Mask_Dropout_Bernoulli()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

                    --this->total_dropout_bernoulli_inverted_layers;

                    return(false);
                }
            }
        }
        else if(retention_probability_received == 1_T && ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_BERNOULLI_INVERTED)
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

            if(this->total_dropout_bernoulli_inverted_layers != 0_zu
              &&
              --this->total_dropout_bernoulli_inverted_layers == 0_zu
              &&
                    (this->Use__Dropout__Bernoulli() == false
                     &&
                     this->Use__Dropout__Bernoulli__Inverted() == false
                     &&
                     this->Use__Dropout__Alpha() == false))
            {
                this->Deallocate__Generator__Dropout_Bernoulli();

                this->Deallocate__Neuron__Mask_Dropout_Bernoulli();
            }
        }
    }

    return(true);
}

bool Neural_Network::Set__Dropout_Gaussian(struct Layer *const ptr_layer_received, T_ const dropout_probability_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_received\" is a nullptr. At line %d." NEW_LINE,
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
    else if(dropout_probability_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of dropout (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(dropout_probability_received),
                                 __LINE__);
        return(false);
    }
    else if(dropout_probability_received > 1_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of retention (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(dropout_probability_received),
                                 __LINE__);

        return(false);
    }
    
    if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN || ptr_layer_received->dropout_values[0u] != dropout_probability_received)
    {
        if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout_None(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }

        //ptr_layer_received->dropout_values[0u] = dropout_probability_received == 1_T ? 0_T : static_cast<T_>(pow(sqrt(static_cast<double>(dropout_probability_received) / (1.0 - static_cast<double>(dropout_probability_received))), 2.0));
        ptr_layer_received->dropout_values[0u] = dropout_probability_received == 1_T ? 0_T : static_cast<T_>(static_cast<double>(dropout_probability_received) / (1.0 - static_cast<double>(dropout_probability_received)));

        if(dropout_probability_received != 0_T && ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE)
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN;

            if(++this->total_dropout_gaussian_layers == 1_zu)
            {
                if(this->Allocate__Generator__Dropout_Gaussian() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Generator__Dropout_Gaussian()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

                    --this->total_dropout_gaussian_layers;

                    return(false);
                }
            }
        }
        else if(dropout_probability_received == 0_T && ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_GAUSSIAN)
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

            if(this->total_dropout_gaussian_layers != 0_zu && --this->total_dropout_gaussian_layers == 0_zu) { this->Deallocate__Generator__Dropout_Gaussian(); }
        }
    }

    return(true);
}

bool Neural_Network::Set__Dropout_ShakeDrop(struct Layer *const ptr_layer_received, T_ const dropout_probability_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(ptr_layer_received->type_layer != MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL)
    {
        PRINT_FORMAT("%s: %s: ERROR: Layer received as argument is not a residual layer. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(dropout_probability_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of dropout (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(dropout_probability_received),
                                 __LINE__);
        return(false);
    }
    else if(dropout_probability_received > 1_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of retention (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(dropout_probability_received),
                                 __LINE__);

        return(false);
    }
    
    if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP || ptr_layer_received->dropout_values[0u] != dropout_probability_received)
    {
        if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout_None(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }

        // The paper recommends linear decay rule to determine pl (pl = layer dropout probability, pL = initial dropout probability).
        // l = block index, L = total block.
        // pl = 1 - ( (l / L * (1 - pL) )
        ptr_layer_received->dropout_values[0u] = dropout_probability_received;

        if(dropout_probability_received != 0_T && ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE)
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP;

            if(++this->total_dropout_shakedrop_layers == 1_zu)
            {
                if(this->Allocate__Generator__Dropout_ShakeDrop() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Generator__Dropout_ShakeDrop()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

                    --this->total_dropout_shakedrop_layers;

                    return(false);
                }
                else if(this->Allocate__Layer__Mask__Dropout__ShakeDrop() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Layer__Mask__Dropout__ShakeDrop()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

                    --this->total_dropout_shakedrop_layers;

                    return(false);
                }
            }
        }
        else if(dropout_probability_received == 0_T && ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_SHAKEDROP)
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

            if(this->total_dropout_shakedrop_layers != 0_zu && --this->total_dropout_shakedrop_layers == 0_zu)
            {
                this->Deallocate__Generator__Dropout_ShakeDrop();
                
                this->Deallocate__Layer__Mask_Dropout_ShakeDrop();
            }
        }
    }

    return(true);
}

bool Neural_Network::Set__Dropout_Uout(struct Layer *const ptr_layer_received, T_ const dropout_probability_received)
{
    if(ptr_layer_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_layer_received\" is a nullptr. At line %d." NEW_LINE,
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
    else if(dropout_probability_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of dropout (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(dropout_probability_received),
                                 __LINE__);
        return(false);
    }
    else if(dropout_probability_received > 1_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of retention (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(dropout_probability_received),
                                 __LINE__);

        return(false);
    }
    
    if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT || ptr_layer_received->dropout_values[0u] != dropout_probability_received)
    {
        if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout_None(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }

        ptr_layer_received->dropout_values[0u] = dropout_probability_received;

        if(dropout_probability_received != 0_T && ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE)
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT;

            if(++this->total_dropout_uout_layers == 1_zu)
            {
                if(this->Allocate__Generator__Dropout_Uout() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Generator__Dropout_Uout()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

                    --this->total_dropout_uout_layers;

                    return(false);
                }
            }
        }
        else if(dropout_probability_received == 0_T && ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_UOUT)
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

            if(this->total_dropout_uout_layers != 0_zu && --this->total_dropout_uout_layers == 0_zu) { this->Deallocate__Generator__Dropout_Uout(); }
        }
    }

    return(true);
}

bool Neural_Network::Set__Dropout_Zoneout(struct Layer *const ptr_layer_received,
                                                                  T_ const zoneout_cell_received,
                                                                  T_ const zoneout_hidden_received)
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
    else if(zoneout_cell_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of zoneout cell (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(zoneout_cell_received),
                                 __LINE__);
        return(false);
    }
    else if(zoneout_cell_received > 1_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of zoneout cell (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(zoneout_cell_received),
                                 __LINE__);

        return(false);
    }
    else if(zoneout_hidden_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of zoneout hidden (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(zoneout_hidden_received),
                                 __LINE__);
        return(false);
    }
    else if(zoneout_hidden_received > 1_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Probability of zoneout hidden (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(zoneout_hidden_received),
                                 __LINE__);

        return(false);
    }
    
    if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT
       ||
       ptr_layer_received->dropout_values[0u] != zoneout_cell_received
       ||
       ptr_layer_received->dropout_values[1u] != zoneout_hidden_received)
    {
        if(ptr_layer_received->type_dropout != MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Set__Dropout_None(ptr)\" function. At line %d." NEW_LINE,
                                         MyEA::Time::Date_Time_Now().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }
        }

        ptr_layer_received->dropout_values[0u] = zoneout_cell_received;
        ptr_layer_received->dropout_values[1u] = zoneout_hidden_received;

        if(ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE
          &&
              (zoneout_cell_received != 0_T
              ||
              zoneout_hidden_received != 0_T))
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT;

            if(++this->total_dropout_zoneout_layers == 1_zu)
            {
                if(this->Allocate__Generator__Dropout_Zoneout() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Generator__Dropout_Zoneout()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

                    --this->total_dropout_zoneout_layers;

                    return(false);
                }
                else if(this->Allocate__Block_Unit__Mask_Dropout_Zoneout() == false)
                {
                    PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Block_Unit__Mask_Dropout_Zoneout()\" function. At line %d." NEW_LINE,
                                             MyEA::Time::Date_Time_Now().c_str(),
                                             __FUNCTION__,
                                             __LINE__);
                    
                    ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;

                    --this->total_dropout_zoneout_layers;

                    return(false);
                }
            }
        }
        else if(zoneout_cell_received == 0_T
                 &&
                 zoneout_hidden_received == 0_T
                 &&
                 ptr_layer_received->type_dropout == MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_ZONEOUT)
        {
            ptr_layer_received->type_dropout = MyEA::Common::ENUM_TYPE_LAYER_DROPOUT::TYPE_LAYER_DROPOUT_NONE;
            
            if(this->total_dropout_zoneout_layers != 0_zu && --this->total_dropout_zoneout_layers == 0_zu)
            {
                this->Deallocate__Generator__Dropout_Zoneout();

                this->Deallocate__Cell_Unit__Mask_Dropout_Zoneout();
            }
        }
    }

    return(true);
}

void Neural_Network::Scale_Weight__Dropout(T_ const scale_factor_received, struct Layer const *const ptr_layer_it_received)
{
    switch(ptr_layer_it_received->type_layer)
    {
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_RECURRENT: this->Scale_Weight__FC__Recurrent__Dropout(scale_factor_received, ptr_layer_it_received);
        case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED: this->Scale_Weight__FC__Forward__Dropout(scale_factor_received, ptr_layer_it_received); break;
        default:
            PRINT_FORMAT("%s: %s: ERROR: Type layer (%u | %s) is not managed in the switch. At line %d." NEW_LINE,
                                     MyEA::Time::Date_Time_Now().c_str(),
                                     __FUNCTION__,
                                     ptr_layer_it_received->type_layer,
                                     MyEA::Common::ENUM_TYPE_LAYER_NAME[ptr_layer_it_received->type_layer].c_str(),
                                     __LINE__);
                return;
    }
}

void Neural_Network::Scale_Weight__FC__Forward__Dropout(T_ const scale_factor_received, struct Layer const *const ptr_layer_it_received)
{
    struct Neuron_unit const *const tmp_ptr_layer_ptr_last_neuron_unit(ptr_layer_it_received->ptr_last_neuron_unit - 1), // Get last neuron unit.
                                 *const tmp_ptr_layer_ptr_first_neuron_unit(ptr_layer_it_received->ptr_array_neuron_units);
    
    T_ const *const tmp_ptr_array_parameters_end(this->ptr_array_parameters + *tmp_ptr_layer_ptr_last_neuron_unit->ptr_last_connection_index);
    T_ *tmp_ptr_array_parameters_it(this->ptr_array_parameters + *tmp_ptr_layer_ptr_first_neuron_unit->ptr_first_connection_index);
    
    for(; tmp_ptr_array_parameters_it != tmp_ptr_array_parameters_end; ++tmp_ptr_array_parameters_it) { *tmp_ptr_array_parameters_it *= scale_factor_received; }
}

void Neural_Network::Scale_Weight__FC__Recurrent__Dropout(T_ const scale_factor_received, struct Layer const *const ptr_layer_it_received)
{
    struct AF_Ind_recurrent_unit const *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);

    size_t const tmp_number_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_first_AF_Ind_recurrent_unit));
    
    T_ *tmp_ptr_array_parameters_it(this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index);
    T_ const *const tmp_ptr_array_parameters_end(tmp_ptr_array_parameters_it + tmp_number_units);
    
    for(; tmp_ptr_array_parameters_it != tmp_ptr_array_parameters_end; ++tmp_ptr_array_parameters_it) { *tmp_ptr_array_parameters_it *= scale_factor_received; }
}
