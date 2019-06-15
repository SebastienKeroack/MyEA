#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

bool Neural_Network::Initialize__LSUV(size_t const maximum_number_trials_received,
                                                        size_t const maximum_batch_size_received,
                                                        T_ const bias_received,
                                                        T_ const variance_target_received,
                                                        T_ const variance_tolerance_received)
{
    if(maximum_batch_size_received == 0_zu)
    {
        PRINT_FORMAT("%s: %s: ERROR: Maximum batch size can not be equal to zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(variance_target_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Variance target (%f) can not be less than zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(variance_target_received),
                                 __LINE__);

        return(false);
    }
    else if(variance_tolerance_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Variance tolerance (%f) can not be less than zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(variance_tolerance_received),
                                 __LINE__);

        return(false);
    }

    this->_LSUV_Parameters.maximum_number_trials = maximum_number_trials_received;
    this->_LSUV_Parameters.maximum_batch_size = maximum_batch_size_received;
    this->_LSUV_Parameters.initial_bias = bias_received;
    this->_LSUV_Parameters.variance_target = variance_target_received;
    this->_LSUV_Parameters.variance_tolerance = variance_tolerance_received;

    this->_initialized__weight = false;
    this->_type_weights_initializer = MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_LSUV;

    return(true);
}

bool Neural_Network::Initialization__LSUV(class Dataset<T_> const *const ptr_Dataset_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_Dataset_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_Dataset_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif

    // Pre-initialize network with orthonormal matrices as in Saxe et al. (2014).
    this->Initialization__Orthogonal(true, this->_LSUV_Parameters.initial_bias);

    size_t const tmp_number_examples(MyEA::Math::Minimum<size_t>(this->_LSUV_Parameters.maximum_batch_size, ptr_Dataset_received->Dataset<T_>::Get__Number_Examples()));
    
    if(this->Update__Batch_Size(tmp_number_examples) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Batch_Size(%zu)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 tmp_number_examples,
                                 __LINE__);

        return(false);
    }
    
    if(this->use_OpenMP && this->is_OpenMP_initialized)
    {
        if(this->Update__Thread_Size(tmp_number_examples) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Update__Thread_Size(%zu)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     tmp_number_examples,
                                     __LINE__);

            return(false);
        }
        
        omp_set_num_threads(static_cast<int>(this->number_threads));
        
        if(this->Initialization__LSUV__OpenMP(ptr_Dataset_received) == false)
        {
            PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialization__LSUV__OpenMP(ptr)\" function. At line %d." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     __FUNCTION__,
                                     __LINE__);

            return(false);
        }
    }
    else if(this->Initialization__LSUV__Loop(ptr_Dataset_received) == false)
    {
        PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Initialization__LSUV__Loop(ptr)\" function. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    // Independently recurrent neural network.
    if(this->number_recurrent_depth > 1_zu
      &&
      this->number_time_delays + 1_zu == this->number_recurrent_depth)
    { this->Initialize__Uniform__AF_Ind_Recurrent__Long_Term_Memory(); }

    if(this->ptr_array_derivatives_parameters != nullptr) { this->Clear_Training_Arrays(); }

    if(this->Use__Normalization()) { this->Clear__Parameter__Normalized_Unit(); }

    this->_initialized__weight = true;

    return(true);
}

// TODO: Intermediate propagation.
bool Neural_Network::Initialization__LSUV__Loop(class Dataset<T_> const *const ptr_Dataset_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_Dataset_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_Dataset_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif
    
    size_t const tmp_maximum_number_trials(this->_LSUV_Parameters.maximum_number_trials);
    size_t tmp_layer_index,
              tmp_trial_index;
    
    T_ const tmp_epsilon(this->_LSUV_Parameters.epsilon),
                 tmp_variance_target(MyEA::Math::Maximum<T_>(this->_LSUV_Parameters.variance_target, tmp_epsilon)),
                 tmp_variance_tolerance(this->_LSUV_Parameters.variance_tolerance);
    T_ tmp_variance;

    struct Layer const *tmp_ptr_layer_initialization;
    
    auto tmp_Is__Valid([](struct Layer const *const ptr_layer_received) -> bool
    {
        return(ptr_layer_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_CONVOLUTION
                 ||
                 ptr_layer_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED);
        
    });

    auto tmp_Get__Variance__Loop([&](size_t const layer_index_received) -> T_
    {
        size_t const tmp_number_examples(MyEA::Math::Minimum<size_t>(this->_LSUV_Parameters.maximum_batch_size, ptr_Dataset_received->Dataset<T_>::Get__Number_Examples())),
                           tmp_maximum_batch_size(this->batch_size),
                           tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
        size_t tmp_batch_size,
                  tmp_batch_index;
        
        T_ tmp_variance(0_T);

        for(tmp_batch_index = 0_zu; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
        {
            tmp_batch_size = tmp_batch_index + 1_zu != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - tmp_batch_index * tmp_maximum_batch_size;

            this->Forward_Pass(tmp_batch_size,
                                            ptr_Dataset_received->Dataset<T_>::Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size,
                                            0ll,
                                            static_cast<long long int>(layer_index_received) + 1ll);

            tmp_variance += this->Get__Outputs__Variance(tmp_ptr_layer_initialization, tmp_batch_size);
        }
        
        return(tmp_variance <= 0_T ? tmp_epsilon : tmp_variance);
    });
    
    // TODO: Make function "Scale__Weight__Loop" global.
    auto tmp_Scale__Weight__Loop([](T_ *ptr_ptr_array_weights_received,
                                                       T_ const *const ptr_last_weight_received,
                                                       T_ const scale_received) -> void
    {
        for(; ptr_ptr_array_weights_received != ptr_last_weight_received; ++ptr_ptr_array_weights_received)
        {
            *ptr_ptr_array_weights_received *= scale_received;
        }
    });

    for(tmp_layer_index = 1_zu; tmp_layer_index != this->total_layers; ++tmp_layer_index)
    {
        tmp_ptr_layer_initialization = this->ptr_array_layers + tmp_layer_index;
        
        if(tmp_Is__Valid(tmp_ptr_layer_initialization))
        {
            tmp_trial_index = 0_zu;
        
            tmp_variance = tmp_Get__Variance__Loop(tmp_layer_index);

            while(MyEA::Math::Absolute<T_>(tmp_variance - tmp_variance_target) >= tmp_variance_tolerance)
            {
                tmp_Scale__Weight__Loop(this->ptr_array_parameters + *tmp_ptr_layer_initialization->ptr_first_connection_index,
                                                         this->ptr_array_parameters + *tmp_ptr_layer_initialization->ptr_last_connection_index,
                                                         1_T / (sqrt(tmp_variance) / sqrt(tmp_variance_target)));

                if(++tmp_trial_index < tmp_maximum_number_trials) { tmp_variance = tmp_Get__Variance__Loop(tmp_layer_index); }
                else { break; }
            }
        }
    }

    return(true);
}

bool Neural_Network::Initialization__LSUV__OpenMP(class Dataset<T_> const *const ptr_Dataset_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(ptr_Dataset_received == nullptr)
    {
        PRINT_FORMAT("%s: %s: ERROR: \"ptr_Dataset_received\" is a nullptr. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
#endif
    
    size_t const tmp_maximum_number_trials(this->_LSUV_Parameters.maximum_number_trials);
    size_t tmp_layer_index,
              tmp_trial_index;
    
    T_ const tmp_epsilon(this->_LSUV_Parameters.epsilon),
                 tmp_variance_target(MyEA::Math::Maximum<T_>(this->_LSUV_Parameters.variance_target, tmp_epsilon)),
                 tmp_variance_tolerance(this->_LSUV_Parameters.variance_tolerance);
    T_ tmp_variance;

    struct Layer const *tmp_ptr_layer_initialization;
    
    auto tmp_Is__Valid([](struct Layer const *const ptr_layer_received) -> bool
    {
        return(ptr_layer_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_CONVOLUTION
                 ||
                 ptr_layer_received->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED);
        
    });

    auto tmp_Get__Variance__OpenMP([&](size_t const layer_index_received) -> T_
    {
        size_t const tmp_number_examples(MyEA::Math::Minimum<size_t>(this->_LSUV_Parameters.maximum_batch_size, ptr_Dataset_received->Dataset<T_>::Get__Number_Examples())),
                           tmp_maximum_batch_size(this->batch_size),
                           tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
        size_t tmp_batch_size(0_zu),
                  tmp_batch_index(0_zu);
        
        T_ tmp_variance(0_T);
        
        #pragma omp parallel private(tmp_batch_index, tmp_batch_size)
        for(tmp_batch_index = 0_zu; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
        {
            tmp_batch_size = tmp_batch_index + 1_zu != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - tmp_batch_index * tmp_maximum_batch_size;
            
            this->Forward_Pass(tmp_batch_size,
                                          ptr_Dataset_received->Dataset<T_>::Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size,
                                          0ll,
                                          static_cast<long long int>(layer_index_received) + 1ll);

            #pragma omp barrier
            #pragma omp single
            tmp_variance += this->Get__Outputs__Variance(tmp_ptr_layer_initialization, tmp_batch_size);
        }

        return(tmp_variance <= 0_T ? tmp_epsilon : tmp_variance);
    });
    
    // TODO: Make function "Scale__Weight__OpenMP" global.
    auto tmp_Scale__Weight__OpenMP([](int const number_connections_received,
                                                             T_ *const ptr_ptr_array_weights_received,
                                                             T_ const scale_received) -> void
    {
        int tmp_connection_index__int;
        
        #pragma omp parallel for schedule(static)
        for(tmp_connection_index__int = 0; tmp_connection_index__int < number_connections_received; ++tmp_connection_index__int) { ptr_ptr_array_weights_received[tmp_connection_index__int] *= scale_received; }
    });

    for(tmp_layer_index = 1_zu; tmp_layer_index != this->total_layers; ++tmp_layer_index)
    {
        tmp_ptr_layer_initialization = this->ptr_array_layers + tmp_layer_index;
        
        if(tmp_Is__Valid(tmp_ptr_layer_initialization))
        {
            tmp_trial_index = 0_zu;
            
            tmp_variance = tmp_Get__Variance__OpenMP(tmp_layer_index);

            while(MyEA::Math::Absolute<T_>(tmp_variance - tmp_variance_target) >= tmp_variance_tolerance)
            {
                tmp_Scale__Weight__OpenMP(static_cast<int>(*tmp_ptr_layer_initialization->ptr_last_connection_index - *tmp_ptr_layer_initialization->ptr_first_connection_index),
                                                               this->ptr_array_parameters + *tmp_ptr_layer_initialization->ptr_first_connection_index,
                                                               1_T / (sqrt(tmp_variance) / sqrt(tmp_variance_target)));

                if(++tmp_trial_index < tmp_maximum_number_trials) { tmp_variance = tmp_Get__Variance__OpenMP(tmp_layer_index); }
                else { break; }
            }
        }
    }

    return(true);
}