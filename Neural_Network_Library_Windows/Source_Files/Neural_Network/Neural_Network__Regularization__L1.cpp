#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

T_ Neural_Network::Get__Regularization__L1(void) const { return(this->regularization__l1); }

bool Neural_Network::Set__Regularization__L1(T_ const regularization__l1_received)
{
    if(regularization__l1_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: L1 regularization (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(regularization__l1_received),
                                 __LINE__);

        return(false);
    }
    else if(regularization__l1_received > 1_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: L1 regularization (%f) bigger than one. At line %d." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 __FUNCTION__,
                                 Cast_T(regularization__l1_received),
                                 __LINE__);

        return(false);
    }

    if(this->regularization__l1 != regularization__l1_received)
    {
        bool const tmp_use_regularization(this->Use__Regularization_Parameter());

        this->regularization__l1 = regularization__l1_received;

        if(tmp_use_regularization == false && regularization__l1_received != 0_T)
        {
            if(this->Allocate__Parameter__Regularization() == false)
            {
                PRINT_FORMAT("%s: %s: ERROR: An error has been triggered from the \"Allocate__Parameter__Regularization()\" function. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         __LINE__);

                return(false);
            }

            if(this->pre_training_level != 0_zu) { this->Indexing_Regularization_Parameters__Pre_training(); }
            else { this->Indexing_Regularization_Parameters(); }
        }

        if(this->Use__Regularization_Parameter() == false)
        { this->Deallocate__Parameter__Regularization(); }

    #if defined(COMPILE_CUDA)
        if(this->is_device_initialized)
        { this->ptr_device_Neural_Network->Set__Regularization__L1(regularization__l1_received); }
    #endif
    }

    return(true);
}

void Neural_Network::Update_Derivative_Weight__Regularization__L1(size_t const start_index_received,
                                                                                                   size_t const end_index_received,
                                                                                                   size_t const batch_size_received)
{
    if(this->use_OpenMP && this->is_OpenMP_initialized)
    {
        this->Update_Derivative_Weight__Regularization__L1__OpenMP(start_index_received,
                                                                                                    end_index_received,
                                                                                                    batch_size_received);
    }
    else
    {
        this->Update_Derivative_Weight__Regularization__L1__Loop(start_index_received,
                                                                                               end_index_received,
                                                                                               batch_size_received);
    }
}
    
void Neural_Network::Update_Derivative_Weight__Regularization__L1__Loop(size_t const start_index_received,
                                                                                                              size_t const end_index_received,
                                                                                                              size_t const batch_size_received)
{
    T_ *tmp_ptr_gradient_it(this->ptr_array_derivatives_parameters + start_index_received);
    T_ const *const tmp_ptr_last_gradient(tmp_ptr_gradient_it + end_index_received),
                  *tmp_ptr_weight_it(this->ptr_array_parameters + start_index_received),
                  *tmp_ptr_connections_mask_regularization_it(this->ptr_array_mask_regularized_parameters + start_index_received);

    for(; tmp_ptr_gradient_it != tmp_ptr_last_gradient; ++tmp_ptr_gradient_it,
                                                                            ++tmp_ptr_weight_it,
                                                                            ++tmp_ptr_connections_mask_regularization_it)
    { *tmp_ptr_gradient_it += *tmp_ptr_connections_mask_regularization_it * MyEA::Math::Sign<T_>(*tmp_ptr_weight_it) * this->regularization__l1; }
}
    
void Neural_Network::Update_Derivative_Weight__Regularization__L1__OpenMP(size_t const start_index_received,
                                                                                                                   size_t const end_index_received,
                                                                                                                   size_t const batch_size_received)
{
    int tmp_end_index__int(static_cast<int>(end_index_received)),
        tmp_connection_index;
        
    T_ const *const tmp_ptr_array_parameters(this->ptr_array_parameters),
                  *const tmp_ptr_array_connections_mask_regularization(this->ptr_array_mask_regularized_parameters);
    T_ *const tmp_ptr_array_gradients(this->ptr_array_derivatives_parameters);
    
    #pragma omp parallel for schedule(static)
    for(tmp_connection_index = static_cast<int>(start_index_received); tmp_connection_index < tmp_end_index__int; ++tmp_connection_index)
    { tmp_ptr_array_gradients[tmp_connection_index] += tmp_ptr_array_connections_mask_regularization[tmp_connection_index] * MyEA::Math::Sign<T_>(tmp_ptr_array_parameters[tmp_connection_index]) * this->regularization__l1; }
}
    