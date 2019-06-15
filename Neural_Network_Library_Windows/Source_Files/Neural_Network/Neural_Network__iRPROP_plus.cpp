#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

void Neural_Network::Update_Parameter__iRPROP_plus(size_t const start_index_received, size_t const end_index_received)
{
    if(this->use_OpenMP && this->is_OpenMP_initialized)
    { this->Update_Parameter__iRPROP_plus__OpenMP(start_index_received, end_index_received); }
    else
    { this->Update_Parameter__iRPROP_plus__Loop(start_index_received, end_index_received); }
}
    
void Neural_Network::Update_Parameter__iRPROP_plus__Loop(size_t const start_index_received, size_t const end_index_received)
{
    bool const tmp_error_is_worst(this->loss_rprop > this->previous_loss_rprop);
    
    T_ const tmp_increase_factor(this->rprop_increase_factor), // 1.2
                 tmp_decrease_factor(this->rprop_decrease_factor), // 0.5
                 tmp_delta_minimum(this->rprop_delta_min), // 1e-6
                 tmp_delta_maximum(this->rprop_delta_max); // 50.0
    T_ *const tmp_ptr_array_partial_derivative(this->ptr_array_derivatives_parameters),
         *const tmp_ptr_array_parameters(this->ptr_array_parameters),
         *const tmp_ptr_array_previous_delta_weight(this->ptr_array_previous_delta_parameters),
         *const tmp_ptr_array_previous_step(this->ptr_array_previous_steps),
         *const tmp_ptr_array_previous_partial_derivative(this->ptr_array_previous_derivatives_parameters),
         tmp_partial_derivative,
         tmp_delta_weight,
         tmp_delta_step;
        
    for(size_t i(start_index_received); i != end_index_received; ++i)
    {
    #if defined(COMPILE_DEBUG_PRINT)
        //if(MyEA::Math::Absolute(tmp_ptr_array_parameters[i]) > 8_T)
        {
            PRINT_FORMAT("|===|" NEW_LINE);
            PRINT_FORMAT("B-parameter[%d](%.9f)" NEW_LINE, i, tmp_ptr_array_parameters[i]);
            PRINT_FORMAT("partial-derivative[%d](%.9f)" NEW_LINE, i, tmp_ptr_array_partial_derivative[i]);
            PRINT_FORMAT("previous-partial-derivative[%d](%.9f)" NEW_LINE, i, tmp_ptr_array_previous_partial_derivative[i]);
            PRINT_FORMAT("previous-step[%d](%.9f)" NEW_LINE, i, tmp_ptr_array_previous_step[i]);
        }
    #endif

        //tmp_partial_derivative = -tmp_ptr_array_partial_derivative[i];  // Gradient ascent
        tmp_partial_derivative = tmp_ptr_array_partial_derivative[i];  // Gradient descent
            
        if(tmp_ptr_array_previous_partial_derivative[i] * tmp_partial_derivative > 0_T)
        {
            tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_increase_factor;
            tmp_ptr_array_previous_step[i] = tmp_delta_step = MyEA::Math::Minimum<T_>(tmp_delta_step, tmp_delta_maximum);

            tmp_ptr_array_previous_delta_weight[i] = tmp_delta_weight = -MyEA::Math::Sign<T_>(tmp_partial_derivative) * tmp_delta_step;
            
            /*
            if(MyEA::Math::Absolute(tmp_ptr_array_parameters[i]) > 8_T)
            {
                PRINT_FORMAT("#0, delta_step = min(previous_step(%f) * increase(%f), delta_max(%f)) = delta_step(%f)" NEW_LINE, tmp_previous_step, tmp_increase_factor, tmp_delta_maximum, tmp_delta_step);
                PRINT_FORMAT("#0, delta_weight = -MyEA::Math::Sign(dWeight(%f)==%f) * delta_step(%f) = delta_weight(%f)" NEW_LINE, tmp_partial_derivative, MyEA::Math::Sign<T_>(tmp_partial_derivative), tmp_delta_step, tmp_delta_weight);
                PRINT_FORMAT("#0, weight[%u](%f) += delta_weight(%f) = weight(%f)" NEW_LINE, i, tmp_ptr_array_parameters[i], tmp_delta_weight, tmp_ptr_array_parameters[i] + tmp_delta_weight);
            }
            */

            tmp_ptr_array_parameters[i] += tmp_delta_weight;

            tmp_ptr_array_previous_partial_derivative[i] = tmp_partial_derivative;
        }
        else if(tmp_ptr_array_previous_partial_derivative[i] * tmp_partial_derivative < 0_T)
        {
            tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_decrease_factor;
            tmp_ptr_array_previous_step[i] = MyEA::Math::Maximum<T_>(tmp_delta_step, tmp_delta_minimum);
                
            /*
            if(MyEA::Math::Absolute(tmp_ptr_array_parameters[i]) > 8_T)
            {
                PRINT_FORMAT("#1, delta_step = min(previous_step(%f) * decrease(%f), delta_min(%f)) = delta_step(%f)" NEW_LINE, tmp_previous_step, tmp_decrease_factor, tmp_delta_minimum, tmp_delta_step);
                PRINT_FORMAT("#1, if(error_is_worst==%u) { weight[%u](%f) -= previous_delta_weight(%f) = weight(%f) }" NEW_LINE, static_cast<size_t>(tmp_error_is_worst), i, tmp_ptr_array_parameters[i], tmp_ptr_array_previous_delta_weight[i], tmp_ptr_array_parameters[i] - tmp_ptr_array_previous_delta_weight[i]);
            }
            */
                
            if(tmp_error_is_worst) { tmp_ptr_array_parameters[i] -= tmp_ptr_array_previous_delta_weight[i]; }
                
            tmp_ptr_array_previous_partial_derivative[i] = 0_T;
        }
        else // if(tmp_ptr_array_previous_partial_derivative[i] * tmp_partial_derivative == 0_T)
        {
            tmp_ptr_array_previous_delta_weight[i] = tmp_delta_weight = -MyEA::Math::Sign<T_>(tmp_partial_derivative) * tmp_ptr_array_previous_step[i];
                
            /*
            if(MyEA::Math::Absolute(tmp_ptr_array_parameters[i]) > 8_T)
            {
                PRINT_FORMAT("#2, delta_weight = -MyEA::Math::Sign(dWeight(%f)==%f) * delta_step(%f) = delta_weight(%f)" NEW_LINE, tmp_partial_derivative, MyEA::Math::Sign<T_>(tmp_partial_derivative), tmp_delta_step, tmp_delta_weight);
                PRINT_FORMAT("#2, weight[%u](%f) += delta_weight(%f) = weight(%f)" NEW_LINE, i, tmp_ptr_array_parameters[i], tmp_delta_weight, tmp_ptr_array_parameters[i] + tmp_delta_weight);
            }
            */
                
            tmp_ptr_array_parameters[i] += tmp_delta_weight;

            tmp_ptr_array_previous_partial_derivative[i] = tmp_partial_derivative;
        }

        tmp_ptr_array_partial_derivative[i] = 0_T; // tmp_partial_derivative

    #if defined(COMPILE_DEBUG_PRINT)
        //if(MyEA::Math::Absolute(tmp_ptr_array_parameters[i]) > 8_T)
        {
            PRINT_FORMAT("A-parameter[%d](%.9f)" NEW_LINE, i, tmp_ptr_array_parameters[i]);
            PRINT_FORMAT("|END|" NEW_LINE);
        }
    #endif
    }
}
    
void Neural_Network::Update_Parameter__iRPROP_plus__OpenMP(size_t const start_index_received, size_t const end_index_received)
{
    bool const tmp_error_is_worst(this->loss_rprop > this->previous_loss_rprop);

    int const tmp_end_index(static_cast<int>(end_index_received));

    T_ const tmp_increase_factor(this->rprop_increase_factor), // 1.2
                 tmp_decrease_factor(this->rprop_decrease_factor), // 0.5
                 tmp_delta_minimum(this->rprop_delta_min), // 1e-6
                 tmp_delta_maximum(this->rprop_delta_max); // 50.0
    T_ *const tmp_ptr_array_partial_derivative(this->ptr_array_derivatives_parameters),
         *const tmp_ptr_array_parameters(this->ptr_array_parameters),
         *const tmp_ptr_array_previous_delta_weight(this->ptr_array_previous_delta_parameters),
         *const tmp_ptr_array_previous_step(this->ptr_array_previous_steps),
         *const tmp_ptr_array_previous_partial_derivative(this->ptr_array_previous_derivatives_parameters),
         tmp_partial_derivative(0),
         tmp_delta_weight(0),
         tmp_delta_step(0);

    #pragma omp parallel for schedule(static) private(tmp_partial_derivative, \
                                                                           tmp_delta_weight, \
                                                                           tmp_delta_step)
    for(int i = static_cast<int>(start_index_received); i < tmp_end_index; ++i)
    {
        //tmp_partial_derivative = -tmp_ptr_array_partial_derivative[i];  // Gradient ascent
        tmp_partial_derivative = tmp_ptr_array_partial_derivative[i];  // Gradient descent

        if(tmp_ptr_array_previous_partial_derivative[i] * tmp_partial_derivative > 0_T)
        {
            tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_increase_factor;
            tmp_ptr_array_previous_step[i] = tmp_delta_step = MyEA::Math::Minimum<T_>(tmp_delta_step, tmp_delta_maximum);

            tmp_ptr_array_previous_delta_weight[i] = tmp_delta_weight = -MyEA::Math::Sign<T_>(tmp_partial_derivative) * tmp_delta_step;

            tmp_ptr_array_parameters[i] += tmp_delta_weight;

            tmp_ptr_array_previous_partial_derivative[i] = tmp_partial_derivative;
        }
        else if(tmp_ptr_array_previous_partial_derivative[i] * tmp_partial_derivative < 0_T)
        {
            tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_decrease_factor;
            tmp_ptr_array_previous_step[i] = MyEA::Math::Maximum<T_>(tmp_delta_step, tmp_delta_minimum);

            if(tmp_error_is_worst) { tmp_ptr_array_parameters[i] -= tmp_ptr_array_previous_delta_weight[i]; }

            tmp_ptr_array_previous_partial_derivative[i] = 0_T;
        }
        else // if(tmp_ptr_array_previous_partial_derivative[i] * tmp_partial_derivative == 0_T)
        {
            tmp_ptr_array_previous_delta_weight[i] = tmp_delta_weight = -MyEA::Math::Sign<T_>(tmp_partial_derivative) * tmp_ptr_array_previous_step[i];

            tmp_ptr_array_parameters[i] += tmp_delta_weight;

            tmp_ptr_array_previous_partial_derivative[i] = tmp_partial_derivative;
        }

        tmp_ptr_array_partial_derivative[i] = 0_T; // tmp_partial_derivative
    }
}
