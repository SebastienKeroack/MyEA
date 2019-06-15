#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

// TODO: Fix it. "Update_Parameter__iRPROP_minus__Loop"
void Neural_Network::Update_Parameter__iRPROP_minus__Loop(size_t const first_weight_received, size_t const past_end_received)
{
    T_ const tmp_increase_factor(this->rprop_increase_factor), // 1.2
                  tmp_decrease_factor(this->rprop_decrease_factor), // 0.5
                  tmp_delta_minimum(this->rprop_delta_min), // 1e-6
                  tmp_delta_maximum(this->rprop_delta_max); // 50.0
        
    T_ *const tmp_ptr_array_partial_derivative(this->ptr_array_derivatives_parameters),
            *const tmp_ptr_array_parameters(this->ptr_array_parameters),
            *const tmp_ptr_array_previous_step(this->ptr_array_previous_steps),
            *const tmp_ptr_array_previous_partial_derivative(this->ptr_array_previous_derivatives_parameters),
            tmp_partial_derivative,
            tmp_delta_step;

    for(size_t i(first_weight_received); i != past_end_received; ++i)
    {
    #if defined(COMPILE_DEBUG_PRINT)
        PRINT_FORMAT("B weight[%d] : %f" NEW_LINE, i, tmp_ptr_array_parameters[i]);
    #endif

        //tmp_partial_derivative = -tmp_ptr_array_partial_derivative[i];  // Gradient ascent
        tmp_partial_derivative = tmp_ptr_array_partial_derivative[i];  // Gradient descent
            
        if(tmp_ptr_array_previous_partial_derivative[i] * tmp_partial_derivative > 0_T)
        {
            tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_increase_factor;
            tmp_ptr_array_previous_step[i] = tmp_delta_step = MyEA::Math::Minimum<T_>(tmp_delta_step, tmp_delta_maximum);
        }
        else if(tmp_ptr_array_previous_partial_derivative[i] * tmp_partial_derivative < 0_T)
        {
            tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_decrease_factor;
            tmp_ptr_array_previous_step[i] = tmp_delta_step = MyEA::Math::Maximum<T_>(tmp_delta_step, tmp_delta_minimum);

            tmp_partial_derivative = 0_T;
        }
        else { tmp_delta_step = tmp_ptr_array_previous_step[i]; }

        tmp_ptr_array_parameters[i] += -MyEA::Math::Sign<T_>(tmp_partial_derivative) * tmp_delta_step;

        tmp_ptr_array_previous_partial_derivative[i] = tmp_partial_derivative;
        tmp_ptr_array_partial_derivative[i] = 0_T; // tmp_partial_derivative

    #if defined(COMPILE_DEBUG_PRINT)
        PRINT_FORMAT("A weight[%d] : %f" NEW_LINE, i, tmp_ptr_array_parameters[i]);
    #endif
    }
}
    