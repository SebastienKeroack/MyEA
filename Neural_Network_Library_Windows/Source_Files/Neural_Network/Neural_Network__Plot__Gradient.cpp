#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#if defined(COMPILE_UI)
    #include <Enums/Enum_Type_Chart.hpp>

    #include <Form.hpp>
#endif // COMPILE_UI

void Neural_Network::Plot__Gradient(void)
{
#if defined(COMPILE_UI)
    if(this->plot_gradient)
    {
        size_t tmp_datapoint_index(0_zu),
                  tmp_connection_index;
    
        T_ tmp_summation;

        struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
        struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + 1);
        
        //MyEA::Form::API__Form__Neural_Network__Chart_Reset(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_GRID_SEARCH);
        
        // Loop though each layer.
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
        {
            // If the current layer is a pooling/residual layer, continue.
            if(tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING
              ||
              tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING
              ||
              tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { continue; }
        
            tmp_summation = 0_T;

            for(tmp_connection_index = *tmp_ptr_layer_it->ptr_first_connection_index; tmp_connection_index != *tmp_ptr_layer_it->ptr_last_connection_index; ++tmp_connection_index)
            {
                tmp_summation += this->ptr_array_derivatives_parameters[tmp_connection_index];
            }

            tmp_summation /= *tmp_ptr_layer_it->ptr_last_connection_index - *tmp_ptr_layer_it->ptr_first_connection_index;
            
            MyEA::Form::API__Form__Neural_Network__Chart_Add_Point(MyEA::Common::ENUM_TYPE_CHART::TYPE_CHART_GRID_SEARCH,
                                                                                                        0u,
                                                                                                        MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                                                                                        static_cast<double>(tmp_datapoint_index),
                                                                                                        MyEA::Math::Absolute<T_>(tmp_summation));
            
            ++tmp_datapoint_index;
        }
    }
#endif // COMPILE_UI
}
