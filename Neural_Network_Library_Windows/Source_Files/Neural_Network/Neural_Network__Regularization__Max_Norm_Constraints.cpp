#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

#include <omp.h>

T_ Neural_Network::Get__Regularization__Max_Norm_Constraints(void) const { return(this->regularization__max_norm_constraints); }

bool Neural_Network::Set__Regularization__Max_Norm_Constraints(T_ const regularization__max_norm_constraints_received)
{
    if(this->regularization__max_norm_constraints == regularization__max_norm_constraints_received) { return(true); }
    else if(regularization__max_norm_constraints_received < 0_T)
    {
        PRINT_FORMAT("%s: %s: ERROR: Max-norm constraints (%f) less than zero. At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 Cast_T(regularization__max_norm_constraints_received),
                                 __LINE__);

        return(false);
    }

    this->regularization__max_norm_constraints = regularization__max_norm_constraints_received;

#if defined(COMPILE_CUDA)
    if(this->is_device_initialized)
    { this->ptr_device_Neural_Network->Set__Regularization__Max_Norm_Constraints(regularization__max_norm_constraints_received); }
#endif

    return(true);
}

void Neural_Network::Euclidean_Norm__Loop(size_t const start_index_received,
                                                                   size_t const end_index_received,
                                                                   T_ const max_norm_received,
                                                                   T_ *const ptr_array_vector_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(start_index_received > end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not begin above end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 start_index_received,
                                 end_index_received,
                                 __LINE__);
        
        return;
    }
#endif

    size_t tmp_index;

    T_ tmp_norms(0),
        tmp_desired_norm;

    // Euclidean Norm.
    for(tmp_index = start_index_received; tmp_index != end_index_received; ++tmp_index)
    { tmp_norms += ptr_array_vector_received[tmp_index] * ptr_array_vector_received[tmp_index]; }
    
    // Square root.
    tmp_norms = sqrt(tmp_norms);

    // Threshold.
    if(tmp_norms >= max_norm_received)
    {
        tmp_desired_norm = max_norm_received / tmp_norms;
        
        for(tmp_index = start_index_received; tmp_index != end_index_received; ++tmp_index)
        { ptr_array_vector_received[tmp_index] *= tmp_desired_norm; }
    }
}

void Neural_Network::Euclidean_Norm__OpenMP(size_t const start_index_received,
                                                                        size_t const end_index_received,
                                                                        T_ const max_norm_received,
                                                                        T_ *const ptr_array_vector_received)
{
#if defined(_DEBUG) || defined(COMPILE_DEBUG)
    if(start_index_received > end_index_received)
    {
        PRINT_FORMAT("%s: %s: ERROR: Start index (%zu) can not begin above end index (%zu). At line %d." NEW_LINE,
                                 MyEA::Time::Date_Time_Now().c_str(),
                                 __FUNCTION__,
                                 start_index_received,
                                 end_index_received,
                                 __LINE__);
        
        return;
    }
#endif

    int const tmp_start_index__int(static_cast<int>(start_index_received)),
                 tmp_end_index__int(static_cast<int>(end_index_received));
    int tmp_index__int;

    T_ tmp_norms(0),
        tmp_desired_norm;

    // Euclidean Norm.
#if defined(COMPILE_AUTODIFF)
    for(tmp_index__int = tmp_start_index__int; tmp_index__int < tmp_end_index__int; ++tmp_index__int)
    { tmp_norms += ptr_array_vector_received[tmp_index__int] * ptr_array_vector_received[tmp_index__int]; }
#else
    #pragma omp parallel for reduction(+ : tmp_norms)
    for(tmp_index__int = tmp_start_index__int; tmp_index__int < tmp_end_index__int; ++tmp_index__int)
    { tmp_norms += ptr_array_vector_received[tmp_index__int] * ptr_array_vector_received[tmp_index__int]; }
#endif

    // Square root.
    tmp_norms = sqrt(tmp_norms);

    // Threshold.
    if(tmp_norms >= max_norm_received)
    {
        tmp_desired_norm = max_norm_received / tmp_norms;
        
        #pragma omp parallel for schedule(static)
        for(tmp_index__int = tmp_start_index__int; tmp_index__int < tmp_end_index__int; ++tmp_index__int)
        { ptr_array_vector_received[tmp_index__int] *= tmp_desired_norm; }
    }
}

void Neural_Network::Update_Weight_Regularization__Max_Norm_Constraints(size_t const start_index_received, size_t const end_index_received)
{
    if(this->use_OpenMP && this->is_OpenMP_initialized)
    { this->Update_Weight_Regularization__Max_Norm_Constraints__OpenMP(start_index_received, end_index_received); }
    else
    { this->Update_Weight_Regularization__Max_Norm_Constraints__Loop(start_index_received, end_index_received); }
}

void Neural_Network::Update_Weight_Regularization__Max_Norm_Constraints__Neurons__Loop(size_t const start_index_received,
                                                                                                                                            size_t const end_index_received,
                                                                                                                                            struct Layer const *const ptr_layer_it_received,
                                                                                                                                            struct Layer const *const ptr_last_layer_received)
{
    struct Neuron_unit const *const tmp_ptr_last_neuron_unit(ptr_last_layer_received->ptr_last_neuron_unit),
                                  *tmp_ptr_neuron_unit_it(ptr_layer_it_received->ptr_array_neuron_units);

    for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
    {
        if(*tmp_ptr_neuron_unit_it->ptr_first_connection_index < start_index_received) { continue; }
        else if(*tmp_ptr_neuron_unit_it->ptr_last_connection_index > end_index_received) { break; }

        this->Euclidean_Norm__Loop(0_zu,
                                                    *tmp_ptr_neuron_unit_it->ptr_number_connections,
                                                    this->regularization__max_norm_constraints,
                                                    this->ptr_array_parameters + *tmp_ptr_neuron_unit_it->ptr_first_connection_index);
    }
}

void Neural_Network::Update_Weight_Regularization__Max_Norm_Constraints__LSTMs__Loop(size_t const start_index_received,
                                                                                                                                           size_t const end_index_received,
                                                                                                                                           struct Layer const *const ptr_layer_it_received,
                                                                                                                                           struct Layer const *const ptr_last_layer_received)
{
    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_last_layer_received->ptr_last_block_unit),
                                       *tmp_ptr_block_unit_it(ptr_layer_it_received->ptr_array_block_units);
        
    struct Cell_unit const *tmp_ptr_last_cell_unit,
                                    *tmp_ptr_cell_unit_it;
        
    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        // [0] Cell input.
        for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
            tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
        {
            //    [1] Input.
            if(tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input > start_index_received)
            {
                if(tmp_ptr_cell_unit_it->last_index_feedforward_connection_cell_input > end_index_received) { break; }

                this->Euclidean_Norm__Loop(0_zu,
                                                            tmp_ptr_cell_unit_it->last_index_feedforward_connection_cell_input - tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input,
                                                            this->regularization__max_norm_constraints,
                                                            this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input);
            }
            //    [1] |END| Input. |END|

            //    [1] Recurrent.
            if(tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input > start_index_received)
            {
                if(tmp_ptr_cell_unit_it->last_index_recurrent_connection_cell_input > end_index_received) { break; }

                this->Euclidean_Norm__Loop(0_zu,
                                                            tmp_ptr_cell_unit_it->last_index_recurrent_connection_cell_input - tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input,
                                                            this->regularization__max_norm_constraints,
                                                            this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input);
            }
            //    [1] |END| Recurrent. |END|
        }
        // [0] |END| Cell input. |END|
        
        // [0] Gates.
        //    [1] Input.
        if(tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate > start_index_received)
        {
            if(tmp_ptr_block_unit_it->last_index_feedforward_connection_input_gate > end_index_received) { break; }
            
            this->Euclidean_Norm__Loop(0_zu,
                                                        tmp_ptr_block_unit_it->last_index_feedforward_connection_input_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate,
                                                        this->regularization__max_norm_constraints,
                                                        this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate);
        }

        if(tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate > start_index_received)
        {
            if(tmp_ptr_block_unit_it->last_index_feedforward_connection_forget_gate > end_index_received) { break; }
            
            this->Euclidean_Norm__Loop(0_zu,
                                                        tmp_ptr_block_unit_it->last_index_feedforward_connection_forget_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate,
                                                        this->regularization__max_norm_constraints,
                                                        this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate);
        }

        if(tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate > start_index_received)
        {
            if(tmp_ptr_block_unit_it->last_index_feedforward_connection_output_gate > end_index_received) { break; }
            
            this->Euclidean_Norm__Loop(0_zu,
                                                        tmp_ptr_block_unit_it->last_index_feedforward_connection_output_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate,
                                                        this->regularization__max_norm_constraints,
                                                        this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate);
        }
        //    [1] |END| Input. |END|

        //    [1] Recurrent.
        if(tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate > start_index_received)
        {
            if(tmp_ptr_block_unit_it->last_index_recurrent_connection_input_gate > end_index_received) { break; }
            
            this->Euclidean_Norm__Loop(0_zu,
                                                        tmp_ptr_block_unit_it->last_index_recurrent_connection_input_gate - tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate,
                                                        this->regularization__max_norm_constraints,
                                                        this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate);
        }

        if(tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate > start_index_received)
        {
            if(tmp_ptr_block_unit_it->last_index_recurrent_connection_forget_gate > end_index_received) { break; }
            
            this->Euclidean_Norm__Loop(0_zu,
                                                        tmp_ptr_block_unit_it->last_index_recurrent_connection_forget_gate - tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate,
                                                        this->regularization__max_norm_constraints,
                                                        this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate);
        }

        if(tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate > start_index_received)
        {
            if(tmp_ptr_block_unit_it->last_index_recurrent_connection_output_gate > end_index_received) { break; }
            
            this->Euclidean_Norm__Loop(0_zu,
                                                        tmp_ptr_block_unit_it->last_index_recurrent_connection_output_gate - tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate,
                                                        this->regularization__max_norm_constraints,
                                                        this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate);
        }
        //    [1] |END| Recurrent. |END|

        //    [1] Peepholes.
    #ifndef NO_PEEPHOLE
        if(tmp_ptr_block_unit_it->first_index_peephole_input_gate > start_index_received)
        {
            if(tmp_ptr_block_unit_it->last_index_peephole_input_gate > end_index_received) { break; }
            
            this->Euclidean_Norm__Loop(0_zu,
                                                        tmp_ptr_block_unit_it->last_index_peephole_input_gate - tmp_ptr_block_unit_it->first_index_peephole_input_gate,
                                                        this->regularization__max_norm_constraints,
                                                        this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate);
        }

        if(tmp_ptr_block_unit_it->first_index_peephole_forget_gate > start_index_received)
        {
            if(tmp_ptr_block_unit_it->last_index_peephole_forget_gate > end_index_received) { break; }
            
            this->Euclidean_Norm__Loop(0_zu,
                                                        tmp_ptr_block_unit_it->last_index_peephole_forget_gate - tmp_ptr_block_unit_it->first_index_peephole_forget_gate,
                                                        this->regularization__max_norm_constraints,
                                                        this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate);
        }

        if(tmp_ptr_block_unit_it->first_index_peephole_output_gate > start_index_received)
        {
            if(tmp_ptr_block_unit_it->last_index_peephole_output_gate > end_index_received) { break; }
            
            this->Euclidean_Norm__Loop(0_zu,
                                                        tmp_ptr_block_unit_it->last_index_peephole_output_gate - tmp_ptr_block_unit_it->first_index_peephole_output_gate,
                                                        this->regularization__max_norm_constraints,
                                                        this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate);
        }
    #endif
        //    [1] |END| Peepholes. |END|
        // [0] |END| Gates. |END|
    }
}
    
void Neural_Network::Update_Weight_Regularization__Max_Norm_Constraints__Loop(size_t const start_index_received, size_t const end_index_received)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer - 1);
    struct Layer const *tmp_ptr_layer_it(this->ptr_array_layers + 1);

    if(this->total_neuron_units != 0_zu)
    {
        this->Update_Weight_Regularization__Max_Norm_Constraints__Neurons__Loop(start_index_received,
                                                                                                                             end_index_received,
                                                                                                                             tmp_ptr_layer_it,
                                                                                                                             tmp_ptr_last_layer);
    }

    if(this->total_block_units != 0_zu)
    {
        this->Update_Weight_Regularization__Max_Norm_Constraints__LSTMs__Loop(start_index_received,
                                                                                                                            end_index_received,
                                                                                                                            tmp_ptr_layer_it,
                                                                                                                            tmp_ptr_last_layer);
    }
}
    
void Neural_Network::Update_Weight_Regularization__Max_Norm_Constraints__Neurons__OpenMP(size_t const start_index_received,
                                                                                                                                                  size_t const end_index_received,
                                                                                                                                                  struct Layer const *const ptr_layer_it_received,
                                                                                                                                                  struct Layer const *const ptr_last_layer_received)
{
    int const tmp_units_size__int(static_cast<int>(ptr_last_layer_received->ptr_last_neuron_unit - ptr_layer_it_received->ptr_array_neuron_units));
    int tmp_unit_index__int;

    struct Neuron_unit const *const tmp_ptr_array_neuron_units(ptr_layer_it_received->ptr_array_neuron_units);
    
    #pragma omp for schedule(static)
    for(tmp_unit_index__int = 0; tmp_unit_index__int < tmp_units_size__int; ++tmp_unit_index__int)
    {
        if(*tmp_ptr_array_neuron_units[tmp_unit_index__int].ptr_first_connection_index < start_index_received
          ||
          *tmp_ptr_array_neuron_units[tmp_unit_index__int].ptr_last_connection_index > end_index_received) { continue; }

        this->Euclidean_Norm__OpenMP(0_zu,
                                                         *tmp_ptr_array_neuron_units[tmp_unit_index__int].ptr_number_connections,
                                                         this->regularization__max_norm_constraints,
                                                         this->ptr_array_parameters + *tmp_ptr_array_neuron_units[tmp_unit_index__int].ptr_first_connection_index);
    }
}

void Neural_Network::Update_Weight_Regularization__Max_Norm_Constraints__LSTMs__OpenMP(size_t const start_index_received,
                                                                                                                                                size_t const end_index_received,
                                                                                                                                                struct Layer const *const ptr_layer_it_received,
                                                                                                                                                struct Layer const *const ptr_last_layer_received)
{
    int const tmp_units_size__int(static_cast<int>(ptr_last_layer_received->ptr_last_block_unit - ptr_layer_it_received->ptr_array_block_units));
    int tmp_unit_index__int;

    struct Block_unit const *const tmp_ptr_array_block_units(ptr_layer_it_received->ptr_array_block_units);
    
    struct Cell_unit const *tmp_ptr_last_cell_unit,
                                    *tmp_ptr_cell_unit_it;
    
    #pragma omp for schedule(static)
    for(tmp_unit_index__int = 0; tmp_unit_index__int < tmp_units_size__int; ++tmp_unit_index__int)
    {
        // [0] Cell input.
        for(tmp_ptr_last_cell_unit = tmp_ptr_array_block_units[tmp_unit_index__int].ptr_last_cell_unit,
            tmp_ptr_cell_unit_it = tmp_ptr_array_block_units[tmp_unit_index__int].ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
        {
            //    [1] Input.
            if(tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input > start_index_received)
            {
                if(tmp_ptr_cell_unit_it->last_index_feedforward_connection_cell_input > end_index_received) { continue; }

                this->Euclidean_Norm__OpenMP(0_zu,
                                                                 tmp_ptr_cell_unit_it->last_index_feedforward_connection_cell_input - tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input,
                                                                 this->regularization__max_norm_constraints,
                                                                 this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input);
            }
            //    [1] |END| Input. |END|

            //    [1] Recurrent.
            if(tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input > start_index_received)
            {
                if(tmp_ptr_cell_unit_it->last_index_recurrent_connection_cell_input > end_index_received) { continue; }

                this->Euclidean_Norm__OpenMP(0_zu,
                                                                 tmp_ptr_cell_unit_it->last_index_recurrent_connection_cell_input - tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input,
                                                                 this->regularization__max_norm_constraints,
                                                                 this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input);
            }
            //    [1] |END| Recurrent. |END|
        }
        // [0] |END| Cell input. |END|
        
        // [0] Gates.
        //    [1] Input.
        if(tmp_ptr_array_block_units[tmp_unit_index__int].first_index_feedforward_connection_input_gate > start_index_received)
        {
            if(tmp_ptr_array_block_units[tmp_unit_index__int].last_index_feedforward_connection_input_gate > end_index_received) { continue; }
            
            this->Euclidean_Norm__OpenMP(0_zu,
                                                             tmp_ptr_array_block_units[tmp_unit_index__int].last_index_feedforward_connection_input_gate - tmp_ptr_array_block_units[tmp_unit_index__int].first_index_feedforward_connection_input_gate,
                                                             this->regularization__max_norm_constraints,
                                                             this->ptr_array_parameters + tmp_ptr_array_block_units[tmp_unit_index__int].first_index_feedforward_connection_input_gate);
        }
        
        if(tmp_ptr_array_block_units[tmp_unit_index__int].first_index_feedforward_connection_forget_gate > start_index_received)
        {
            if(tmp_ptr_array_block_units[tmp_unit_index__int].last_index_feedforward_connection_forget_gate > end_index_received) { continue; }
            
            this->Euclidean_Norm__OpenMP(0_zu,
                                                             tmp_ptr_array_block_units[tmp_unit_index__int].last_index_feedforward_connection_forget_gate - tmp_ptr_array_block_units[tmp_unit_index__int].first_index_feedforward_connection_forget_gate,
                                                             this->regularization__max_norm_constraints,
                                                             this->ptr_array_parameters + tmp_ptr_array_block_units[tmp_unit_index__int].first_index_feedforward_connection_forget_gate);
        }
        
        if(tmp_ptr_array_block_units[tmp_unit_index__int].first_index_feedforward_connection_output_gate > start_index_received)
        {
            if(tmp_ptr_array_block_units[tmp_unit_index__int].last_index_feedforward_connection_output_gate > end_index_received) { continue; }
            
            this->Euclidean_Norm__OpenMP(0_zu,
                                                             tmp_ptr_array_block_units[tmp_unit_index__int].last_index_feedforward_connection_output_gate - tmp_ptr_array_block_units[tmp_unit_index__int].first_index_feedforward_connection_output_gate,
                                                             this->regularization__max_norm_constraints,
                                                             this->ptr_array_parameters + tmp_ptr_array_block_units[tmp_unit_index__int].first_index_feedforward_connection_output_gate);
        }
        //    [1] |END| Input. |END|

        //    [1] Recurrent.
        if(tmp_ptr_array_block_units[tmp_unit_index__int].first_index_recurrent_connection_input_gate > start_index_received)
        {
            if(tmp_ptr_array_block_units[tmp_unit_index__int].last_index_recurrent_connection_input_gate > end_index_received) { continue; }
            
            this->Euclidean_Norm__OpenMP(0_zu,
                                                             tmp_ptr_array_block_units[tmp_unit_index__int].last_index_recurrent_connection_input_gate - tmp_ptr_array_block_units[tmp_unit_index__int].first_index_recurrent_connection_input_gate,
                                                             this->regularization__max_norm_constraints,
                                                             this->ptr_array_parameters + tmp_ptr_array_block_units[tmp_unit_index__int].first_index_recurrent_connection_input_gate);
        }
        
        if(tmp_ptr_array_block_units[tmp_unit_index__int].first_index_recurrent_connection_forget_gate > start_index_received)
        {
            if(tmp_ptr_array_block_units[tmp_unit_index__int].last_index_recurrent_connection_forget_gate > end_index_received) { continue; }
            
            this->Euclidean_Norm__OpenMP(0_zu,
                                                             tmp_ptr_array_block_units[tmp_unit_index__int].last_index_recurrent_connection_forget_gate - tmp_ptr_array_block_units[tmp_unit_index__int].first_index_recurrent_connection_forget_gate,
                                                             this->regularization__max_norm_constraints,
                                                             this->ptr_array_parameters + tmp_ptr_array_block_units[tmp_unit_index__int].first_index_recurrent_connection_forget_gate);
        }
        
        if(tmp_ptr_array_block_units[tmp_unit_index__int].first_index_recurrent_connection_output_gate > start_index_received)
        {
            if(tmp_ptr_array_block_units[tmp_unit_index__int].last_index_recurrent_connection_output_gate > end_index_received) { continue; }
            
            this->Euclidean_Norm__OpenMP(0_zu,
                                                             tmp_ptr_array_block_units[tmp_unit_index__int].last_index_recurrent_connection_output_gate - tmp_ptr_array_block_units[tmp_unit_index__int].first_index_recurrent_connection_output_gate,
                                                             this->regularization__max_norm_constraints,
                                                             this->ptr_array_parameters + tmp_ptr_array_block_units[tmp_unit_index__int].first_index_recurrent_connection_output_gate);
        }
        //    [1] |END| Recurrent. |END|

    #ifndef NO_PEEPHOLE
        //    [1] Peepholes.
        if(tmp_ptr_array_block_units[tmp_unit_index__int].first_index_peephole_input_gate > start_index_received)
        {
            if(tmp_ptr_array_block_units[tmp_unit_index__int].last_index_peephole_input_gate > end_index_received) { continue; }
            
            this->Euclidean_Norm__OpenMP(0_zu,
                                                             tmp_ptr_array_block_units[tmp_unit_index__int].last_index_peephole_input_gate - tmp_ptr_array_block_units[tmp_unit_index__int].first_index_peephole_input_gate,
                                                             this->regularization__max_norm_constraints,
                                                             this->ptr_array_parameters + tmp_ptr_array_block_units[tmp_unit_index__int].first_index_peephole_input_gate);
        }
        
        if(tmp_ptr_array_block_units[tmp_unit_index__int].first_index_peephole_forget_gate > start_index_received)
        {
            if(tmp_ptr_array_block_units[tmp_unit_index__int].last_index_peephole_forget_gate > end_index_received) { continue; }
            
            this->Euclidean_Norm__OpenMP(0_zu,
                                                             tmp_ptr_array_block_units[tmp_unit_index__int].last_index_peephole_forget_gate - tmp_ptr_array_block_units[tmp_unit_index__int].first_index_peephole_forget_gate,
                                                             this->regularization__max_norm_constraints,
                                                             this->ptr_array_parameters + tmp_ptr_array_block_units[tmp_unit_index__int].first_index_peephole_forget_gate);
        }
        
        if(tmp_ptr_array_block_units[tmp_unit_index__int].first_index_peephole_output_gate > start_index_received)
        {
            if(tmp_ptr_array_block_units[tmp_unit_index__int].last_index_peephole_output_gate > end_index_received) { continue; }
            
            this->Euclidean_Norm__OpenMP(0_zu,
                                                             tmp_ptr_array_block_units[tmp_unit_index__int].last_index_peephole_output_gate - tmp_ptr_array_block_units[tmp_unit_index__int].first_index_peephole_output_gate,
                                                             this->regularization__max_norm_constraints,
                                                             this->ptr_array_parameters + tmp_ptr_array_block_units[tmp_unit_index__int].first_index_peephole_output_gate);
        }
        //    [1] |END| Peepholes. |END|
    #endif
        // [0] |END| Gates. |END|
    }
}

void Neural_Network::Update_Weight_Regularization__Max_Norm_Constraints__OpenMP(size_t const start_index_received, size_t const end_index_received)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer - 1);
    struct Layer const *tmp_ptr_layer_it(this->ptr_array_layers + 1);
    
    #pragma omp parallel
    {
        if(this->total_neuron_units != 0_zu)
        {
            this->Update_Weight_Regularization__Max_Norm_Constraints__Neurons__OpenMP(start_index_received,
                                                                                                                                       end_index_received,
                                                                                                                                       tmp_ptr_layer_it,
                                                                                                                                       tmp_ptr_last_layer);
        }

        if(this->total_block_units != 0_zu)
        {
            this->Update_Weight_Regularization__Max_Norm_Constraints__LSTMs__OpenMP(start_index_received,
                                                                                                                                     end_index_received,
                                                                                                                                     tmp_ptr_layer_it,
                                                                                                                                     tmp_ptr_last_layer);
        }
    }
}
