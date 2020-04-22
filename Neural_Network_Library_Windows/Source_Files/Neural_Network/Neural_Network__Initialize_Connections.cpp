#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

template<typename U> void Neural_Network::Initialize_Connections__FC(struct Layer *const ptr_layer_it_received, U *const ptr_previous_layer_array_units_received)
{
    size_t tmp_number_connections,
              tmp_connection_index;
    
    void **tmp_ptr_array_ptr_connections;

    T_ *tmp_ptr_array_parameters;

    struct Neuron_unit const *const tmp_ptr_last_neuron_unit(ptr_layer_it_received->ptr_last_neuron_unit);
    struct Neuron_unit *tmp_ptr_neuron_unit_it(ptr_layer_it_received->ptr_array_neuron_units);

    for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
    {
        tmp_ptr_array_ptr_connections = this->ptr_array_ptr_connections + *tmp_ptr_neuron_unit_it->ptr_first_connection_index;

        tmp_ptr_array_parameters = this->ptr_array_parameters + *tmp_ptr_neuron_unit_it->ptr_first_connection_index;

        tmp_number_connections = *tmp_ptr_neuron_unit_it->ptr_number_connections;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_connections; ++tmp_connection_index)
        {
            tmp_ptr_array_ptr_connections[tmp_connection_index] = ptr_previous_layer_array_units_received + tmp_connection_index;
            tmp_ptr_array_parameters[tmp_connection_index] = this->Class_Generator_Real();
        }
    }
}

template<typename U> void Neural_Network::Initialize_Connections__LSTM(struct Layer *const ptr_layer_it_received, U *const ptr_previous_layer_array_units_received)
{
    void **tmp_ptr_array_ptr_cell_input_connections,
          **tmp_ptr_array_ptr_input_gate_connections,
          **tmp_ptr_array_ptr_forget_gate_connections,
          **tmp_ptr_array_ptr_output_gate_connections;
    
    T_ *tmp_ptr_array_cell_input_parameters,
         *tmp_ptr_array_input_gate_parameters,
         *tmp_ptr_array_forget_gate_parameters,
         *tmp_ptr_array_output_gate_parameters;

    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it(ptr_layer_it_received->ptr_array_block_units);
    
    size_t const tmp_number_peephole_connections(tmp_ptr_block_unit_it->last_index_peephole_input_gate - tmp_ptr_block_unit_it->first_index_peephole_input_gate),
                       tmp_number_inputs_connections(tmp_ptr_block_unit_it->last_index_feedforward_connection_input_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrents_connection(tmp_ptr_block_unit_it->last_index_recurrent_connection_input_gate - tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate);
    size_t tmp_connection_index;

    struct Cell_unit const *tmp_ptr_block_ptr_last_cell_unit,
                                    *tmp_ptr_block_ptr_cell_unit_it;
    struct Cell_unit *const tmp_ptr_layer_ptr_first_cell_unit(ptr_layer_it_received->ptr_array_cell_units),
                           *tmp_ptr_block_ptr_first_cell_unit;

    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        tmp_ptr_block_ptr_first_cell_unit = tmp_ptr_block_unit_it->ptr_array_cell_units;

        // [0] Cell input.
        for(tmp_ptr_block_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
            tmp_ptr_block_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
        {
            //    [1] Input, cell input.
            tmp_ptr_array_cell_input_parameters = this->ptr_array_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

            tmp_ptr_array_ptr_cell_input_connections = this->ptr_array_ptr_connections + tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

            for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
            {
                tmp_ptr_array_cell_input_parameters[tmp_connection_index] = this->Class_Generator_Real();
                tmp_ptr_array_ptr_cell_input_connections[tmp_connection_index] = ptr_previous_layer_array_units_received + tmp_connection_index;
            }
            //    [1] |END| Input, cell input. |END|

            //    [1] Recurrent, input.
            tmp_ptr_array_cell_input_parameters = this->ptr_array_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;
            
            tmp_ptr_array_ptr_cell_input_connections = this->ptr_array_ptr_connections + tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;

            for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
            {
                tmp_ptr_array_cell_input_parameters[tmp_connection_index] = this->Class_Generator_Real();
                tmp_ptr_array_ptr_cell_input_connections[tmp_connection_index] = tmp_ptr_layer_ptr_first_cell_unit + tmp_connection_index;
            }
            //    [1] |END| Recurrent, input. |END|
        }
        // [0] |END| Cell input. |END|
        
        // [0] Input, gates.
        tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;
        tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;
        tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;
        
        tmp_ptr_array_ptr_input_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;
        tmp_ptr_array_ptr_forget_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;
        tmp_ptr_array_ptr_output_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
        {
            tmp_ptr_array_input_gate_parameters[tmp_connection_index] = this->Class_Generator_Real();
            tmp_ptr_array_forget_gate_parameters[tmp_connection_index] = this->Class_Generator_Real();
            tmp_ptr_array_output_gate_parameters[tmp_connection_index] = this->Class_Generator_Real();

            tmp_ptr_array_ptr_input_gate_connections[tmp_connection_index] = ptr_previous_layer_array_units_received + tmp_connection_index;
            tmp_ptr_array_ptr_forget_gate_connections[tmp_connection_index] = ptr_previous_layer_array_units_received + tmp_connection_index;
            tmp_ptr_array_ptr_output_gate_connections[tmp_connection_index] = ptr_previous_layer_array_units_received + tmp_connection_index;
        }
        // [0] |END| Input, gates. |END|

        // [0] Recurrent, gates.
        tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;
        tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;
        tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;
        
        tmp_ptr_array_ptr_input_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;
        tmp_ptr_array_ptr_forget_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;
        tmp_ptr_array_ptr_output_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
        {
            tmp_ptr_array_input_gate_parameters[tmp_connection_index] = this->Class_Generator_Real();
            tmp_ptr_array_forget_gate_parameters[tmp_connection_index] = this->Class_Generator_Real();
            tmp_ptr_array_output_gate_parameters[tmp_connection_index] = this->Class_Generator_Real();

            tmp_ptr_array_ptr_input_gate_connections[tmp_connection_index] = tmp_ptr_layer_ptr_first_cell_unit + tmp_connection_index;
            tmp_ptr_array_ptr_forget_gate_connections[tmp_connection_index] = tmp_ptr_layer_ptr_first_cell_unit + tmp_connection_index;
            tmp_ptr_array_ptr_output_gate_connections[tmp_connection_index] = tmp_ptr_layer_ptr_first_cell_unit + tmp_connection_index;
        }
        // [0] |END| Recurrent, gates. |END|

    #ifndef NO_PEEPHOLE
        // [0] Peepholes.
        tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate;
        tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate;
        tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate;
        
        tmp_ptr_array_ptr_input_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_peephole_input_gate;
        tmp_ptr_array_ptr_forget_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_peephole_forget_gate;
        tmp_ptr_array_ptr_output_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_peephole_output_gate;

        for(tmp_connection_index = 0_zu; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index)
        {
            tmp_ptr_array_input_gate_parameters[tmp_connection_index] = this->Class_Generator_Real();
            tmp_ptr_array_forget_gate_parameters[tmp_connection_index] = this->Class_Generator_Real();
            tmp_ptr_array_output_gate_parameters[tmp_connection_index] = this->Class_Generator_Real();

            tmp_ptr_array_ptr_input_gate_connections[tmp_connection_index] = tmp_ptr_block_ptr_first_cell_unit + tmp_connection_index;
            tmp_ptr_array_ptr_forget_gate_connections[tmp_connection_index] = tmp_ptr_block_ptr_first_cell_unit + tmp_connection_index;
            tmp_ptr_array_ptr_output_gate_connections[tmp_connection_index] = tmp_ptr_block_ptr_first_cell_unit + tmp_connection_index;
        }
        // [0] |END| Peepholes. |END|
    #endif
    }
}

void Neural_Network::Initialize_Connections__AF_Ind_Recurrent(struct Layer *const ptr_layer_it_received)
{
    struct AF_Ind_recurrent_unit const *const tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit);
    struct AF_Ind_recurrent_unit *tmp_ptr_AF_Ind_recurrent_unit_it(ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units);

    for(; tmp_ptr_AF_Ind_recurrent_unit_it != tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit; ++tmp_ptr_AF_Ind_recurrent_unit_it)
    {
        this->ptr_array_ptr_connections[*tmp_ptr_AF_Ind_recurrent_unit_it->ptr_recurrent_connection_index] = tmp_ptr_AF_Ind_recurrent_unit_it;
        this->ptr_array_parameters[*tmp_ptr_AF_Ind_recurrent_unit_it->ptr_recurrent_connection_index] = this->Class_Generator_Real();
    }
}

void Neural_Network::Initialize_Connections__Bias(struct Layer *const ptr_layer_it_received)
{
    size_t const tmp_number_connections(ptr_layer_it_received->last_bias_connection_index - ptr_layer_it_received->first_bias_connection_index);

    if(tmp_number_connections != 0_zu)
    {
        void **tmp_ptr_array_ptr_connections(this->ptr_array_ptr_connections + ptr_layer_it_received->first_bias_connection_index);
        MyEA::Memory::Cpp::Fill_Nullptr(tmp_ptr_array_ptr_connections, tmp_ptr_array_ptr_connections + tmp_number_connections);

        T_ *tmp_ptr_array_parameters(this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index);
        MEMSET(tmp_ptr_array_parameters,
                    0,
                    tmp_number_connections * sizeof(T_));
    }
}

void Neural_Network::Initialize_Connections__LSTM__Bias(struct Layer *const ptr_layer_it_received)
{
    size_t const tmp_number_connections(ptr_layer_it_received->last_bias_connection_index - ptr_layer_it_received->first_bias_connection_index);
    
    if(tmp_number_connections != 0_zu)
    {
        void **tmp_ptr_array_ptr_connections(this->ptr_array_ptr_connections + ptr_layer_it_received->first_bias_connection_index);
        MyEA::Memory::Cpp::Fill_Nullptr(tmp_ptr_array_ptr_connections, tmp_ptr_array_ptr_connections + tmp_number_connections);

        // Bias.
        size_t const tmp_number_cell_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units)),
                           tmp_number_block_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - ptr_layer_it_received->ptr_array_block_units));

        T_ *tmp_ptr_array_parameters(this->ptr_array_parameters + ptr_layer_it_received->first_bias_connection_index);

        //  Cell input && Input gate.
        MEMSET(tmp_ptr_array_parameters,
                    0,
                    tmp_number_cell_units + tmp_number_block_units * sizeof(T_));
        tmp_ptr_array_parameters += tmp_number_cell_units + tmp_number_block_units;
        //  |END| Cell input && Input gate. |END|
        
        //  Forget gate.
        for(T_ const *const tmp_parameter_end(tmp_ptr_array_parameters + tmp_number_block_units); tmp_ptr_array_parameters != tmp_parameter_end; ++tmp_ptr_array_parameters) { *tmp_ptr_array_parameters = 1_T; }
        //  |END| Forget gate. |END|

        //  Output gate.
        MEMSET(tmp_ptr_array_parameters,
                    0,
                    tmp_number_block_units * sizeof(T_));
        //  |END| Output gate. |END|
        // |END| Bias. |END|
    }
}

void Neural_Network::Initialize_Connections__FC_to_FC(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received) { this->Initialize_Connections__FC<struct Neuron_unit>(ptr_layer_it_received, ptr_previous_layer_it_received->ptr_array_neuron_units); }

void Neural_Network::Initialize_Connections__FC_to_LSTM(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received)
{
    if(ptr_layer_it_received->Use__Bidirectional())
    {
        this->Initialize_Connections__LSTM<struct Neuron_unit>(&ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer, ptr_previous_layer_it_received->ptr_array_neuron_units);
        this->Initialize_Connections__LSTM<struct Neuron_unit>(&ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer, ptr_previous_layer_it_received->ptr_array_neuron_units);
    }
    else { this->Initialize_Connections__LSTM<struct Neuron_unit>(ptr_layer_it_received, ptr_previous_layer_it_received->ptr_array_neuron_units); }
}

void Neural_Network::Initialize_Connections__LSTM_to_FC(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received) { this->Initialize_Connections__FC<struct Cell_unit>(ptr_layer_it_received, ptr_previous_layer_it_received->ptr_array_cell_units); }

void Neural_Network::Initialize_Connections__LSTM_to_LSTM(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received)
{
    if(ptr_layer_it_received->Use__Bidirectional())
    {
        this->Initialize_Connections__LSTM<struct Cell_unit>(&ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer, ptr_previous_layer_it_received->ptr_array_cell_units);
        this->Initialize_Connections__LSTM<struct Cell_unit>(&ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer, ptr_previous_layer_it_received->ptr_array_cell_units);
    }
    else { this->Initialize_Connections__LSTM<struct Cell_unit>(ptr_layer_it_received, ptr_previous_layer_it_received->ptr_array_cell_units); }
}

void Neural_Network::Initialize_Connections__Basic_unit_to_FC(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received) { this->Initialize_Connections__FC<struct Basic_unit>(ptr_layer_it_received, ptr_previous_layer_it_received->ptr_array_basic_units); }

void Neural_Network::Initialize_Connections__Basic_unit_to_LSTM(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received)
{
    if(ptr_layer_it_received->Use__Bidirectional())
    {
        this->Initialize_Connections__LSTM<struct Basic_unit>(&ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer, ptr_previous_layer_it_received->ptr_array_basic_units);
        this->Initialize_Connections__LSTM<struct Basic_unit>(&ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer, ptr_previous_layer_it_received->ptr_array_basic_units);
    }
    else { this->Initialize_Connections__LSTM<struct Basic_unit>(ptr_layer_it_received, ptr_previous_layer_it_received->ptr_array_basic_units); }
}

void Neural_Network::Initialize_Connections__Basic_indice_unit_to_FC(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received) { this->Initialize_Connections__FC<struct Basic_indice_unit>(ptr_layer_it_received, ptr_previous_layer_it_received->ptr_array_basic_indice_units); }

void Neural_Network::Initialize_Connections__Basic_indice_unit_to_LSTM(struct Layer *const ptr_layer_it_received, struct Layer const *const ptr_previous_layer_it_received)
{
    if(ptr_layer_it_received->Use__Bidirectional())
    {
        this->Initialize_Connections__LSTM<struct Basic_indice_unit>(&ptr_layer_it_received->ptr_Bidirectional_Layer->forward_layer, ptr_previous_layer_it_received->ptr_array_basic_indice_units);
        this->Initialize_Connections__LSTM<struct Basic_indice_unit>(&ptr_layer_it_received->ptr_Bidirectional_Layer->backward_layer, ptr_previous_layer_it_received->ptr_array_basic_indice_units);
    }
    else { this->Initialize_Connections__LSTM<struct Basic_indice_unit>(ptr_layer_it_received, ptr_previous_layer_it_received->ptr_array_basic_indice_units); }
}
