#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

size_t Neural_Network::Prepare__Connections__FC(size_t const input_size_received, struct Layer *const ptr_layer_it_received)
{
    size_t tmp_allocated_connections(0u);

    struct Neuron_unit const *const tmp_ptr_last_neuron_unit(ptr_layer_it_received->ptr_last_neuron_unit);
    struct Neuron_unit *tmp_ptr_neuron_unit_it(ptr_layer_it_received->ptr_array_neuron_units);
    
    *ptr_layer_it_received->ptr_first_connection_index = this->total_weights;

    for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
    {
        *tmp_ptr_neuron_unit_it->ptr_first_connection_index = this->total_weights + tmp_allocated_connections;

        tmp_allocated_connections += input_size_received;

        *tmp_ptr_neuron_unit_it->ptr_last_connection_index = this->total_weights + tmp_allocated_connections;

        *tmp_ptr_neuron_unit_it->ptr_number_connections = input_size_received;
    }
    
    *ptr_layer_it_received->ptr_last_connection_index = this->total_weights + tmp_allocated_connections;

    return(tmp_allocated_connections);
}

size_t Neural_Network::Prepare__Connections__FC_Ind_RNN(size_t const input_size_received, struct Layer *const ptr_layer_it_received)
{
    size_t tmp_allocated_connections(0u);

    struct Neuron_unit const *const tmp_ptr_last_neuron_unit(ptr_layer_it_received->ptr_last_neuron_unit);
    struct Neuron_unit *tmp_ptr_neuron_unit_it;
    
    struct AF_Ind_recurrent_unit const *const tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit(ptr_layer_it_received->ptr_last_AF_Ind_recurrent_unit);
    struct AF_Ind_recurrent_unit *tmp_ptr_AF_Ind_recurrent_unit_it;
    
    *ptr_layer_it_received->ptr_first_connection_index = this->total_weights;

    // Forward connection(s).
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_received->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
    {
        *tmp_ptr_neuron_unit_it->ptr_first_connection_index = this->total_weights + tmp_allocated_connections;

        tmp_allocated_connections += input_size_received;

        *tmp_ptr_neuron_unit_it->ptr_last_connection_index = this->total_weights + tmp_allocated_connections;

        *tmp_ptr_neuron_unit_it->ptr_number_connections = input_size_received;
    }
    
    // Recurrent connection(s).
    for(tmp_ptr_AF_Ind_recurrent_unit_it = ptr_layer_it_received->ptr_array_AF_Ind_recurrent_units; tmp_ptr_AF_Ind_recurrent_unit_it != tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit; ++tmp_ptr_AF_Ind_recurrent_unit_it)
    {
        *tmp_ptr_AF_Ind_recurrent_unit_it->ptr_recurrent_connection_index = this->total_weights + tmp_allocated_connections;

        ++tmp_allocated_connections; // Self connection.
    }
    
    *ptr_layer_it_received->ptr_last_connection_index = this->total_weights + tmp_allocated_connections;

    return(tmp_allocated_connections);
}

size_t Neural_Network::Prepare__Connections__LSTM(size_t const input_size_received, struct Layer *const ptr_layer_it_received)
{
    /* If LSTM don't use recurrent projection layer and non-recurrent projection layer, use this equivalent.
            (nc * nc * 4 * di) + (ni * nc * 4 * di) + (nc * no) + (nc * 3 * di)
            nc = number of LSTM cells
            ni = number of input
            no = number of outputs
            di = number of layers

        if LSTM use projection layer, number of ptr_array_parameters obtain from this:
            (nc * nr * 4 * di) + (ni * nc * 4* di) + ((nr + np) * no) + (nc * nr * di) + (nc * 3 * di) + (nr * nc * 4 * (di - 1))
            nr = number of recurrent projection layer
            np = number of non-recurrent projection layer */

    size_t const tmp_number_connections_per_inputs_gates(input_size_received),
                       tmp_number_blocks(static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - ptr_layer_it_received->ptr_array_block_units)),
                       tmp_number_blocks_per_layer(tmp_number_blocks >> static_cast<size_t>(ptr_layer_it_received->Use__Bidirectional())),
                       tmp_number_cell_units(static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units)),
                       tmp_number_cell_units_per_layer(tmp_number_cell_units >> static_cast<size_t>(ptr_layer_it_received->Use__Bidirectional())),
                       tmp_number_cell_units_per_layer_per_block(tmp_number_cell_units_per_layer / tmp_number_blocks_per_layer);
                       // 3(forget, input, output gate) * nInputs * nBlocks +
                       //tmp_number_connections(3u * tmp_number_connections_per_inputs_gates * tmp_number_blocks +
                       // 1(block input) * nInputs * nCells
                       //                                         1u * tmp_number_connections_per_inputs_gates * tmp_number_cell_units +
                       // 4(forget, input, outputs gate, block input) * nCells * nCells
                       //                                         4u * tmp_number_cell_units * tmp_number_cell_units +
                       // 3(peepholes connections) * nCells
                       //                                         3u * tmp_number_cell_units);
    size_t tmp_allocated_connections(0_zu);
    
    struct Block_unit const *const tmp_ptr_last_block_unit(ptr_layer_it_received->ptr_last_block_unit);
    struct Block_unit *tmp_ptr_block_unit_it(ptr_layer_it_received->ptr_array_block_units);

    struct Cell_unit const *tmp_ptr_first_cell_unit,
                                    *tmp_ptr_last_cell_unit;
    struct Cell_unit *tmp_ptr_cell_unit_it;

    ptr_layer_it_received->type_activation = MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_SYMMETRIC;
    
    *ptr_layer_it_received->ptr_first_connection_index = this->total_weights;

    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        tmp_ptr_block_unit_it->activation_function_gate = MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID;
        tmp_ptr_block_unit_it->activation_function_io = MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_TANH;

        tmp_ptr_block_unit_it->first_index_connection = this->total_weights + tmp_allocated_connections;

        // [0] Cell input.
        for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
            tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
        {
            //    [1] Input, cell input.
            tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input = this->total_weights + tmp_allocated_connections;
            tmp_allocated_connections += tmp_number_connections_per_inputs_gates;
            tmp_ptr_cell_unit_it->last_index_feedforward_connection_cell_input = this->total_weights + tmp_allocated_connections;
            //    [1] |END| Input, cell input. |END|

            //    [1] Recurrent, cell input.
            tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input = this->total_weights + tmp_allocated_connections;
            tmp_allocated_connections += tmp_number_cell_units_per_layer;
            tmp_ptr_cell_unit_it->last_index_recurrent_connection_cell_input = this->total_weights + tmp_allocated_connections;
            //    [1] |END| Recurrent, cell input. |END|
        }
        // [0] |END| Cell input. |END|
        
        // [0] Input, gates.
        //    [1] Input gate.
        tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate = this->total_weights + tmp_allocated_connections;
        tmp_allocated_connections += tmp_number_connections_per_inputs_gates;
        tmp_ptr_block_unit_it->last_index_feedforward_connection_input_gate = this->total_weights + tmp_allocated_connections;
        //    [1] |END| Input gate. |END|
        
        //    [1] Forget gate.
        tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate = this->total_weights + tmp_allocated_connections;
        tmp_allocated_connections += tmp_number_connections_per_inputs_gates;
        tmp_ptr_block_unit_it->last_index_feedforward_connection_forget_gate = this->total_weights + tmp_allocated_connections;
        //    [1] |END| Forget gate. |END|
        
        //    [1] Output gate.
        tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate = this->total_weights + tmp_allocated_connections;
        tmp_allocated_connections += tmp_number_connections_per_inputs_gates;
        tmp_ptr_block_unit_it->last_index_feedforward_connection_output_gate = this->total_weights + tmp_allocated_connections;
        //    [1] |END| Output gate. |END|
        // [0] |END| Input, gates. |END|
        
        // [0] Recurrent, gates.
        //    [1] Input gate.
        tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate = this->total_weights + tmp_allocated_connections;
        tmp_allocated_connections += tmp_number_cell_units_per_layer;
        tmp_ptr_block_unit_it->last_index_recurrent_connection_input_gate = this->total_weights + tmp_allocated_connections;
        //    [1] |END| Input gate. |END|

        //    [1] Forget gate.
        tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate = this->total_weights + tmp_allocated_connections;
        tmp_allocated_connections += tmp_number_cell_units_per_layer;
        tmp_ptr_block_unit_it->last_index_recurrent_connection_forget_gate = this->total_weights + tmp_allocated_connections;
        //    [1] |END| Forget gate. |END|

        //    [1] Output gate.
        tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate = this->total_weights + tmp_allocated_connections;
        tmp_allocated_connections += tmp_number_cell_units_per_layer;
        tmp_ptr_block_unit_it->last_index_recurrent_connection_output_gate = this->total_weights + tmp_allocated_connections;
        //    [1] |END| Output gate. |END|
        // [0] |END| Recurrent, gates. |END|
        
    #ifndef NO_PEEPHOLE
        //    [1] Peepholes.
        tmp_ptr_block_unit_it->first_index_peephole_input_gate = this->total_weights + tmp_allocated_connections;
        tmp_allocated_connections += tmp_number_cell_units_per_layer_per_block;
        tmp_ptr_block_unit_it->last_index_peephole_input_gate = this->total_weights + tmp_allocated_connections;

        tmp_ptr_block_unit_it->first_index_peephole_forget_gate = this->total_weights + tmp_allocated_connections;
        tmp_allocated_connections += tmp_number_cell_units_per_layer_per_block;
        tmp_ptr_block_unit_it->last_index_peephole_forget_gate = this->total_weights + tmp_allocated_connections;

        tmp_ptr_block_unit_it->first_index_peephole_output_gate = this->total_weights + tmp_allocated_connections;
        tmp_allocated_connections += tmp_number_cell_units_per_layer_per_block;
        tmp_ptr_block_unit_it->last_index_peephole_output_gate = this->total_weights + tmp_allocated_connections;

        for(tmp_ptr_first_cell_unit = tmp_ptr_block_unit_it->ptr_array_cell_units,
            tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
            tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
        {
            tmp_ptr_cell_unit_it->index_peephole_input_gate = tmp_ptr_block_unit_it->first_index_peephole_input_gate + static_cast<size_t>(tmp_ptr_cell_unit_it - tmp_ptr_first_cell_unit);
            tmp_ptr_cell_unit_it->index_peephole_forget_gate = tmp_ptr_block_unit_it->first_index_peephole_forget_gate + static_cast<size_t>(tmp_ptr_cell_unit_it - tmp_ptr_first_cell_unit);
            tmp_ptr_cell_unit_it->index_peephole_output_gate = tmp_ptr_block_unit_it->first_index_peephole_output_gate + static_cast<size_t>(tmp_ptr_cell_unit_it - tmp_ptr_first_cell_unit);
        }
        //    [1] |END| Peepholes. |END|
    #endif

        tmp_ptr_block_unit_it->last_index_connection = this->total_weights + tmp_allocated_connections;

        tmp_ptr_block_unit_it->activation_function_gate = MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_SIGMOID;
        tmp_ptr_block_unit_it->activation_function_io = MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION::TYPE_NN_A_F_TANH;
    }
    
    *ptr_layer_it_received->ptr_last_connection_index = this->total_weights + tmp_allocated_connections;

    return(tmp_allocated_connections);
}

size_t Neural_Network::Prepare__Bias__FC(size_t const shift_index_received, struct Layer *const ptr_layer_it_received)
{
    size_t tmp_allocated_connections(0u);

    ptr_layer_it_received->first_bias_connection_index = shift_index_received + this->total_bias;
    tmp_allocated_connections += *ptr_layer_it_received->ptr_number_outputs;
    ptr_layer_it_received->last_bias_connection_index = shift_index_received + this->total_bias + tmp_allocated_connections;

    return(tmp_allocated_connections);
}

size_t Neural_Network::Prepare__Bias__LSTM(size_t const shift_index_received, struct Layer *const ptr_layer_it_received)
{
    size_t tmp_allocated_connections(0u);

    // Cell unit(s).
    ptr_layer_it_received->first_bias_connection_index = shift_index_received + this->total_bias;
    tmp_allocated_connections += static_cast<size_t>(ptr_layer_it_received->ptr_last_cell_unit - ptr_layer_it_received->ptr_array_cell_units);;
    ptr_layer_it_received->last_bias_connection_index = shift_index_received + this->total_bias + tmp_allocated_connections;
    // |END| Cell unit(s). |END|
    
    // Block unit(s).
    ptr_layer_it_received->first_bias_connection_index = shift_index_received + this->total_bias + tmp_allocated_connections;
    tmp_allocated_connections += 3_zu * static_cast<size_t>(ptr_layer_it_received->ptr_last_block_unit - ptr_layer_it_received->ptr_array_block_units);
    ptr_layer_it_received->last_bias_connection_index = shift_index_received + this->total_bias + tmp_allocated_connections;
    // |END| Block unit(s). |END|

    return(tmp_allocated_connections);
}
