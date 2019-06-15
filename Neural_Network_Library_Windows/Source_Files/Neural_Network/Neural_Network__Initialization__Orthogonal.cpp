#include "stdafx.hpp"

#include <Neural_Network/Neural_Network.hpp>

void Neural_Network::Initialization__Orthogonal(bool const pre_initialize_received, T_ const bias_received)
{
    struct Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct Layer *tmp_ptr_layer_it(this->ptr_array_layers + 1);

    // Loop though each layer.
    for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it)
    {
        // If the current layer is a pooling/residual layer, continue.
        if(tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_AVERAGE_POOLING
          ||
          tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_MAX_POOLING
          ||
          tmp_ptr_layer_it->type_layer == MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_RESIDUAL) { continue; }
        
        switch(tmp_ptr_layer_it->type_layer)
        {
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED:
                this->Initialize__Orthogonal(*tmp_ptr_layer_it->ptr_array_neuron_units->ptr_number_connections,
                                                        static_cast<size_t>(tmp_ptr_layer_it->ptr_last_neuron_unit - tmp_ptr_layer_it->ptr_array_neuron_units),
                                                        this->Initialization__Gain__Scale(*tmp_ptr_layer_it->ptr_array_AF_units->ptr_type_activation_function),
                                                        this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_first_connection_index);

                this->Initialize__Constant__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->Initialize__Orthogonal(*tmp_ptr_layer_it->ptr_array_neuron_units->ptr_number_connections,
                                                        static_cast<size_t>(tmp_ptr_layer_it->ptr_last_neuron_unit - tmp_ptr_layer_it->ptr_array_neuron_units),
                                                        this->Initialization__Gain__Scale(*tmp_ptr_layer_it->ptr_array_AF_Ind_recurrent_units->ptr_type_activation_function),
                                                        this->ptr_array_parameters + *tmp_ptr_layer_it->ptr_first_connection_index);

                this->Initialize__Uniform__AF_Ind_Recurrent(tmp_ptr_layer_it);

                this->Initialize__Constant__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            case MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_LSTM:
                // TODO: Intialize orthogonal LSTM.
                PRINT_FORMAT("%s: %s: ERROR: TODO: Intialize orthogonal LSTM." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__);
                //this->Initialize__Orthogonal__LSTM(tmp_ptr_layer_it);

                this->Initialize__Constant__LSTM__Bias(bias_received, tmp_ptr_layer_it);
                    break;
            default:
                PRINT_FORMAT("%s: %s: ERROR: Can not initialize weights in the layer %zu with (%u | %s) as the type layer. At line %d." NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         static_cast<size_t>(tmp_ptr_layer_it - this->ptr_array_layers),
                                         tmp_ptr_layer_it->type_layer,
                                         MyEA::Common::ENUM_TYPE_LAYER_NAME[tmp_ptr_layer_it->type_layer].c_str(),
                                         __LINE__);
                    break;
        }
    }

    if(pre_initialize_received == false)
    {
        // Independently recurrent neural network.
        if(this->number_recurrent_depth > 1_zu
          &&
          this->number_time_delays + 1_zu == this->number_recurrent_depth)
        { this->Initialize__Uniform__AF_Ind_Recurrent__Long_Term_Memory(); }

        if(this->ptr_array_derivatives_parameters != nullptr) { this->Clear_Training_Arrays(); }

        if(this->Use__Normalization()) { this->Clear__Parameter__Normalized_Unit(); }

        this->_initialized__weight = true;
        this->_type_weights_initializer = MyEA::Common::ENUM_TYPE_WEIGHTS_INITIALIZERS::TYPE_WEIGHTS_INITIALIZER_ORTHOGONAL;
    }
}

#if defined(COMPILE_FLOAT)
    typedef Eigen::MatrixXf MatrixXT;
#elif defined(COMPILE_DOUBLE)
    typedef Eigen::MatrixXd MatrixXT;
#endif

void diag_part(MatrixXT &A_received)
{
    MatrixXT tmp_diag(A_received.diagonal());

    A_received.setZero();

    size_t const tmp_square(A_received.rows() < A_received.cols() ? A_received.rows() : A_received.cols());

    for(size_t s(0_zu); s != tmp_square; ++s) { A_received(s, s) = tmp_diag(s); }
}

void Neural_Network::Initialize__Orthogonal(size_t const rows_received,
                                                               size_t const columns_received,
                                                               T_ const scale_received,
                                                               T_ *ptr_array_weights_received)
{
    this->Initialize__Gaussian(ptr_array_weights_received,
                                          ptr_array_weights_received + rows_received * columns_received,
                                          1_T);
    
    size_t const tmp_minimum_size(MyEA::Math::Minimum<size_t>(rows_received, columns_received)),
                       tmp_maximum_size(MyEA::Math::Maximum<size_t>(rows_received, columns_received));

#if defined(COMPILE_ADEPT)
    ST_ *tmp_ptr_array_parameters(new ST_[rows_received * columns_received]);
    for(size_t i(0_zu); i != rows_received * columns_received; ++i) { tmp_ptr_array_parameters[i] = ptr_array_weights_received[i].value(); }
    MatrixXT tmp_Matrix_Weights(Eigen::Map<MatrixXT>(tmp_ptr_array_parameters, tmp_maximum_size, tmp_minimum_size));
    delete[](tmp_ptr_array_parameters);
#else
    MatrixXT tmp_Matrix_Weights(Eigen::Map<MatrixXT>(ptr_array_weights_received, tmp_maximum_size, tmp_minimum_size));
#endif

    Eigen::ColPivHouseholderQR<Eigen::DenseBase<MatrixXT>::PlainMatrix> tmp_QR(tmp_Matrix_Weights.colPivHouseholderQr());
    MatrixXT tmp_dR(tmp_QR.matrixR()); diag_part(tmp_dR);
    MatrixXT tmp_W(tmp_QR.matrixQ() * tmp_dR.cwiseSign());
    if(rows_received < columns_received) { tmp_W.transposeInPlace(); }

    MatrixXT tmp_Wt(tmp_W.transpose());

    if(scale_received != 1_T) { tmp_Wt *= scale_received; }
    
#if defined(COMPILE_ADEPT)
    tmp_ptr_array_parameters = tmp_Wt.data();
    for(size_t i(0_zu); i != rows_received * columns_received; ++i) { ptr_array_weights_received[i] = tmp_ptr_array_parameters[i]; }
#else
    MEMCPY(ptr_array_weights_received,
                    tmp_Wt.data(),
                    rows_received * columns_received * sizeof(T_));
#endif
}
