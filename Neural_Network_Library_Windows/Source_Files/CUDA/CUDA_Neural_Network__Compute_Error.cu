#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

__device__ void CUDA_Neural_Network::Compute__Error(size_t const batch_size_received, T_ **const ptr_array_outputs_received)
{
    switch(this->type_network)
    {
        case MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_FEEDFORWARD:
            this->FF__Compute__Error(batch_size_received, ptr_array_outputs_received);
                break;
        default:
            PRINT_FORMAT("%s: ERROR: Undefined type network (%u)." NEW_LINE,
                                    __FUNCTION__,
                                    this->type_network);
                break;
    }
}

__device__ void CUDA_Neural_Network::FF__Compute__Error(size_t const batch_size_received, T_ **const ptr_array_outputs_received)
{
    switch(this->type_loss_function)
    {
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_ME:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_L1:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MAE:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_L2:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MSE:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_RMSE:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MAPE:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_SMAPE:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MASE_SEASONAL:
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_MASE_NON_SEASONAL:
            this->FF__Compute__Error__Standard(batch_size_received, ptr_array_outputs_received);
                break;
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_CROSS_ENTROPY:
            this->FF__Compute__Error__Binary_Cross_Entropy(batch_size_received, ptr_array_outputs_received);
                break;
        case MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_BIT: this->FF__Compute__Error__Bit_Fail(batch_size_received, ptr_array_outputs_received); break;
        default:
            PRINT_FORMAT("%s: ERROR: Undefined type loss function (%u)." NEW_LINE,
                                    __FUNCTION__,
                                    this->type_loss_function);
                break;
    }
}
