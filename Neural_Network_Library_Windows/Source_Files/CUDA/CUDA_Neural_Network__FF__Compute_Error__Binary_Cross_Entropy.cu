#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

__device__ void CUDA_Neural_Network::FF__Compute__Error__Binary_Cross_Entropy(size_t const batch_size_received, T_ **const ptr_array_outputs_received)
{
    PRINT_FORMAT("%s: [FUNCTION DEPRECATED] Fix \"FF__Compute__Error__Binary_Cross_Entropy\"." NEW_LINE, __FUNCTION__);
}