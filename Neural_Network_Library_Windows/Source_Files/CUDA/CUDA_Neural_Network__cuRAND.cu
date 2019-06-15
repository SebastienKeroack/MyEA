#include <Tools/CUDA_Configuration.cuh>
#include <Tools/CUDA_Reallocate.cuh>
#include <Tools/CUDA_Zero_1D.cuh>
#include <CUDA/CUDA_cuRAND.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>
#include <CUDA/CUDA_Dataset_Manager.cuh>
#include <Math/CUDA_Mathematic.cuh>

#include <curand_kernel.h>

__device__ bool cuRAND_Bernoulli(float const probability_received, float const curand_uniform_received)
{ return((probability_received == 1.0f) ? true : ((probability_received == 0.0f) ? false : ((curand_uniform_received <= probability_received) ? true : false))); }
/*
{
    if(probability_received == 1.0f)
    {
        return(true);
    }
    else if(probability_received == 0.0f)
    {
        return(false);
    }
    else
    {
        float tmp_curand_uniform(curand_uniform(ptr_cuRAND_State_MTGP32_received));

        //PRINT_FORMAT("tmp_curand_uniform: %f" NEW_LINE, tmp_curand_uniform);

        ++tmp_count_total;

        if(tmp_curand_uniform <= probability_received)
        {
            return(true);
        }
        else
        {
            return(false);
        }
    }
}
*/

__global__ void kernel__CNeural_Network__Total_Blocks_cuRAND_MTGP32(int *const ptr_number_states_MTGP32_received,
                                                                                                                enum ENUM_TYPE_CURAND_GENERATOR const type_curand_generator_received,
                                                                                                                class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{ *ptr_number_states_MTGP32_received = ptr_CNeural_Network_received->Total_Blocks_cuRAND_MTGP32(type_curand_generator_received); }

__device__ int CUDA_Neural_Network::Total_Blocks_cuRAND_MTGP32(enum ENUM_TYPE_CURAND_GENERATOR const type_curand_generator_received)
{
    class CUDA_Device_Information *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

    size_t const tmp_maximum_states_usable(static_cast<size_t>(ceil(static_cast<double>(tmp_ptr_CUDA_Device->Get__Maximum_Threads()) / 256.0)));
    size_t tmp_number_blocks;

    switch(type_curand_generator_received)
    {
        case ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_WEIGHTS: tmp_number_blocks = static_cast<size_t>(ceil(static_cast<double>(this->total_weights_allocated) / 256.0)); break;
        case ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_BERNOULLI:
            for(size_t tmp_number_blocks = static_cast<size_t>(ceil(static_cast<double>(this->ptr_array_number_neurons_by_layer[0u] - 1u) / 256.0)),
                          tmp_number_blocks_in_layer = 0u,
                          i = 1u; i != this->total_layers; ++i)
            {
                tmp_number_blocks_in_layer = static_cast<size_t>(ceil(static_cast<double>(this->ptr_array_number_neurons_by_layer[i] - 1u) / 256.0));

                tmp_number_blocks = MyEA::Math::Maximum<size_t>(tmp_number_blocks, tmp_number_blocks_in_layer);
            }
                break;
        default: return(0);
    }
    
    tmp_number_blocks = MyEA::Math::Minimum<size_t>(tmp_number_blocks, tmp_maximum_states_usable); 
    
    if(tmp_number_blocks > (std::numeric_limits<int>::max)())
    {
        PRINT_FORMAT("%s: ERROR: Overflow conversion (%zu) to int (%d). At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 tmp_number_blocks,
                                 (std::numeric_limits<int>::max)(),
                                 __LINE__);
    }

    return(static_cast<int>(tmp_number_blocks));
}

__global__ void kernel__CNeural_Network__Initialize_cuRAND_MTGP32(int const size_received,
                                                                                                         enum ENUM_TYPE_CURAND_GENERATOR const type_curand_generator_received,
                                                                                                         struct curandStateMtgp32 *const ptr_curandStateMtgp32_received,
                                                                                                         class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    if(ptr_CNeural_Network_received->Initialize_cuRAND_MTGP32(size_received,
                                                                                               type_curand_generator_received,
                                                                                               ptr_curandStateMtgp32_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: From \"Initialize_cuRAND_MTGP32\"." NEW_LINE,
                                __FUNCTION__);
    }
}

__device__ bool CUDA_Neural_Network::Initialize_cuRAND_MTGP32(int const size_received,
                                                                                                     enum ENUM_TYPE_CURAND_GENERATOR const type_curand_generator_received,
                                                                                                     struct curandStateMtgp32 *const ptr_curandStateMtgp32_received)
{
    if(size_received == 0)
    {
        PRINT_FORMAT("%s: ERROR: Can not initialize cuRAND. Size of the array equal zero." NEW_LINE,
                                __FUNCTION__);

        return(false);
    }

    struct mtgp32_kernel_params *tmp_ptr_array_mtgp32_kernel_params_t;
    
    struct dim3 tmp_dim3_grid(1u, 1u, 1u),
                     tmp_dim3_block(1u, 1u, 1u);
    
    switch(type_curand_generator_received)
    {
        case ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_WEIGHTS:
            {
                // Allocate cuRAND State MTGP32 parametred.
                struct curandStateMtgp32 *tmp_ptr_array_cuRAND_State_MTGP32_weighted(new struct curandStateMtgp32[size_received]);
                if(tmp_ptr_array_cuRAND_State_MTGP32_weighted == nullptr)
                {
                    PRINT_FORMAT("%s: ERROR: Can not Allocate memory. new struct curandStateMtgp32(%u)[size_received(%d)]" NEW_LINE,
                                            __FUNCTION__,
                                            sizeof(struct curandStateMtgp32),
                                            size_received);

                    return(false);
                }
                this->ptr_array_cuRAND_State_MTGP32_weighted = tmp_ptr_array_cuRAND_State_MTGP32_weighted;
                // |END| Allocate cuRAND State MTGP32 parametred. |END|

                // Copy cuRAND State MTGP32 parametred.
                Memory::Copy_Loop<struct curandStateMtgp32>(ptr_curandStateMtgp32_received,
                                                                                        ptr_curandStateMtgp32_received + size_received,
                                                                                        this->ptr_array_cuRAND_State_MTGP32_weighted);
                // |END| Copy cuRAND State MTGP32 parametred. |END|
                
                // Allocate tmp_ptr_array_mtgp32_kernel_params_t.
                tmp_ptr_array_mtgp32_kernel_params_t = new struct mtgp32_kernel_params[size_received];
                if(tmp_ptr_array_mtgp32_kernel_params_t == nullptr)
                {
                    PRINT_FORMAT("%s: ERROR: Can not Allocate memory. new struct mtgp32_kernel_params(%u)[size_received(%d)]" NEW_LINE,
                                            __FUNCTION__,
                                            sizeof(struct mtgp32_kernel_params),
                                            size_received);

                    return(false);
                }
                // |END| Allocate tmp_ptr_array_mtgp32_kernel_params_t. |END|
                
                // Assign cuRAND State MTGP32 parametred variable.
                if(USE_PARALLEL && size_received >= warpSize)
                {
                    this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(static_cast<size_t>(size_received),
                                                                                                                                                           0_zu,
                                                                                                                                                           tmp_dim3_grid,
                                                                                                                                                           tmp_dim3_block);
                }

                cuRAND__Memcpy_cuRAND_State_MTGP32(size_received,
                                                                                    tmp_ptr_array_cuRAND_State_MTGP32_weighted,
                                                                                    ptr_curandStateMtgp32_received,
                                                                                    tmp_ptr_array_mtgp32_kernel_params_t,
                                                                                    &tmp_dim3_grid,
                                                                                    &tmp_dim3_block);

                this->number_cuRAND_State_MTGP32_weighted = size_received;
                // |END| Assign cuRAND State MTGP32 parametred variable. |END|
            }
                break;
        case ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_BERNOULLI:
            {
                // Allocate cuRAND State MTGP32 neuroyed.
                struct curandStateMtgp32 *tmp_ptr_array_cuRAND_State_MTGP32_neuroyed(new struct curandStateMtgp32[size_received]);
                if(tmp_ptr_array_cuRAND_State_MTGP32_neuroyed == nullptr)
                {
                    PRINT_FORMAT("%s: ERROR: Can not Allocate memory. new struct curandStateMtgp32(%u)[size_received(%d)]" NEW_LINE,
                                            __FUNCTION__,
                                            sizeof(struct curandStateMtgp32),
                                            size_received);

                    return(false);
                }
                this->ptr_array_cuRAND_State_MTGP32_neuroyed = tmp_ptr_array_cuRAND_State_MTGP32_neuroyed;
                // |END| Allocate cuRAND State MTGP32 neuroyed. |END|

                // Copy cuRAND State MTGP32 neuroyed.
                Memory::Copy_Loop<struct curandStateMtgp32>(ptr_curandStateMtgp32_received,
                                                                                        ptr_curandStateMtgp32_received + size_received,
                                                                                        this->ptr_array_cuRAND_State_MTGP32_neuroyed);
                // |END| Copy cuRAND State MTGP32 neuroyed. |END|

                // Allocate tmp_ptr_array_mtgp32_kernel_params_t.
                tmp_ptr_array_mtgp32_kernel_params_t = new struct mtgp32_kernel_params[size_received];
                if(tmp_ptr_array_mtgp32_kernel_params_t == nullptr)
                {
                    PRINT_FORMAT("%s: ERROR: Can not Allocate memory. new struct mtgp32_kernel_params(%u)[size_received(%d)]" NEW_LINE,
                                            __FUNCTION__,
                                            sizeof(struct mtgp32_kernel_params),
                                            size_received);

                    return(false);
                }
                // |END| Allocate tmp_ptr_array_mtgp32_kernel_params_t. |END|

                // Assign cuRAND State MTGP32 neuroyed variable.
                if(USE_PARALLEL && size_received >= warpSize)
                {
                    this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(static_cast<size_t>(size_received),
                                                                                                                                                           0_zu,
                                                                                                                                                           tmp_dim3_grid,
                                                                                                                                                           tmp_dim3_block);
                }

                cuRAND__Memcpy_cuRAND_State_MTGP32(size_received,
                                                                                    this->ptr_array_cuRAND_State_MTGP32_neuroyed,
                                                                                    ptr_curandStateMtgp32_received,
                                                                                    tmp_ptr_array_mtgp32_kernel_params_t,
                                                                                    &tmp_dim3_grid,
                                                                                    &tmp_dim3_block);

                this->number_cuRAND_State_MTGP32_neuroyed = size_received;
                // |END| Assign cuRAND State MTGP32 neuroyed variable. |END|
            }
                break;
        default: return(false);
    }

    return(true);
}

__host__ bool CUDA_Neural_Network::Initialize_cuRAND(size_t const seed_received)
{
    int tmp_number_states_MTGP32,
         *tmp_ptr_device_number_states_MTGP32(nullptr);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_number_states_MTGP32, sizeof(int)));
    
    // Weights
    kernel__CNeural_Network__Total_Blocks_cuRAND_MTGP32 <<< 1u, 1u >>> (tmp_ptr_device_number_states_MTGP32,
                                                                                                                      ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_WEIGHTS,
                                                                                                                      this);
    
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&tmp_number_states_MTGP32,
                                                    tmp_ptr_device_number_states_MTGP32,
                                                    sizeof(int),
                                                    cudaMemcpyKind::cudaMemcpyDeviceToHost));

    if(tmp_number_states_MTGP32 != 0)
    {
        struct mtgp32_kernel_params *tmp_ptr_mtgp32_kernel_params(NULL);

        struct curandStateMtgp32 *tmp_ptr_curandStateMtgp32_t(NULL);

        if(Allocate_cuRAND_MTGP32(tmp_number_states_MTGP32,
                                                    seed_received,
                                                    tmp_ptr_mtgp32_kernel_params,
                                                    tmp_ptr_curandStateMtgp32_t) == false)
        {
            PRINT_FORMAT("%s: ERROR: From \"Allocate_cuRAND_MTGP32\"." NEW_LINE, __FUNCTION__);
            
            CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));

            return(false);
        }
        
        kernel__CNeural_Network__Initialize_cuRAND_MTGP32 <<< 1u, 1u >>> (tmp_number_states_MTGP32,
                                                                                                                   ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_WEIGHTS,
                                                                                                                   tmp_ptr_curandStateMtgp32_t,
                                                                                                                   this);
            
    #if defined(COMPILE_DEBUG)
        CUDA__Check_Error();
    #endif

        Cleanup_cuRAND_MTGP32(tmp_ptr_mtgp32_kernel_params, tmp_ptr_curandStateMtgp32_t);
    }
    // |END| Weights |END|
    
    // Dropout bernoulli
    kernel__CNeural_Network__Total_Blocks_cuRAND_MTGP32 <<< 1u, 1u >>> (tmp_ptr_device_number_states_MTGP32,
                                                                                                                      ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_BERNOULLI,
                                                                                                                      this);
    
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&tmp_number_states_MTGP32,
                                                    tmp_ptr_device_number_states_MTGP32,
                                                    sizeof(int),
                                                    cudaMemcpyKind::cudaMemcpyDeviceToHost));

    if(tmp_number_states_MTGP32 != 0)
    {
        struct mtgp32_kernel_params *tmp_ptr_mtgp32_kernel_params(NULL);

        struct curandStateMtgp32 *tmp_ptr_curandStateMtgp32_t(NULL);

        if(Allocate_cuRAND_MTGP32(tmp_number_states_MTGP32,
                                                    seed_received,
                                                    tmp_ptr_mtgp32_kernel_params,
                                                    tmp_ptr_curandStateMtgp32_t) == false)
        {
            PRINT_FORMAT("%s: ERROR: From \"Allocate_cuRAND_MTGP32\"." NEW_LINE, __FUNCTION__);
            
            CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));

            return(false);
        }
        
        kernel__CNeural_Network__Initialize_cuRAND_MTGP32 <<< 1u, 1u >>> (tmp_number_states_MTGP32,
                                                                                                                   ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_BERNOULLI,
                                                                                                                   tmp_ptr_curandStateMtgp32_t,
                                                                                                                   this);
            
    #if defined(COMPILE_DEBUG)
        CUDA__Check_Error();
    #endif

        Cleanup_cuRAND_MTGP32(tmp_ptr_mtgp32_kernel_params, tmp_ptr_curandStateMtgp32_t);
    }
    // |END| Dropout bernoulli |END|
    
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));
    
    return(true);
}