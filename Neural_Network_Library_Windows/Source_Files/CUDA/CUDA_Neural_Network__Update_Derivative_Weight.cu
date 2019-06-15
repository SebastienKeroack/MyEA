#include <Tools/CUDA_Configuration.cuh>
#include <Math/CUDA_Mathematic.cuh>
#include <CUDA/CUDA_Multiply_1D.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

__device__ void CUDA_Neural_Network::Update_Derivative_Weight(size_t const batch_size_received, size_t const time_step_index_received) { this->FF__Update_Derivative_Weight(batch_size_received); }

__device__ void CUDA_Neural_Network::FF__Update_Derivative_Weight(size_t const batch_size_received)
{
    // By default the synchronized state is set to true.
    bool tmp_synchronized(true);

    struct CUDA_Layer const *const tmp_ptr_last_layer(this->ptr_last_layer);
    struct CUDA_Layer *tmp_ptr_previous_layer_it(this->ptr_array_layers),
                                            *tmp_ptr_layer_it(tmp_ptr_previous_layer_it + 1);
    
    // Variable to cache optimal size to launch dynamic parallelisme through the GPU.
    struct dim3 tmp_dim3_grid,
                     tmp_dim3_block;

    // If we can go into dynamic parallelisme, prepare the dimension kernel.
    if(batch_size_received >= warpSize)
    {
        size_t const tmp_batch_size_scale(MyEA::Math::Minimum<T_>(batch_size_received, this->number_threads));

        if(tmp_batch_size_scale == this->number_threads)
        {
            tmp_dim3_grid = this->ptr_array_dim3_grid[7u];
            tmp_dim3_block = this->ptr_array_dim3_block[7u];
        }
        else
        {
            this->ptr_array_layers->ptr_Class_Storage_Dim3_Batch->Get__Dim3_Dynamic_Parallelisme(tmp_batch_size_scale,
                                                                                                                                                 tmp_dim3_grid,
                                                                                                                                                 tmp_dim3_block,
                                                                                                                                                 this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
        }
    }
    
    if(this->use_Dropout)
    {
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                        ++tmp_ptr_previous_layer_it)
        {
            this->Update_Derivative_Weight__FC_to__Dropout(tmp_synchronized,
                                                                                           batch_size_received,
                                                                                           tmp_ptr_layer_it,
                                                                                           tmp_ptr_previous_layer_it,
                                                                                           &tmp_dim3_grid,
                                                                                           &tmp_dim3_block);
        }
    }
    else
    {
        for(; tmp_ptr_layer_it != tmp_ptr_last_layer; ++tmp_ptr_layer_it,
                                                                        ++tmp_ptr_previous_layer_it)
        {
            this->Update_Derivative_Weight__FC_to(tmp_synchronized,
                                                                            batch_size_received,
                                                                            tmp_ptr_layer_it,
                                                                            tmp_ptr_previous_layer_it,
                                                                            &tmp_dim3_grid,
                                                                            &tmp_dim3_block);
        }
    }
    
    // Synchronisation before using the output of the neural nework.
    CUDA__Device_Synchronise(tmp_synchronized, MyEA::Common::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);
}
    
__device__ void CUDA_Neural_Network::Update_Derivative_Weight__FC_to(bool &ref_synchronized_received,
                                                                                                          size_t const batch_size_received,
                                                                                                          struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                          struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                          struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                          struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    this->Update_Derivative_Weight__FC_to_FC(ref_synchronized_received,
                                                                               batch_size_received,
                                                                               ptr_layer_it_received,
                                                                               ptr_previous_layer_it_received,
                                                                               ptr_dim3_batch_size_grid_received,
                                                                               ptr_dim3_batch_size_block_received);
}
    
__device__ void CUDA_Neural_Network::Update_Derivative_Weight__FC_to__Dropout(bool &ref_synchronized_received,
                                                                                                                         size_t const batch_size_received,
                                                                                                                         struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                         struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                         struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                         struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    this->Update_Derivative_Weight__FC_to_FC__Dropout(ref_synchronized_received,
                                                                                              batch_size_received,
                                                                                              ptr_layer_it_received,
                                                                                              ptr_previous_layer_it_received,
                                                                                              ptr_dim3_batch_size_grid_received,
                                                                                              ptr_dim3_batch_size_block_received);
}
