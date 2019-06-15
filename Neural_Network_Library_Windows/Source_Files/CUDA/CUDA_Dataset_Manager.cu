#include <Tools/CUDA_Memory_Initialize_1D.cuh>

#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Shuffle.cuh>
#include <CUDA/CUDA_cuRAND.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>
#include <CUDA/CUDA_Dataset_Manager.cuh>

#include <Neural_Network/Neural_Network.hpp>

#include <curand_kernel.h>

#include<Files/File.hpp>

#include <chrono>

template<typename T>
__host__ __device__ Dataset_device<T>::Dataset_device(void) { }

template<typename T>
__global__ void kernel__Dataset_device__Add_CUDA_Device(int const index_device_received,
                                                                                          struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received,
                                                                                          class Dataset_device<T> *const ptr_Dataset_device_received)
{ ptr_Dataset_device_received->Add_CUDA_Device(index_device_received, ptr_struct_cudaDeviceProp_received); }
    
template<typename T>
__device__ bool Dataset_device<T>::Add_CUDA_Device(int const index_device_received, struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received)
{
    if(this->p_ptr_Class_Device_Information_Array == nullptr)
    { this->p_ptr_Class_Device_Information_Array = new class CUDA_Device_Information_Array; }

    return(this->p_ptr_Class_Device_Information_Array->Push_Back(index_device_received, ptr_struct_cudaDeviceProp_received));
}

template<typename T>
__host__ bool Dataset_device<T>::Initialize_CUDA_Device(void)
{
    int tmp_index_device(0),
        tmp_number_CUDA_devices;
        
    struct cudaDeviceProp tmp_struct_cudaDeviceProp,
                                     *tmp_ptr_device_struct_cudaDeviceProp(NULL);

    CUDA__Safe_Call(cudaGetDeviceCount(&tmp_number_CUDA_devices));
        
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_struct_cudaDeviceProp, sizeof(struct cudaDeviceProp)));

    for(; tmp_index_device != tmp_number_CUDA_devices; ++tmp_index_device)
    {
        CUDA__Safe_Call(cudaGetDeviceProperties(&tmp_struct_cudaDeviceProp, tmp_index_device));

        CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_struct_cudaDeviceProp,
                                                        &tmp_struct_cudaDeviceProp,
                                                        sizeof(struct cudaDeviceProp),
                                                        cudaMemcpyKind::cudaMemcpyHostToDevice));

        kernel__Dataset_device__Add_CUDA_Device <<< 1u, 1u >>> (tmp_index_device,
                                                                                                         tmp_ptr_device_struct_cudaDeviceProp,
                                                                                                         this);
            
    #if defined(COMPILE_DEBUG)
        CUDA__Check_Error();
    #endif
    }

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_struct_cudaDeviceProp));

    return(true);
}

template<typename T>
__host__ Dataset_device<T>::Dataset_device(enum MyEA::Common::ENUM_TYPE_DATASET_FILE const type_data_file_read_received, std::string const &ref_path_received) { }

template<typename T>
__host__ __device__ Dataset_device<T>::~Dataset_device(void)
{ this->Deallocate(); }
    
template<class T>
__device__ Dataset_device<T>& Dataset_device<T>::operator=(class Dataset_device<T> const &ref_Dataset_received)
{
    if(&ref_Dataset_received != this) { this->Copy(ref_Dataset_received); }
        
    return(*this);
}
    
template<typename T>
__global__ void kernel__Dataset_device__Initialize(enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_gradient_descent_received, class Dataset_device<T> *const ptr_Dataset_device_received)
{ ptr_Dataset_device_received->Initialize(type_gradient_descent_received); }
template __global__ void kernel__Dataset_device__Initialize(enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const, class Dataset_device<T_> *const);

template<typename T>
__host__ __device__ bool Dataset_device<T>::Initialize(enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_gradient_descent_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__Dataset_device__Initialize<T> <<< 1u, 1u >>> (type_gradient_descent_received, this);
    
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    if(this->p_type_dataset_process == MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_NONE)
    {
        this->p_number_examples = 0u;
        this->p_number_recurrent_depth = 0u;
        this->p_number_inputs = 0u;
        this->p_number_outputs = 0u;

        this->p_ptr_array_inputs_array = nullptr;
        this->p_ptr_array_outputs_array = nullptr;

        this->p_type_dataset_process = type_gradient_descent_received;
        
        this->ptr_array_dim3_grid_batch = NULL;
        this->ptr_array_dim3_block_batch = NULL;

        this->ptr_array_dim3_grid_batch_fold = NULL;
        this->ptr_array_dim3_block_batch_fold = NULL;

        this->ptr_array_dim3_grid_shuffle = NULL;
        this->ptr_array_dim3_block_shuffle = NULL;
        
        this->p_ptr_Class_Device_Information_Array = nullptr;

        // Mini-Batch Stochastic
        this->use_shuffle = true;

        this->p_number_mini_batch = 0u;
        this->p_number_data_mini_batch = 0u;
        this->p_number_data_per_iteration = 0u;
        this->p_number_data_last_iteration = 0u;
        this->ptr_array_stochastic_index = nullptr;

        this->ptr_array_inputs_array_stochastic = nullptr;
        this->ptr_array_outputs_array_stochastic = nullptr;
        // - Mini-Batch Stochastic -

        // Cross Validation k-fold
        this->number_data_k_fold = 0u;
        this->number_k_fold = 0u;
        this->number_k_sub_fold = 0u;
        this->number_data_per_fold = 0u;
        this->number_data_training = 0u;
        this->number_data_validating = 0u;
        this->number_data_per_sub_iteration = 0u;
        this->number_data_last_sub_iteration = 0u;

        this->ptr_array_inputs_array_k_fold = nullptr;
        this->ptr_array_outputs_array_k_fold = nullptr;
        this->ptr_array_inputs_array_k_sub_fold = nullptr;
        this->ptr_array_outputs_array_k_sub_fold = nullptr;
            
        this->ptr_Validation_Dataset = nullptr;
        // - Cross Validation k-fold -

        // cuRAND.
        this->p_number_cuRAND_State_MTGP32_shuffle = 0u;
        this->p_number_blocks_shuffle = 0u;

        this->ptr_array_cuRAND_State_MTGP32_shuffle = nullptr;
        // |END| cuRAND. |END|
    }
    else { return(false); }

    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__Dataset_device__Initialize_Mini_Batch_Stochastic_Gradient_Descent(bool const use_shuffle_received,
                                                                                                                                              size_t const desired_number_data_per_mini_batch_received,
                                                                                                                                              size_t const number_mini_batch_maximum_received,
                                                                                                                                              class Dataset_device<T> *const ptr_Dataset_device_received)
{ 
    ptr_Dataset_device_received->Initialize_Mini_Batch_Stochastic_Gradient_Descent(use_shuffle_received,
                                                                                                                                 desired_number_data_per_mini_batch_received,
                                                                                                                                 number_mini_batch_maximum_received);
}
template __global__ void kernel__Dataset_device__Initialize_Mini_Batch_Stochastic_Gradient_Descent(bool const,
                                                                                                                                                            size_t const,
                                                                                                                                                            size_t const,
                                                                                                                                                            class Dataset_device<T_> *const);

template<typename T>
__host__ __device__ bool Dataset_device<T>::Initialize_Mini_Batch_Stochastic_Gradient_Descent(bool const use_shuffle_received,
                                                                                                                                                     size_t const desired_number_data_per_mini_batch_received,
                                                                                                                                                     size_t const number_mini_batch_maximum_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__Dataset_device__Initialize_Mini_Batch_Stochastic_Gradient_Descent<T> <<< 1u, 1u >>> (use_shuffle_received,
                                                                                                                                                          desired_number_data_per_mini_batch_received,
                                                                                                                                                          number_mini_batch_maximum_received,
                                                                                                                                                          this);
    
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
#if defined(COMPILE_DEBUG)
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: ERROR: No data available. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else
#endif
    if(desired_number_data_per_mini_batch_received == 0u)
    {
        PRINT_FORMAT("%s: ERROR: Desired number data per mini-batch equal zero. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    
    // 34875 / 128 = 272.46
    // 101 / 16 = 6.3125
    double const tmp_number_mini_batch(static_cast<double>(this->p_number_examples) / static_cast<double>(desired_number_data_per_mini_batch_received));
    
    // 272.46 = 272
    // 6.3125 = 6
    this->p_number_mini_batch = static_cast<size_t>(tmp_number_mini_batch);
    if(number_mini_batch_maximum_received != 0u) { this->p_number_mini_batch = this->p_number_mini_batch > number_mini_batch_maximum_received ? number_mini_batch_maximum_received : this->p_number_mini_batch; }
        
    // 128
    // 16
    this->p_number_data_per_iteration = desired_number_data_per_mini_batch_received;
    // 128 + (272.46 - 272) * 128 = 187
    // 16 + (6.3125 - 6) * 16 = 21
    this->p_number_data_last_iteration = this->p_number_data_per_iteration + static_cast<size_t>((tmp_number_mini_batch - static_cast<double>(this->p_number_mini_batch)) * static_cast<double>(this->p_number_data_per_iteration));
        
    this->p_number_data_mini_batch = this->p_number_data_last_iteration;

    this->ptr_array_inputs_array_stochastic = new T*[this->p_number_data_last_iteration];

    this->ptr_array_outputs_array_stochastic = new T*[this->p_number_data_last_iteration];
        
    this->use_shuffle = use_shuffle_received;

    this->ptr_array_stochastic_index = new size_t[this->p_number_examples];
    if(this->ptr_array_stochastic_index == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 this->p_number_examples * sizeof(size_t),
                                 __LINE__);

        return(false);
    }

    if(use_shuffle_received)
    {  this->Mini_Batch_Stochastic__Initialize_Shuffle(); }
    else
    {
        Memory::Memory_Initialize_Index<size_t>(this->p_number_examples,
                                                                               this->ptr_array_stochastic_index,
                                                                               this->ptr_array_dim3_grid_batch,
                                                                               this->ptr_array_dim3_block_batch);
    }

    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__Dataset_device__Initialize_Cross_Validation_K_Fold(bool const use_shuffle_received,
                                                                                                                     size_t const number_k_fold_received,
                                                                                                                     size_t const number_k_sub_fold_received,
                                                                                                                     class CUDA_Dataset_Manager<T> const *const ptr_CUDA_Dataset_Manager_received,
                                                                                                                     class Dataset_device<T> *const ptr_Dataset_device_received)
{
    ptr_Dataset_device_received->Initialize__Cross_Validation(use_shuffle_received,
                                                                                                        number_k_fold_received,
                                                                                                        number_k_sub_fold_received,
                                                                                                        ptr_CUDA_Dataset_Manager_received);
}
template __global__ void kernel__Dataset_device__Initialize_Cross_Validation_K_Fold(bool const,
                                                                                                                                   size_t const,
                                                                                                                                   size_t const,
                                                                                                                                   class CUDA_Dataset_Manager<T_> const *const,
                                                                                                                                   class Dataset_device<T_> *const);

template<typename T>
__host__ __device__ bool Dataset_device<T>::Initialize__Cross_Validation(bool const use_shuffle_received,
                                                                                                                            size_t const number_k_fold_received,
                                                                                                                            size_t const number_k_sub_fold_received,
                                                                                                                            class CUDA_Dataset_Manager<T> const *const ptr_CUDA_Dataset_Manager_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__Dataset_device__Initialize_Cross_Validation_K_Fold<T> <<< 1u, 1u >>> (use_shuffle_received,
                                                                                                                                 number_k_fold_received,
                                                                                                                                 number_k_sub_fold_received,
                                                                                                                                 ptr_CUDA_Dataset_Manager_received,
                                                                                                                                 this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
#if defined(COMPILE_DEBUG)
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: ERROR: Amount of data not available." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else
#endif
    if(number_k_fold_received < 2u)
    {
        PRINT_FORMAT("%s: ERROR: Not enough K-fold. Need to be at least at 2." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else if(ptr_CUDA_Dataset_Manager_received == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: \"ptr_CUDA_Dataset_Manager_received\" is a nullptr." NEW_LINE, __FUNCTION__);

        return(false);
    }

    class Dataset_device<T> *const tmp_ptr_Dataset_device_validation(ptr_CUDA_Dataset_Manager_received->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION));
        
    if(tmp_ptr_Dataset_device_validation == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: \"Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION)\" is a nullptr." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else if(this == tmp_ptr_Dataset_device_validation)
    {
        PRINT_FORMAT("%s: ERROR: Can not use cross-validation without a testing or validating set." NEW_LINE, __FUNCTION__);

        return(false);
    }
    
    size_t const tmp_number_data_TnV(this->Get__Total_Data() + tmp_ptr_Dataset_device_validation->Get__Total_Data()),
                                tmp_maximum_number_data(ptr_CUDA_Dataset_Manager_received->Get__Number_Examples()),
                                tmp_number_examples(MyEA::Math::Minimum<T>(tmp_number_data_TnV, tmp_maximum_number_data));
    
    if(tmp_number_examples == number_k_fold_received)
    {
        PRINT_FORMAT("%s: ERROR: K-fold can not be equal to the amount of data available." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else if(tmp_number_examples < number_k_fold_received)
    {
        PRINT_FORMAT("%s: ERROR: K-fold can not be larger than the number of data available." NEW_LINE, __FUNCTION__);

        return(false);
    }

    this->Copy(*ptr_CUDA_Dataset_Manager_received);
        
    this->p_number_examples = tmp_number_examples;
    
    this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(tmp_number_examples,
                                                                                                                                           0u,
                                                                                                                                           *this->ptr_array_dim3_grid_batch,
                                                                                                                                           *this->ptr_array_dim3_block_batch);
    
    size_t const tmp_number_data_per_fold(this->p_number_examples / number_k_fold_received);

    this->number_k_fold = number_k_fold_received;
        
    this->number_data_per_fold = tmp_number_data_per_fold;
    this->number_data_training = (number_k_fold_received - 1u) * tmp_number_data_per_fold;
    this->number_data_validating = this->p_number_examples - this->number_data_training;

    if(number_k_sub_fold_received > this->number_data_training)
    {
        PRINT_FORMAT("%s: ERROR: K-sub-fold (%u) > (%u) amount of training data." NEW_LINE,
                                    __FUNCTION__,
                                    number_k_sub_fold_received,
                                    this->number_data_training);

        return(false);
    }
        
    this->number_k_sub_fold = number_k_sub_fold_received == 0u ? number_k_fold_received - 1u : number_k_sub_fold_received;

    // 8 / 2 = 4
    // 31383 / 240 = 130.7625
    double const tmp_number_data_per_sub_fold(static_cast<double>(this->number_data_training) / static_cast<double>(this->number_k_sub_fold));

    // 4
    // 130
    this->number_data_per_sub_iteration = static_cast<size_t>(tmp_number_data_per_sub_fold);
        
    // 4 + (4 - 4) * 2 = 0
    // 130 + (130.7625 - 130) * 240 = 183
    this->number_data_last_sub_iteration = this->number_data_per_sub_iteration + static_cast<size_t>((tmp_number_data_per_sub_fold - static_cast<double>(this->number_data_per_sub_iteration)) * static_cast<double>(this->number_k_sub_fold));
        
    // 4 * 1 + 4 = 8
    // 130 * 239 + (130 + 183) = 31383
    this->number_data_k_fold = this->number_data_last_sub_iteration;
    
    this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(this->number_data_training,
                                                                                                                                           0u,
                                                                                                                                           *this->ptr_array_dim3_grid_batch_fold,
                                                                                                                                           *this->ptr_array_dim3_block_batch_fold);
    
    this->ptr_array_inputs_array_k_fold = new T*[this->number_data_training];

    this->ptr_array_outputs_array_k_fold = new T*[this->number_data_training];

    this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_k_fold;

    this->ptr_array_outputs_array_k_sub_fold = this->ptr_array_outputs_array_k_fold;
        
    this->use_shuffle = use_shuffle_received;
    
    this->ptr_array_stochastic_index = new size_t[this->p_number_examples];
    if(this->ptr_array_stochastic_index == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 this->p_number_examples * sizeof(size_t),
                                 __LINE__);

        return(false);
    }

    if(use_shuffle_received)
    { this->Cross_Validation_K_Fold__Initialize_Shuffle(); }
    else
    {
        Memory::Memory_Initialize_Index<size_t>(this->p_number_examples,
                                                                               this->ptr_array_stochastic_index,
                                                                               this->ptr_array_dim3_grid_batch,
                                                                               this->ptr_array_dim3_block_batch);
    }

    this->ptr_Validation_Dataset = tmp_ptr_Dataset_device_validation;
    
    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__Dataset_device__Initialize_Cross_Validation_K_Fold(class CUDA_Dataset_Manager<T> const *const ptr_CUDA_Dataset_Manager_received, class Dataset_device<T> *const ptr_Dataset_device_received)
{ ptr_Dataset_device_received->Initialize__Cross_Validation(ptr_CUDA_Dataset_Manager_received); }
template __global__ void kernel__Dataset_device__Initialize_Cross_Validation_K_Fold(class CUDA_Dataset_Manager<T_> const *const, class Dataset_device<T_> *const);

template<typename T>
__host__ __device__ bool Dataset_device<T>::Initialize__Cross_Validation(class CUDA_Dataset_Manager<T> const *const ptr_CUDA_Dataset_Manager_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__Dataset_device__Initialize_Cross_Validation_K_Fold<T> <<< 1u, 1u >>> (ptr_CUDA_Dataset_Manager_received, this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
#if defined(COMPILE_DEBUG)
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: ERROR: Amount of data not available." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else
#endif
    if(ptr_CUDA_Dataset_Manager_received == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: \"ptr_Dataset_Manager_received\" is a nullptr." NEW_LINE, __FUNCTION__);

        return(false);
    }
        
    class Dataset_device<T> *const tmp_ptr_Dataset_Cross_Validation_training(ptr_CUDA_Dataset_Manager_received->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));

    if(tmp_ptr_Dataset_Cross_Validation_training == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: \"Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING)\" is a nullptr." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else if(this == tmp_ptr_Dataset_Cross_Validation_training)
    {
        PRINT_FORMAT("%s: ERROR: Can not use cross-validation without a testing or validating set." NEW_LINE, __FUNCTION__);

        return(false);
    }

    this->Copy(*ptr_CUDA_Dataset_Manager_received);

    this->p_number_examples = tmp_ptr_Dataset_Cross_Validation_training->Get__Total_Data();
    
    this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(this->p_number_examples,
                                                                                                                                           0u,
                                                                                                                                           *this->ptr_array_dim3_grid_batch,
                                                                                                                                           *this->ptr_array_dim3_block_batch);
    
    this->number_k_fold = tmp_ptr_Dataset_Cross_Validation_training->Get__Number_CV_K_Fold();
    this->number_data_per_fold = tmp_ptr_Dataset_Cross_Validation_training->Get__Number_CV_Data_Per_Fold();
    this->number_data_training = tmp_ptr_Dataset_Cross_Validation_training->Get__Number_CV_Data_Training();
    this->number_data_validating = tmp_ptr_Dataset_Cross_Validation_training->Get__Number_CV_Data_Validating();

    this->number_k_sub_fold = tmp_ptr_Dataset_Cross_Validation_training->Get__Number_CV_K_Sub_Fold();
    this->number_data_per_sub_iteration = tmp_ptr_Dataset_Cross_Validation_training->Get__Number_CV_Data_Per_Sub_Iteration();
    this->number_data_last_sub_iteration = tmp_ptr_Dataset_Cross_Validation_training->Get__Number_CV_Data_Last_Sub_Iteration();
    
    this->number_data_k_fold = this->number_data_validating;
    
    this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(this->number_data_validating,
                                                                                                                                           0u,
                                                                                                                                           *this->ptr_array_dim3_grid_batch_fold,
                                                                                                                                           *this->ptr_array_dim3_block_batch_fold);

    this->ptr_array_inputs_array_k_fold = new T*[this->number_data_validating];

    this->ptr_array_outputs_array_k_fold = new T*[this->number_data_validating];
    
    this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_k_fold;

    this->ptr_array_outputs_array_k_sub_fold = this->ptr_array_outputs_array_k_fold;

    return(true);
#endif
}

template<typename T>
__device__ void Dataset_device<T>::Mini_Batch_Stochastic__Reset(void)
{
    this->p_number_data_mini_batch = this->p_number_data_last_iteration;
}

template<typename T>
__global__ void kernel__Two_Memory_2D_Copy_Stochastic(size_t const *const ptr_array_stochastic_index_received,
                                                                                  T **const ptr_array_destination_0_received,
                                                                                  T **const ptr_array_destination_1_received,
                                                                                  T **const ptr_array_source_0_received,
                                                                                  T **const ptr_array_source_1_received)
{
    size_t const tmp_thread_index(blockIdx.x * blockDim.x + threadIdx.x);

    ptr_array_destination_0_received[tmp_thread_index] = ptr_array_source_0_received[ptr_array_stochastic_index_received[tmp_thread_index]];
    ptr_array_destination_1_received[tmp_thread_index] = ptr_array_source_1_received[ptr_array_stochastic_index_received[tmp_thread_index]];
}

template<typename T>
__global__ void kernel__Two_Memory_2D_Copy_Stochastic(size_t const size_received,
                                                                                  size_t const *const ptr_array_stochastic_index_received,
                                                                                  T **const ptr_array_destination_0_received,
                                                                                  T **const ptr_array_destination_1_received,
                                                                                  T **const ptr_array_source_0_received,
                                                                                  T **const ptr_array_source_1_received)
{
    size_t const tmp_thread_index(blockIdx.x * blockDim.x + threadIdx.x);

    if(tmp_thread_index < size_received)
    {
        ptr_array_destination_0_received[tmp_thread_index] = ptr_array_source_0_received[ptr_array_stochastic_index_received[tmp_thread_index]];
        ptr_array_destination_1_received[tmp_thread_index] = ptr_array_source_1_received[ptr_array_stochastic_index_received[tmp_thread_index]];
    }
}

template<typename T>
__global__ void kernel_while__Two_Memory_2D_Copy_Stochastic(size_t const size_received,
                                                                                           size_t const *const ptr_array_stochastic_index_received,
                                                                                           T **const ptr_array_destination_0_received,
                                                                                           T **const ptr_array_destination_1_received,
                                                                                           T **const ptr_array_source_0_received,
                                                                                           T **const ptr_array_source_1_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_index(blockIdx.x * blockDim.x + threadIdx.x);

    do
    {
        ptr_array_destination_0_received[tmp_thread_index] = ptr_array_source_0_received[ptr_array_stochastic_index_received[tmp_thread_index]];
        ptr_array_destination_1_received[tmp_thread_index] = ptr_array_source_1_received[ptr_array_stochastic_index_received[tmp_thread_index]];

        tmp_thread_index += tmp_grid_stride;
    } while(tmp_thread_index < size_received);
}

template<typename T>
__device__ void Two_Memory_2D_Copy_Stochastic(size_t const size_received,
                                                                           size_t const *const ptr_array_stochastic_index_received,
                                                                           T **const ptr_array_destination_0_received,
                                                                           T **const ptr_array_destination_1_received,
                                                                           T **const ptr_array_source_0_received,
                                                                           T **const ptr_array_source_1_received,
                                                                           struct dim3 const *const ptr_dimension_grid_received,
                                                                           struct dim3 const *const ptr_dimension_block_received)
{
    if(USE_PARALLEL && size_received >= warpSize)
    {
        LAUNCH_KERNEL_POINTER_1D(Two_Memory_2D_Copy_Stochastic<T>,
                                                          ptr_dimension_grid_received,
                                                          ptr_dimension_block_received,
                                                          0_zu,
                                                          size_received,
                                                          ptr_array_stochastic_index_received,
                                                          ptr_array_destination_0_received,
                                                          ptr_array_destination_1_received,
                                                          ptr_array_source_0_received,
                                                          ptr_array_source_1_received)
    }
    else
    {
        for(size_t i(0_zu); i != size_received; ++i)
        {
            ptr_array_destination_0_received[i] = ptr_array_source_0_received[ptr_array_stochastic_index_received[i]];
            
            ptr_array_destination_1_received[i] = ptr_array_source_1_received[ptr_array_stochastic_index_received[i]];
        }
    }
}

template<typename T>
__device__ bool Dataset_device<T>::Mini_Batch_Stochastic__Increment_Mini_Batch(size_t const mini_batch_iteration_received)
{
    size_t const tmp_data_per_mini_batch(mini_batch_iteration_received + 1u != this->p_number_mini_batch ? this->p_number_data_per_iteration : this->p_number_data_last_iteration);
    size_t tmp_last_element_start_index,
                        tmp_last_element_end_index;

    tmp_last_element_start_index = mini_batch_iteration_received * this->p_number_data_per_iteration;
    tmp_last_element_end_index = tmp_last_element_start_index + tmp_data_per_mini_batch;

    // Index global inputs to local inputs.
    Two_Memory_2D_Copy_Stochastic<T>(tmp_last_element_end_index - tmp_last_element_start_index,
                                                              this->ptr_array_stochastic_index + tmp_last_element_start_index,
                                                              this->ptr_array_inputs_array_stochastic,
                                                              this->ptr_array_outputs_array_stochastic,
                                                              this->p_ptr_array_inputs_array,
                                                              this->p_ptr_array_outputs_array,
                                                              this->ptr_array_dim3_grid_batch,
                                                              this->ptr_array_dim3_block_batch);
    // |END| Index global inputs to local inputs. |END|

    this->p_number_data_mini_batch = tmp_data_per_mini_batch;
    
    // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic" Function.
    // => Synchronisation before using the training mini-batch.
    if(tmp_last_element_end_index - tmp_last_element_start_index >= warpSize)
    { CUDA__Check_Error(); }

    return(true);
}

template<typename T>
__device__ T_ Dataset_device<T>::Test_Epoch_Cross_Validation_K_Fold(class CUDA_Neural_Network *ptr_CNeural_Network_received)
{
    ptr_CNeural_Network_received->Reset__Loss();

    ptr_CNeural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_INFERENCE;
    
    size_t const tmp_number_examples(this->number_data_k_fold),
                               tmp_maximum_batch_size(ptr_CNeural_Network_received->batch_size),
                               tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_batch_size,
                      i;

    for(i = 0u; i != tmp_number_batchs; ++i)
    {
        tmp_batch_size = i + 1u != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - i * tmp_maximum_batch_size;

        ptr_CNeural_Network_received->Forward_Pass(tmp_batch_size, this->ptr_array_inputs_array_k_fold + i * tmp_maximum_batch_size);
        
        ptr_CNeural_Network_received->Test(tmp_batch_size, this->ptr_array_outputs_array_k_fold + i * tmp_maximum_batch_size);
    }

    *ptr_CNeural_Network_received->ptr_array_number_loss = tmp_number_examples * this->Get__Number_Outputs();
    ptr_CNeural_Network_received->number_accuracy_trial = tmp_number_examples * this->Get__Number_Outputs();

    // Synchronize the computed error before merging between threads.
    CUDA__Check_Error();
        
    ptr_CNeural_Network_received->Merge__Post__Training();

    ptr_CNeural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING;

    return(ptr_CNeural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
}

template<typename T>
__device__ void Dataset_device<T>::Cross_Validation_K_Fold__Initialize_Shuffle(void)
{
    class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

    // Tree shift shuffle.
    if(this->ptr_array_dim3_grid_shuffle == NULL)
    {
        struct dim3 *tmp_ptr_array_dim3_grid_shuffle(static_cast<struct dim3*>(malloc(sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_grid_shuffle == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(struct dim3));

            return;
        }
        *tmp_ptr_array_dim3_grid_shuffle = dim3(1u, 1u, 1u);
        this->ptr_array_dim3_grid_shuffle = tmp_ptr_array_dim3_grid_shuffle;
    }
    
    if(this->ptr_array_dim3_block_shuffle == NULL)
    {
        struct dim3 *tmp_ptr_array_dim3_block_shuffle(static_cast<struct dim3*>(malloc(sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_block_shuffle == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(struct dim3));

            return;
        }
        *tmp_ptr_array_dim3_block_shuffle = dim3(1u, 1u, 1u);
        this->ptr_array_dim3_block_shuffle = tmp_ptr_array_dim3_block_shuffle;
    }

    this->p_number_blocks_shuffle = static_cast<size_t>(ceil(static_cast<double>(this->p_number_examples) / static_cast<double>(this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Get__Warp_Size())));

    tmp_ptr_CUDA_Device->Grid_Block_cuRAND_1Dimensions(this->p_number_blocks_shuffle,
                                                                                            0u,
                                                                                            this->ptr_array_dim3_grid_shuffle[0u],
                                                                                            this->ptr_array_dim3_block_shuffle[0u]);
    // |END| Tree shift shuffle. |END|
}

template<typename T>
__device__ void Dataset_device<T>::Cross_Validation_K_Fold__Shuffle(void)
{
    Memory::Memory_Initialize_Index_Shift<size_t>(this->p_number_examples,
                                                                                   curand(this->ptr_array_cuRAND_State_MTGP32_shuffle) % this->p_number_examples,
                                                                                   this->ptr_array_stochastic_index,
                                                                                   this->ptr_array_dim3_grid_batch,
                                                                                   this->ptr_array_dim3_block_batch);

    Shuffle::Tree_Shuffle<size_t>(this->p_number_blocks_shuffle,
                                                        this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Get__Warp_Size(),
                                                        this->p_number_examples,
                                                        this->ptr_array_stochastic_index,
                                                        this->ptr_array_cuRAND_State_MTGP32_shuffle,
                                                        this->ptr_array_dim3_grid_shuffle,
                                                        this->ptr_array_dim3_block_shuffle);
}

template<typename T>
__device__ void Dataset_device<T>::Cross_Validation_K_Fold__Reset(void)
{
    this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_k_fold;

    this->ptr_array_outputs_array_k_sub_fold = this->ptr_array_outputs_array_k_fold;

    this->number_data_k_fold = this->number_data_last_sub_iteration;
}

template<typename T>
__device__ bool Dataset_device<T>::Cross_Validation_K_Fold__Increment_Fold(size_t const fold_received)
{
#if defined(COMPILE_DEBUG)
    if(this->p_number_examples == 0_zu)
    {
        PRINT_FORMAT("%s: ERROR: Amount of data not available." NEW_LINE, __FUNCTION__);

        return(false);
    }
#endif

    bool tmp_synchronized(true);

    if(fold_received >= this->number_k_fold) { return(false); }

    size_t const tmp_number_data_training_per_fold(this->number_data_per_fold),
                                tmp_number_data_validating(this->ptr_Validation_Dataset->number_data_validating),
                                tmp_validating_index_start(fold_received * tmp_number_data_training_per_fold),
                                tmp_validating_index_end(tmp_validating_index_start + tmp_number_data_validating);
    size_t *tmp_ptr_array_stochastic_index(this->ptr_array_stochastic_index);
    
    if(tmp_validating_index_start == 0u) // First iteration.
    {
        // Validation sample.
        // (0, 1, 2)   [3, 4, 5   6, 7, 8   9, 10, 11]
        Two_Memory_2D_Copy_Stochastic<T>(tmp_number_data_validating,
                                                                  tmp_ptr_array_stochastic_index,
                                                                  this->ptr_Validation_Dataset->ptr_array_inputs_array_k_fold,
                                                                  this->ptr_Validation_Dataset->ptr_array_outputs_array_k_fold,
                                                                  this->p_ptr_array_inputs_array,
                                                                  this->p_ptr_array_outputs_array,
                                                                  this->ptr_Validation_Dataset->ptr_array_dim3_grid_batch_fold,
                                                                  this->ptr_Validation_Dataset->ptr_array_dim3_block_batch_fold);
        
        // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic" Function.
        if(tmp_number_data_validating >= warpSize) { tmp_synchronized = false; }
        // |END| Validation sample. |END|

        // Training sample.
        tmp_ptr_array_stochastic_index += tmp_validating_index_end;
        
        // (0, 1, 2)   [3, 4, 5   6, 7, 8   9, 10, 11]
        Two_Memory_2D_Copy_Stochastic<T>(this->number_data_training,
                                                                  tmp_ptr_array_stochastic_index,
                                                                  this->ptr_array_inputs_array_k_fold,
                                                                  this->ptr_array_outputs_array_k_fold,
                                                                  this->p_ptr_array_inputs_array,
                                                                  this->p_ptr_array_outputs_array,
                                                                  this->ptr_array_dim3_grid_batch_fold,
                                                                  this->ptr_array_dim3_block_batch_fold);
        
        // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic" Function.
        if(this->number_data_training >= warpSize) { tmp_synchronized = false; }
        // |END| Training sample. |END|
    }
    else if(tmp_validating_index_end == this->p_number_examples) // Last iteration.
    {
        // Training sample.
        // [0, 1, 2   3, 4, 5   6, 7, 8]   (9, 10, 11)
        Two_Memory_2D_Copy_Stochastic<T>(this->number_data_training,
                                                                tmp_ptr_array_stochastic_index,
                                                                this->ptr_array_inputs_array_k_fold,
                                                                this->ptr_array_outputs_array_k_fold,
                                                                this->p_ptr_array_inputs_array,
                                                                this->p_ptr_array_outputs_array,
                                                                this->ptr_array_dim3_grid_batch_fold,
                                                                this->ptr_array_dim3_block_batch_fold);
        
        // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic" Function.
        if(this->number_data_training >= warpSize) { tmp_synchronized = false; }
        // |END| Training sample. |END|

        // Validation sample.
        tmp_ptr_array_stochastic_index += tmp_validating_index_start;

        // [0, 1, 2   3, 4, 5   6, 7, 8]   (9, 10, 11)
        Two_Memory_2D_Copy_Stochastic<T>(tmp_number_data_validating,
                                                                tmp_ptr_array_stochastic_index,
                                                                this->ptr_Validation_Dataset->ptr_array_inputs_array_k_fold,
                                                                this->ptr_Validation_Dataset->ptr_array_outputs_array_k_fold,
                                                                this->p_ptr_array_inputs_array,
                                                                this->p_ptr_array_outputs_array,
                                                                this->ptr_Validation_Dataset->ptr_array_dim3_grid_batch_fold,
                                                                this->ptr_Validation_Dataset->ptr_array_dim3_block_batch_fold);
        
        // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic" Function.
        if(tmp_number_data_validating >= warpSize) { tmp_synchronized = false; }
        // |END| Validation sample. |END|
    }
    else // The remaining iterations.
    {
        // Training sample.
        // [0, 1, 2]   (3, 4, 5)   [6, 7, 8   9, 10, 11]
        Two_Memory_2D_Copy_Stochastic<T>(tmp_validating_index_start,
                                                                tmp_ptr_array_stochastic_index,
                                                                this->ptr_array_inputs_array_k_fold,
                                                                this->ptr_array_outputs_array_k_fold,
                                                                this->p_ptr_array_inputs_array,
                                                                this->p_ptr_array_outputs_array,
                                                                this->ptr_array_dim3_grid_batch_fold,
                                                                this->ptr_array_dim3_block_batch_fold);
        
        // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic" Function.
        if(tmp_validating_index_start >= warpSize) { tmp_synchronized = false; }
        // |END| Training sample. |END|

        // Validation sample.
        tmp_ptr_array_stochastic_index += tmp_validating_index_start;

        // [0, 1, 2]   (3, 4, 5)   [6, 7, 8   9, 10, 11]
        Two_Memory_2D_Copy_Stochastic<T>(tmp_number_data_validating,
                                                                tmp_ptr_array_stochastic_index,
                                                                this->ptr_Validation_Dataset->ptr_array_inputs_array_k_fold,
                                                                this->ptr_Validation_Dataset->ptr_array_outputs_array_k_fold,
                                                                this->p_ptr_array_inputs_array,
                                                                this->p_ptr_array_outputs_array,
                                                                this->ptr_Validation_Dataset->ptr_array_dim3_grid_batch_fold,
                                                                this->ptr_Validation_Dataset->ptr_array_dim3_block_batch_fold);
        
        // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic" Function.
        if(tmp_number_data_validating >= warpSize) { tmp_synchronized = false; }
        // |END| Validation sample. |END|

        // Training sample.
        tmp_ptr_array_stochastic_index = this->ptr_array_stochastic_index + tmp_number_data_validating;

        // [0, 1, 2]   (3, 4, 5)   [6, 7, 8   9, 10, 11]
        Two_Memory_2D_Copy_Stochastic<T>(this->number_data_training - tmp_validating_index_start,
                                                                tmp_ptr_array_stochastic_index + tmp_validating_index_start,
                                                                this->ptr_array_inputs_array_k_fold + tmp_validating_index_start,
                                                                this->ptr_array_outputs_array_k_fold + tmp_validating_index_start,
                                                                this->p_ptr_array_inputs_array,
                                                                this->p_ptr_array_outputs_array,
                                                                this->ptr_array_dim3_grid_batch_fold,
                                                                this->ptr_array_dim3_block_batch_fold);
        
        // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic" Function.
        if(this->number_data_training - tmp_validating_index_start >= warpSize) { tmp_synchronized = false; }
        // |END| Training sample. |END|
    }
    
    // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic" Function.
    // => Synchronisation before using the training fold batch.
    if(tmp_synchronized == false) { CUDA__Check_Error(); }

    return(true);
}
    
template<typename T>
__device__ bool Dataset_device<T>::Cross_Validation_K_Fold__Increment_Sub_Fold(size_t const sub_fold_received)
{
    if(this->number_k_sub_fold == 1u) { return(true); }
    else if(sub_fold_received >= this->number_k_sub_fold) { return(false); }

    size_t const tmp_data_per_sub_fold(sub_fold_received + 1u != this->number_k_sub_fold ? this->number_data_per_sub_iteration : this->number_data_last_sub_iteration);
    
    this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_k_fold + sub_fold_received * this->number_data_per_sub_iteration;

    this->ptr_array_outputs_array_k_sub_fold = this->ptr_array_outputs_array_k_fold + sub_fold_received * this->number_data_per_sub_iteration;

    this->number_data_k_fold = tmp_data_per_sub_fold;

    return(true);
}

template<typename T>
__global__ void kernel__Dataset_device__Set__Type_Gradient_Descent(enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_gradient_descent_received, class Dataset_device<T> *const ptr_Dataset_device_received)
{ ptr_Dataset_device_received->Set__Type_Gradient_Descent(type_gradient_descent_received); }
template __global__ void kernel__Dataset_device__Set__Type_Gradient_Descent(enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const, class Dataset_device<T_> *const);

template<typename T>
__host__ __device__ bool Dataset_device<T>::Set__Type_Gradient_Descent(enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_gradient_descent_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__Dataset_device__Set__Type_Gradient_Descent<T> <<< 1u, 1u >>> (type_gradient_descent_received, this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    this->p_type_dataset_process = type_gradient_descent_received;

    return(true);
#endif
}
    
template<typename T>
__device__ void Dataset_device<T>::Copy(class Dataset_device<T> const &ref_Dataset_received)
{
    if(this->_reference == false)
    { this->Deallocate(); }

    this->p_number_examples = ref_Dataset_received.p_number_examples;
    this->p_number_inputs = ref_Dataset_received.p_number_inputs;
    this->p_number_outputs = ref_Dataset_received.p_number_outputs;
    this->p_number_recurrent_depth = ref_Dataset_received.p_number_recurrent_depth;
        
    this->p_ptr_array_inputs_array = ref_Dataset_received.p_ptr_array_inputs_array;
    this->p_ptr_array_outputs_array = ref_Dataset_received.p_ptr_array_outputs_array;

    this->_reference = true;
}
    
template<typename T>
__device__ void Dataset_device<T>::Copy(class CUDA_Dataset_Manager<T> const &ref_Dataset_Manager_received)
{
    if(this->_reference == false)
    { this->Deallocate(); }

    this->p_number_examples = ref_Dataset_Manager_received.Get__Number_Examples();
    this->p_number_inputs = ref_Dataset_Manager_received.Get__Number_Inputs();
    this->p_number_outputs = ref_Dataset_Manager_received.Get__Number_Outputs();
    this->p_number_recurrent_depth = ref_Dataset_Manager_received.Get__Number_Recurrent_Depth();
        
    this->p_ptr_array_inputs_array = const_cast<T**>(ref_Dataset_Manager_received.Get__Input_Array());
    this->p_ptr_array_outputs_array = const_cast<T**>(ref_Dataset_Manager_received.Get__Output_Array());
        
    this->_reference = true;
}

template<typename T>
__device__ bool Dataset_device<T>::Allocate_Dim3(void)
{
    // Allocate dim3 batch.
    if(this->ptr_array_dim3_grid_batch == NULL)
    {
        struct dim3 *tmp_ptr_array_dim3_grid_batch(static_cast<struct dim3*>(malloc(sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_grid_batch == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(struct dim3));

            return(false);
        }
        *tmp_ptr_array_dim3_grid_batch = struct dim3(1u, 1u, 1u);
        this->ptr_array_dim3_grid_batch = tmp_ptr_array_dim3_grid_batch;
    }
    
    if(this->ptr_array_dim3_block_batch == NULL)
    {
        struct dim3 *tmp_ptr_array_dim3_block_batch(static_cast<struct dim3*>(malloc(sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_block_batch == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(struct dim3));

            return(false);
        }
        *tmp_ptr_array_dim3_block_batch = struct dim3(1u, 1u, 1u);
        this->ptr_array_dim3_block_batch = tmp_ptr_array_dim3_block_batch;
    }
    // |END| Allocate dim3 batch. |END|
    
    // Allocate dim3 batch.
    if(this->ptr_array_dim3_grid_batch_fold == NULL)
    {
        struct dim3 *tmp_ptr_array_dim3_grid_batch_fold(static_cast<struct dim3*>(malloc(sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_grid_batch_fold == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(struct dim3));

            return(false);
        }
        *tmp_ptr_array_dim3_grid_batch_fold = struct dim3(1u, 1u, 1u);
        this->ptr_array_dim3_grid_batch_fold = tmp_ptr_array_dim3_grid_batch_fold;
    }
    
    if(this->ptr_array_dim3_block_batch_fold == NULL)
    {
        struct dim3 *tmp_ptr_array_dim3_block_batch_fold(static_cast<struct dim3*>(malloc(sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_block_batch_fold == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(struct dim3));

            return(false);
        }
        *tmp_ptr_array_dim3_block_batch_fold = struct dim3(1u, 1u, 1u);
        this->ptr_array_dim3_block_batch_fold = tmp_ptr_array_dim3_block_batch_fold;
    }
    // |END| Allocate dim3 batch. |END|

    return(true);
}

template<typename T>
__device__ void Dataset_device<T>::Reference(size_t const number_data_received,
                                                                                           size_t const number_inputs_received,
                                                                                           size_t const number_outputs_received,
                                                                                           size_t const number_recurrent_depth_received,
                                                                                           T **const ptr_array_inputs_array_received,
                                                                                           T **const ptr_array_outputs_array_received,
                                                                                           size_t const number_cuRAND_State_MTGP32_shuffle_received,
                                                                                           struct curandStateMtgp32 *const ptr_cuRAND_State_MTGP32_received,
                                                                                           class CUDA_Device_Information_Array *const ptr_Class_Device_Information_Array_received)
{
    this->Deallocate();

    if(this->Allocate_Dim3() == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Allocate_Dim3()\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 __LINE__);

        return;
    }

    this->p_number_examples = number_data_received;
    this->p_number_inputs = number_inputs_received;
    this->p_number_outputs = number_outputs_received;
    this->p_number_recurrent_depth = number_recurrent_depth_received;

    this->p_ptr_array_inputs_array = ptr_array_inputs_array_received;
    this->p_ptr_array_outputs_array = ptr_array_outputs_array_received;
    
    this->p_number_cuRAND_State_MTGP32_shuffle = number_cuRAND_State_MTGP32_shuffle_received;

    this->ptr_array_cuRAND_State_MTGP32_shuffle = ptr_cuRAND_State_MTGP32_received;

    this->p_ptr_Class_Device_Information_Array = ptr_Class_Device_Information_Array_received;
    
    this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(number_data_received,
                                                                                                                                           0u,
                                                                                                                                           *this->ptr_array_dim3_grid_batch,
                                                                                                                                           *this->ptr_array_dim3_block_batch);
    
    this->_reference = true;
}

template<typename T>
__device__ void Dataset_device<T>::Train_Epoch_Batch(class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    if(ptr_CNeural_Network_received->use_Dropout)
    { ptr_CNeural_Network_received->Dropout(); }

    ptr_CNeural_Network_received->Reset__Loss();
    
    switch(ptr_CNeural_Network_received->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: 
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: this->Train_Batch_Batch(ptr_CNeural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
            ptr_CNeural_Network_received->previous_loss_rprop = ptr_CNeural_Network_received->loss_rprop;

            this->Train_Batch_Batch(ptr_CNeural_Network_received);
            
            ptr_CNeural_Network_received->loss_rprop = MyEA::Math::Absolute<T>(ptr_CNeural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
                break;
        default:
            PRINT_FORMAT("%s: ERROR: Undefined optimizer function type (%u)." NEW_LINE,
                    __FUNCTION__,
                    ptr_CNeural_Network_received->type_optimizer_function);
                break;
    }

    ptr_CNeural_Network_received->Merge__Post__Training();
}
    
template<typename T>
__device__ void Dataset_device<T>::Train_Epoch_Mini_Batch_Stochastic(class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    if(ptr_CNeural_Network_received->use_Dropout)
    { ptr_CNeural_Network_received->Dropout(); }

    ptr_CNeural_Network_received->Reset__Loss();
        
    switch(ptr_CNeural_Network_received->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: 
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: this->Train_Batch_Mini_Batch_Stochastic(ptr_CNeural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
            ptr_CNeural_Network_received->previous_loss_rprop = ptr_CNeural_Network_received->loss_rprop;

            this->Train_Batch_Mini_Batch_Stochastic(ptr_CNeural_Network_received);
            
            ptr_CNeural_Network_received->loss_rprop = MyEA::Math::Absolute<T>(ptr_CNeural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
                break;
        default:
            PRINT_FORMAT("%s: ERROR: Undefined optimizer function type (%u)." NEW_LINE,
                    __FUNCTION__,
                    ptr_CNeural_Network_received->type_optimizer_function);
                break;
    }

    ptr_CNeural_Network_received->Merge__Post__Training();
}
    
template<typename T>
__device__ void Dataset_device<T>::Train_Epoch_Cross_Validation_K_Fold(class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    if(ptr_CNeural_Network_received->use_Dropout)
    { ptr_CNeural_Network_received->Dropout(); }

    ptr_CNeural_Network_received->Reset__Loss();
        
    switch(ptr_CNeural_Network_received->type_optimizer_function)
    {
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_GD:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_QUICKPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_SARPROP:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAM:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_ADAMAX:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_AMSGrad: 
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NosADAM: this->Train_Batch_Cross_Validation_K_Fold(ptr_CNeural_Network_received); break;
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_minus:
        case MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_iRPROP_plus:
            ptr_CNeural_Network_received->previous_loss_rprop = ptr_CNeural_Network_received->loss_rprop;

            this->Train_Batch_Cross_Validation_K_Fold(ptr_CNeural_Network_received);
            
            ptr_CNeural_Network_received->loss_rprop = MyEA::Math::Absolute<T>(ptr_CNeural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE));
                break;
        default:
            PRINT_FORMAT("%s: ERROR: Undefined optimizer function type (%u)." NEW_LINE,
                    __FUNCTION__,
                    ptr_CNeural_Network_received->type_optimizer_function);
                break;
    }

    ptr_CNeural_Network_received->Merge__Post__Training();
}

template<typename T>
__device__ void Dataset_device<T>::Train_Batch_Batch(class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    size_t const tmp_number_examples(this->Get__Number_Examples()),
                               tmp_maximum_batch_size(ptr_CNeural_Network_received->batch_size),
                               tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_batch_size,
                      tmp_batch_index;
    
    for(tmp_batch_index = 0u; tmp_batch_index != tmp_number_batchs; ++tmp_batch_index)
    {
        tmp_batch_size = tmp_batch_index + 1u != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - tmp_batch_index * tmp_maximum_batch_size;

        ptr_CNeural_Network_received->Forward_Pass(tmp_batch_size, this->Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);
        
        ptr_CNeural_Network_received->Compute__Error(tmp_batch_size, this->Get__Output_Array() + tmp_batch_index * tmp_maximum_batch_size);
        
        ptr_CNeural_Network_received->Backward_Pass(tmp_batch_size);
        
        ptr_CNeural_Network_received->Update_Derivative_Weight(tmp_batch_size);
    }

    *ptr_CNeural_Network_received->ptr_array_number_loss = tmp_number_examples * this->Get__Number_Outputs();
    ptr_CNeural_Network_received->number_accuracy_trial = tmp_number_examples * this->Get__Number_Outputs();
}
    
template<typename T>
__device__ void Dataset_device<T>::Train_Batch_Mini_Batch_Stochastic(class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    size_t const tmp_number_examples(this->p_number_data_mini_batch),
                               tmp_maximum_batch_size(ptr_CNeural_Network_received->batch_size),
                               tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_batch_size,
                      i;

    for(i = 0u; i != tmp_number_batchs; ++i)
    {
        tmp_batch_size = i + 1u != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - i * tmp_maximum_batch_size;

        ptr_CNeural_Network_received->Forward_Pass(tmp_batch_size, this->ptr_array_inputs_array_stochastic + i * tmp_maximum_batch_size);
        
        ptr_CNeural_Network_received->Compute__Error(tmp_batch_size, this->ptr_array_outputs_array_stochastic + i * tmp_maximum_batch_size);
        
        ptr_CNeural_Network_received->Backward_Pass(tmp_batch_size);
        
        ptr_CNeural_Network_received->Update_Derivative_Weight(tmp_batch_size);
    }

    *ptr_CNeural_Network_received->ptr_array_number_loss = tmp_number_examples * this->Get__Number_Outputs();
    ptr_CNeural_Network_received->number_accuracy_trial = tmp_number_examples * this->Get__Number_Outputs();
}
    
template<typename T>
__device__ void Dataset_device<T>::Train_Batch_Cross_Validation_K_Fold(class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    size_t const tmp_number_examples(this->number_data_k_fold),
                               tmp_maximum_batch_size(ptr_CNeural_Network_received->batch_size),
                               tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_batch_size,
                      i;
    
    for(i = 0u; i != tmp_number_batchs; ++i)
    {
        tmp_batch_size = i + 1u != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - i * tmp_maximum_batch_size;

        ptr_CNeural_Network_received->Forward_Pass(tmp_batch_size, this->ptr_array_inputs_array_k_sub_fold + i * tmp_maximum_batch_size);
        
        ptr_CNeural_Network_received->Compute__Error(tmp_batch_size, this->ptr_array_outputs_array_k_sub_fold + i * tmp_maximum_batch_size);
        
        ptr_CNeural_Network_received->Backward_Pass(tmp_batch_size);
        
        ptr_CNeural_Network_received->Update_Derivative_Weight(tmp_batch_size);
    }

    *ptr_CNeural_Network_received->ptr_array_number_loss = tmp_number_examples * this->Get__Number_Outputs();
    ptr_CNeural_Network_received->number_accuracy_trial = tmp_number_examples * this->Get__Number_Outputs();
}

template<typename T>
__global__ void kernel__Dataset_device__Allocate(size_t const number_data_received,
                                                                               size_t const number_inputs_received,
                                                                               size_t const number_outputs_received,
                                                                               size_t const number_recurrent_depth_received,
                                                                               T *const ptr_array_inputs_received,
                                                                               T *const ptr_array_outputs_received,
                                                                               class CUDA_Device_Information *const ptr_Class_Device_Information_received,
                                                                               class Dataset_device<T> *const ptr_Dataset_device_received)
{
    ptr_Dataset_device_received->device_Allocate(number_data_received,
                                                                        number_inputs_received,
                                                                        number_outputs_received,
                                                                        number_recurrent_depth_received,
                                                                        ptr_array_inputs_received,
                                                                        ptr_array_outputs_received,
                                                                        ptr_Class_Device_Information_received);
}

template<typename T>
__global__ void kernel__Two_Memory_Assign_1D_to_2D(size_t const step_source_0_received,
                                                                                   size_t const step_source_1_received,
                                                                                   T **const ptr_array_destination_0_received,
                                                                                   T **const ptr_array_destination_1_received,
                                                                                   T *const ptr_array_source_0_received,
                                                                                   T *const ptr_array_source_1_received)
{
    size_t const tmp_thread_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    ptr_array_destination_0_received[tmp_thread_index] = ptr_array_source_0_received + tmp_thread_index * step_source_0_received;
    
    ptr_array_destination_1_received[tmp_thread_index] = ptr_array_source_1_received + tmp_thread_index * step_source_1_received;
}

template<typename T>
__global__ void kernel__Two_Memory_Assign_1D_to_2D(size_t const size_received,
                                                                                   size_t const step_source_0_received,
                                                                                   size_t const step_source_1_received,
                                                                                   T **const ptr_array_destination_0_received,
                                                                                   T **const ptr_array_destination_1_received,
                                                                                   T *const ptr_array_source_0_received,
                                                                                   T *const ptr_array_source_1_received)
{
    size_t const tmp_thread_index(blockIdx.x * blockDim.x + threadIdx.x);

    if(tmp_thread_index < size_received)
    {
        ptr_array_destination_0_received[tmp_thread_index] = ptr_array_source_0_received + tmp_thread_index * step_source_0_received;
    
        ptr_array_destination_1_received[tmp_thread_index] = ptr_array_source_1_received + tmp_thread_index * step_source_1_received;
    }
}

template<typename T>
__global__ void kernel_while__Two_Memory_Assign_1D_to_2D(size_t const size_received,
                                                                                            size_t const step_source_0_received,
                                                                                            size_t const step_source_1_received,
                                                                                            T **const ptr_array_destination_0_received,
                                                                                            T **const ptr_array_destination_1_received,
                                                                                            T *const ptr_array_source_0_received,
                                                                                            T *const ptr_array_source_1_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_index(blockIdx.x * blockDim.x + threadIdx.x);

    do
    {
        ptr_array_destination_0_received[tmp_thread_index] = ptr_array_source_0_received + tmp_thread_index * step_source_0_received;
        
        ptr_array_destination_1_received[tmp_thread_index] = ptr_array_source_1_received + tmp_thread_index * step_source_1_received;
        
        tmp_thread_index += tmp_grid_stride;
    } while(tmp_thread_index < size_received);
}

template<typename T>
__device__ void Two_Memory_Assign_1D_to_2D(size_t const size_received,
                                                                        size_t const step_source_0_received,
                                                                        size_t const step_source_1_received,
                                                                        T **const ptr_array_destination_0_received,
                                                                        T **const ptr_array_destination_1_received,
                                                                        T *const ptr_array_source_0_received,
                                                                        T *const ptr_array_source_1_received,
                                                                        struct dim3 const *const ptr_dimension_grid_received,
                                                                        struct dim3 const *const ptr_dimension_block_received)
{
    if(USE_PARALLEL && size_received >= warpSize)
    {
        LAUNCH_KERNEL_POINTER_1D(Two_Memory_Assign_1D_to_2D<T>,
                                                          ptr_dimension_grid_received,
                                                          ptr_dimension_block_received,
                                                          0_zu,
                                                          size_received,
                                                          step_source_0_received,
                                                          step_source_1_received,
                                                          ptr_array_destination_0_received,
                                                          ptr_array_destination_1_received,
                                                          ptr_array_source_0_received,
                                                          ptr_array_source_1_received)
    }
    else
    {
        for(size_t i(0_zu); i != size_received; ++i)
        {
            ptr_array_destination_0_received[i] = ptr_array_source_0_received + i * step_source_0_received;

            ptr_array_destination_1_received[i] = ptr_array_source_1_received + i * step_source_1_received;
        }
    }
}

template<typename T>
__device__ bool Dataset_device<T>::device_Allocate(size_t const number_data_received,
                                                                                   size_t const number_inputs_received,
                                                                                   size_t const number_outputs_received,
                                                                                   size_t const number_recurrent_depth_received,
                                                                                   T const *ptr_array_inputs_received,
                                                                                   T const *ptr_array_outputs_received,
                                                                                   class CUDA_Device_Information *const ptr_Class_Device_Information_received)
{
    T *tmp_ptr_array_inputs,
       *tmp_ptr_array_outputs;

    this->p_number_examples = number_data_received;
    
    if(this->Allocate_Dim3() == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Allocate_Dim3()\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    this->p_number_inputs = number_inputs_received;
    this->p_number_outputs = number_outputs_received;
    this->p_number_recurrent_depth = number_recurrent_depth_received;

    this->p_ptr_array_inputs_array = new T*[number_data_received];
    if(this->p_ptr_array_inputs_array == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 static_cast<size_t>(number_data_received) * sizeof(T*),
                                 __LINE__);

        this->Deallocate();

        return(false);
    }

    this->p_ptr_array_outputs_array = new T*[number_data_received];
    if(this->p_ptr_array_outputs_array == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 static_cast<size_t>(number_data_received) * sizeof(T*),
                                 __LINE__);

        this->Deallocate();

        return(false);
    }
        
    tmp_ptr_array_inputs = new T[number_inputs_received * number_data_received];
    if(tmp_ptr_array_inputs == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 static_cast<size_t>(number_inputs_received * number_data_received) * sizeof(T),
                                 __LINE__);

        this->Deallocate();

        return(false);
    }
        
    tmp_ptr_array_outputs = new T[number_outputs_received * number_data_received];
    if(tmp_ptr_array_outputs == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 static_cast<size_t>(number_outputs_received * number_data_received) * sizeof(T),
                                 __LINE__);

        this->Deallocate();

        return(false);
    }
    
    // Memcpy array inputs.
    struct dim3 tmp_dim3_grid,
                     tmp_dim3_block;

    ptr_Class_Device_Information_received->Grid_Block_1Dimensions(number_inputs_received * number_data_received,
                                                                                                    0u,
                                                                                                    tmp_dim3_grid,
                                                                                                    tmp_dim3_block);
    
    Memory::Memory_Copy_1D<T>(number_inputs_received * number_data_received,
                                                   tmp_ptr_array_inputs,
                                                   ptr_array_inputs_received,
                                                   &tmp_dim3_grid,
                                                   &tmp_dim3_block);
    // |END| Memcpy array inputs. |END|
    
    // Memcpy array outputs.
    ptr_Class_Device_Information_received->Grid_Block_1Dimensions(number_outputs_received * number_data_received,
                                                                                                    0u,
                                                                                                    tmp_dim3_grid,
                                                                                                    tmp_dim3_block);
    
    Memory::Memory_Copy_1D<T>(number_outputs_received * number_data_received,
                                                   tmp_ptr_array_outputs,
                                                   ptr_array_outputs_received,
                                                   &tmp_dim3_grid,
                                                   &tmp_dim3_block);
    // |END| Memcpy array outputs. |END|
    
    ptr_Class_Device_Information_received->Grid_Block_1Dimensions(number_data_received,
                                                                                                    0u,
                                                                                                    *this->ptr_array_dim3_grid_batch,
                                                                                                    *this->ptr_array_dim3_block_batch);
    
    Two_Memory_Assign_1D_to_2D<T>(number_data_received,
                                                         number_inputs_received,
                                                         number_outputs_received,
                                                         this->p_ptr_array_inputs_array,
                                                         this->p_ptr_array_outputs_array,
                                                         tmp_ptr_array_inputs,
                                                         tmp_ptr_array_outputs,
                                                         this->ptr_array_dim3_grid_batch,
                                                         this->ptr_array_dim3_block_batch);

    return(true);
}

template<typename T>
__device__ bool Dataset_device<T>::Check_Topology(size_t const &ref_number_inputs_received, size_t const &ref_number_outputs_received) const
{
    if(ref_number_inputs_received != this->Get__Number_Inputs())
    {
        PRINT_FORMAT("%s: ERROR: Inputs not equal. %d != %d." NEW_LINE,
                    __FUNCTION__,
                    ref_number_inputs_received,
                    this->Get__Number_Inputs());

        return(false);
    }
    else if(ref_number_outputs_received != this->Get__Number_Outputs())
    {
        PRINT_FORMAT("%s: ERROR: Outputs not equal. %d != %d." NEW_LINE,
                    __FUNCTION__,
                    ref_number_inputs_received,
                    this->Get__Number_Inputs());

        return(false);
    }
    else { return(true); }
}

template<typename T>
__device__ class CUDA_Device_Information_Array *Dataset_device<T>::Get__Class_Device_Information_Array(void) const { return(this->p_ptr_Class_Device_Information_Array); }

template<typename T>
__global__ void kernel__Dataset_device__Deallocate(class Dataset_device<T> *const ptr_Dataset_device_received)
{ ptr_Dataset_device_received->Deallocate(); }
template __global__ void kernel__Dataset_device__Deallocate(class Dataset_device<T_> *const);

template<typename T>
__host__ __device__ bool Dataset_device<T>::Deallocate(void)
{
#if defined(__CUDA_ARCH__) == false
    kernel__Dataset_device__Deallocate<T> <<< 1u, 1u >>> (this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    if(this->_reference == false)
    {
        if(this->p_ptr_array_inputs_array != nullptr)
        {
            SAFE_DELETE_ARRAY(this->p_ptr_array_inputs_array[0u]);

            delete[](this->p_ptr_array_inputs_array);
            this->p_ptr_array_inputs_array = nullptr;
        }

        if(this->p_ptr_array_outputs_array != nullptr)
        {
            SAFE_DELETE_ARRAY(this->p_ptr_array_outputs_array[0u]);

            delete[](this->p_ptr_array_outputs_array);
            this->p_ptr_array_outputs_array = nullptr;
        }

        SAFE_DELETE(this->p_ptr_Class_Device_Information_Array);
        
        // cuRAND.
        if(this->ptr_array_cuRAND_State_MTGP32_shuffle != nullptr)
        {
            SAFE_DELETE_ARRAY(this->ptr_array_cuRAND_State_MTGP32_shuffle->k);

            delete(this->ptr_array_cuRAND_State_MTGP32_shuffle);
        }
        // |END| cuRAND. |END|
    }

    SAFE_FREE(this->ptr_array_dim3_grid_batch);
    SAFE_FREE(this->ptr_array_dim3_block_batch);

    SAFE_FREE(this->ptr_array_dim3_grid_batch_fold);
    SAFE_FREE(this->ptr_array_dim3_block_batch_fold);

    SAFE_FREE(this->ptr_array_dim3_grid_shuffle);
    SAFE_FREE(this->ptr_array_dim3_block_shuffle);
    
    // Mini-Batch Stochastic
    SAFE_DELETE_ARRAY(this->ptr_array_stochastic_index);

    SAFE_DELETE_ARRAY(this->ptr_array_inputs_array_stochastic);
    SAFE_DELETE_ARRAY(this->ptr_array_outputs_array_stochastic);
    // - Mini-Batch Stochastic -
        
    // Cross Validation k-fold
    SAFE_DELETE_ARRAY(this->ptr_array_inputs_array_k_fold);
    SAFE_DELETE_ARRAY(this->ptr_array_outputs_array_k_fold);
    // - Cross Validation k-fold -

    this->p_type_dataset_process = MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_NONE;

    return(true);
#endif
}

template<typename T>
__host__ __device__ bool Dataset_device<T>::Get__Use__Shuffle(void) const { return(this->use_shuffle); }

template<typename T>
__global__ void kernel__Dataset_device__Get__Total_Data(size_t *const ptr_number_data_received, class Dataset_device<T> const *const ptr_Dataset_device_received)
{ *ptr_number_data_received = ptr_Dataset_device_received->Get__Total_Data(); }

template<typename T>
__host__ __device__ size_t Dataset_device<T>::Get__Total_Data(void) const
{
#if defined(__CUDA_ARCH__) == false
    size_t tmp_number_examples,
                      *tmp_ptr_device_number_data;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_number_data, sizeof(size_t)));

    kernel__Dataset_device__Get__Total_Data<T> <<< 1u, 1u >>> (tmp_ptr_device_number_data, this);
    
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&tmp_number_examples,
                                                    tmp_ptr_device_number_data,
                                                    sizeof(size_t),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_data));

    return(tmp_number_examples);
#else
    return(this->p_number_examples);
#endif
}

template<typename T>
__global__ void kernel__Dataset_device__Get__Number_Data(size_t *const ptr_number_data_received, class Dataset_device<T> const *const ptr_Dataset_device_received)
{ *ptr_number_data_received = ptr_Dataset_device_received->Get__Number_Examples(); }

template<typename T>
__host__ __device__ size_t Dataset_device<T>::Get__Number_Examples(void) const
{
#if defined(__CUDA_ARCH__) == false
    size_t tmp_number_examples,
                      *tmp_ptr_device_number_data;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_number_data, sizeof(size_t)));

    kernel__Dataset_device__Get__Number_Data<T> <<< 1u, 1u >>> (tmp_ptr_device_number_data, this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&tmp_number_examples,
                                                    tmp_ptr_device_number_data,
                                                    sizeof(size_t),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_data));

    return(tmp_number_examples);
#else
    switch(this->Get__Type_Dataset_Process())
    {
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH: return(this->p_number_data_mini_batch);
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION: return(this->number_data_k_fold);
        default: return(this->p_number_examples);
    }
#endif
}

template<typename T>
__host__ __device__ size_t Dataset_device<T>::Get__Number_CV_K_Fold(void) const { return(this->number_k_fold); }
    
template<typename T>
__host__ __device__ size_t Dataset_device<T>::Get__Number_CV_K_Sub_Fold(void) const { return(this->number_k_sub_fold); }
    
template<typename T>
__host__ __device__ size_t Dataset_device<T>::Get__Number_CV_Data_Per_Fold(void) const { return(this->number_data_per_fold); }

template<typename T>
__host__ __device__ size_t Dataset_device<T>::Get__Number_CV_Data_Training(void) const { return(this->number_data_training); }

template<typename T>
__host__ __device__ size_t Dataset_device<T>::Get__Number_CV_Data_Validating(void) const { return(this->number_data_validating); }

template<typename T>
__host__ __device__ size_t Dataset_device<T>::Get__Number_CV_Data_Per_Sub_Iteration(void) const { return(this->number_data_per_sub_iteration); }

template<typename T>
__host__ __device__ size_t Dataset_device<T>::Get__Number_CV_Data_Last_Sub_Iteration(void) const { return(this->number_data_last_sub_iteration); }

template<typename T>
__host__ __device__ size_t Dataset_device<T>::Get__Number_Inputs(void) const { return(this->p_number_inputs); }

template<typename T>
__host__ __device__ size_t Dataset_device<T>::Get__Number_Outputs(void) const { return(this->p_number_outputs); }

template<typename T>
__host__ __device__ size_t Dataset_device<T>::Get__Number_Recurrent_Depth(void) const { return(this->p_number_recurrent_depth); }
    
template<typename T>
__global__ void kernel__Dataset_device__Training_Process_Batch(T_ *const ptr_loss_received,
                                                                                                       T_ *const ptr_accuracy_received,
                                                                                                       class CUDA_Neural_Network *const ptr_CNeural_Network_received,
                                                                                                       class Dataset_device<T> *const ptr_Dataset_device_received)
{
    ptr_Dataset_device_received->device__Training_Process_Batch(*ptr_loss_received,
                                                                                                      *ptr_accuracy_received,
                                                                                                      ptr_CNeural_Network_received);
}
    
template<typename T>
__host__ T_ Dataset_device<T>::Training_Process_Batch(class Neural_Network *const ptr_Neural_Network_received)
{
    T_ tmp_loss,
         tmp_accuracy,
         *tmp_ptr_device_loss,
         *tmp_ptr_device_accuracy;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_loss, sizeof(T_)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_accuracy, sizeof(T_)));
    
    kernel__Dataset_device__Training_Process_Batch<T> <<< 1u, 1u >>> (tmp_ptr_device_loss,
                                                                                                                  tmp_ptr_device_accuracy,
                                                                                                                  ptr_Neural_Network_received->ptr_device_Neural_Network,
                                                                                                                  this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&tmp_loss,
                                                    tmp_ptr_device_loss,
                                                    sizeof(T_),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&tmp_accuracy,
                                                    tmp_ptr_device_accuracy,
                                                    sizeof(T_),
                                                    cudaMemcpyDeviceToHost));
        
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_loss)); // T_
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_accuracy)); // T_

    ptr_Neural_Network_received->is_update_from_device = false;
        
    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_loss);
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_accuracy);

    return(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
}
    
template<typename T>
__device__ void Dataset_device<T>::device__Training_Process_Batch(T_ &ref_loss_received,
                                                                                                            T_ &ref_accuracy_received,
                                                                                                            class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    if(ptr_CNeural_Network_received->Update__Thread_Size(this->Get__Number_Examples()) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Update__Thread_Size(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 this->Get__Number_Examples(),
                                 __LINE__);

        return;
    }

    if(ptr_CNeural_Network_received->Update__Batch_Size(this->Get__Number_Examples()) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Update__Batch_Size(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 this->Get__Number_Examples(),
                                 __LINE__);

        return;
    }
    
    this->Train_Epoch_Batch(ptr_CNeural_Network_received);

    ptr_CNeural_Network_received->Update_Parameter(this->Get__Number_Examples(), this->Get__Total_Data());
    
    ++ptr_CNeural_Network_received->epoch_time_step;

    ref_loss_received = ptr_CNeural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE);
    ref_accuracy_received = ptr_CNeural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE);

    ptr_CNeural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ref_loss_received);
    ptr_CNeural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ref_accuracy_received);
}
    
template<typename T>
__global__ void kernel__Dataset_device__Training_Process_Mini_Batch_Stochastic(T_ *const ptr_loss_received,
                                                                                                                                T_ *const ptr_accuracy_received,
                                                                                                                                class CUDA_Neural_Network *const ptr_CNeural_Network_received,
                                                                                                                                class Dataset_device<T> *const ptr_Dataset_device_received)
{
    ptr_Dataset_device_received->device__Training_Process_Mini_Batch_Stochastic(*ptr_loss_received,
                                                                                                                               *ptr_accuracy_received,
                                                                                                                               ptr_CNeural_Network_received);
}
    
template<typename T>
__host__ T_ Dataset_device<T>::Training_Process_Mini_Batch_Stochastic(class Neural_Network *const ptr_Neural_Network_received)
{
    T_ tmp_loss,
            tmp_accuracy,
            *tmp_ptr_device_loss,
            *tmp_ptr_device_accuracy;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_loss, sizeof(T_)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_accuracy, sizeof(T_)));
    
    kernel__Dataset_device__Training_Process_Mini_Batch_Stochastic<T> <<< 1u, 1u >>> (tmp_ptr_device_loss,
                                                                                                                                            tmp_ptr_device_accuracy,
                                                                                                                                            ptr_Neural_Network_received->ptr_device_Neural_Network,
                                                                                                                                            this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&tmp_loss,
                                                    tmp_ptr_device_loss,
                                                    sizeof(T_),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&tmp_accuracy,
                                                    tmp_ptr_device_accuracy,
                                                    sizeof(T_),
                                                    cudaMemcpyDeviceToHost));
        
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_loss)); // T_
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_accuracy)); // T_

    ptr_Neural_Network_received->is_update_from_device = false;
        
    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_loss);
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_accuracy);

    return(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
}
    
template<typename T>
__device__ void Dataset_device<T>::device__Training_Process_Mini_Batch_Stochastic(T_ &ref_loss_received,
                                                                                                                                    T_ &ref_accuracy_received,
                                                                                                                                    class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    if(ptr_CNeural_Network_received->Update__Thread_Size(this->p_number_data_mini_batch) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Update__Thread_Size(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 this->p_number_data_mini_batch,
                                 __LINE__);

        return;
    }

    if(ptr_CNeural_Network_received->Update__Batch_Size(this->p_number_data_mini_batch) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Update__Batch_Size(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 this->p_number_data_mini_batch,
                                 __LINE__);

        return;
    }
    
    T_ tmp_summation_loss(0_T),
           tmp_summation_accurancy(0_T);
                
    if(this->use_shuffle) { this->Mini_Batch_Stochastic__Shuffle(); }

    for(size_t j(0u); j != this->p_number_mini_batch; ++j)
    {
        if(this->Mini_Batch_Stochastic__Increment_Mini_Batch(j))
        {
            this->Train_Epoch_Mini_Batch_Stochastic(ptr_CNeural_Network_received);
                        
            tmp_summation_loss += ptr_CNeural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE);
            tmp_summation_accurancy += ptr_CNeural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE);

            ptr_CNeural_Network_received->Update_Parameter(this->p_number_data_mini_batch, this->Get__Total_Data());
        }
        else
        {
            PRINT_FORMAT("%s: ERROR: From \"Mini_Batch_Stochastic__Increment_Mini_Batch\"." NEW_LINE, __FUNCTION__);
                        
            return;
        }
    }

    this->Mini_Batch_Stochastic__Reset();

    ++ptr_CNeural_Network_received->epoch_time_step;

    ref_loss_received = tmp_summation_loss /= static_cast<T_>(this->p_number_mini_batch);
    ref_accuracy_received = tmp_summation_accurancy /= static_cast<T_>(this->p_number_mini_batch);

    ptr_CNeural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_summation_loss);
    ptr_CNeural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_summation_accurancy);
}

template<typename T>
__device__ void Dataset_device<T>::Mini_Batch_Stochastic__Initialize_Shuffle(void)
{
    class CUDA_Device_Information const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

    // Tree shift shuffle.
    if(this->ptr_array_dim3_grid_shuffle == NULL)
    {
        struct dim3 *tmp_ptr_array_dim3_grid_shuffle(static_cast<struct dim3*>(malloc(sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_grid_shuffle == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(struct dim3));

            return;
        }
        *tmp_ptr_array_dim3_grid_shuffle = dim3(1u, 1u, 1u);
        this->ptr_array_dim3_grid_shuffle = tmp_ptr_array_dim3_grid_shuffle;
    }
    
    if(this->ptr_array_dim3_block_shuffle == NULL)
    {
        struct dim3 *tmp_ptr_array_dim3_block_shuffle(static_cast<struct dim3*>(malloc(sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_block_shuffle == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not Allocate memory. malloc(sizeof(%u))" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(struct dim3));

            return;
        }
        *tmp_ptr_array_dim3_block_shuffle = dim3(1u, 1u, 1u);
        this->ptr_array_dim3_block_shuffle = tmp_ptr_array_dim3_block_shuffle;
    }
    
    this->p_number_blocks_shuffle = static_cast<size_t>(ceil(static_cast<double>(this->p_number_examples) / static_cast<double>(this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Get__Warp_Size())));

    tmp_ptr_CUDA_Device->Grid_Block_cuRAND_1Dimensions(this->p_number_blocks_shuffle,
                                                                                            0u,
                                                                                            this->ptr_array_dim3_grid_shuffle[0u],
                                                                                            this->ptr_array_dim3_block_shuffle[0u]);
    // |END| Tree shift shuffle. |END|
}

template<typename T>
__device__ void Dataset_device<T>::Mini_Batch_Stochastic__Shuffle(void)
{
    Memory::Memory_Initialize_Index_Shift<size_t>(this->p_number_examples,
                                                                                   curand(this->ptr_array_cuRAND_State_MTGP32_shuffle) % this->p_number_examples,
                                                                                   this->ptr_array_stochastic_index,
                                                                                   this->ptr_array_dim3_grid_batch,
                                                                                   this->ptr_array_dim3_block_batch);

    Shuffle::Tree_Shuffle<size_t>(this->p_number_blocks_shuffle,
                                                        this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Get__Warp_Size(),
                                                        this->p_number_examples,
                                                        this->ptr_array_stochastic_index,
                                                        this->ptr_array_cuRAND_State_MTGP32_shuffle,
                                                        this->ptr_array_dim3_grid_shuffle,
                                                        this->ptr_array_dim3_block_shuffle);
}

template<typename T>
__global__ void kernel__Dataset_device__Training_Process_Cross_Validation_K_Fold(T_ *const ptr_loss_received,
                                                                                                                                  T_ *const ptr_accuracy_received,
                                                                                                                                  class CUDA_Neural_Network *const ptr_CNeural_Network_received,
                                                                                                                                  class Dataset_device<T> *const ptr_Dataset_device_received)
{
    ptr_Dataset_device_received->device__Training_Process_Cross_Validation_K_Fold(*ptr_loss_received,
                                                                                                                                  *ptr_accuracy_received,
                                                                                                                                  ptr_CNeural_Network_received);
}
    
template<typename T>
__host__ T_ Dataset_device<T>::Training_Process_Cross_Validation_K_Fold(class Neural_Network *const ptr_Neural_Network_received)
{
    T_ tmp_loss,
         tmp_accuracy,
         *tmp_ptr_device_loss,
         *tmp_ptr_device_accuracy;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_loss, sizeof(T_)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_accuracy, sizeof(T_)));
    
    kernel__Dataset_device__Training_Process_Cross_Validation_K_Fold<T> <<< 1u, 1u >>> (tmp_ptr_device_loss,
                                                                                                                                              tmp_ptr_device_accuracy,
                                                                                                                                              ptr_Neural_Network_received->ptr_device_Neural_Network,
                                                                                                                                              this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&tmp_loss,
                                                    tmp_ptr_device_loss,
                                                    sizeof(T_),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&tmp_accuracy,
                                                    tmp_ptr_device_accuracy,
                                                    sizeof(T_),
                                                    cudaMemcpyDeviceToHost));
        
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_loss)); // T_
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_accuracy)); // T_

    ptr_Neural_Network_received->is_update_from_device = false;
        
    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_loss);
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_accuracy);

    return(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
}
    
template<typename T>
__device__ void Dataset_device<T>::device__Training_Process_Cross_Validation_K_Fold(T_ &ref_loss_received,
                                                                                                                                       T_ &ref_accuracy_received,
                                                                                                                                       class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    // Training.
    if(ptr_CNeural_Network_received->Update__Thread_Size(this->number_data_k_fold) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Update__Thread_Size(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 this->number_data_k_fold,
                                 __LINE__);

        return;
    }
    
    if(ptr_CNeural_Network_received->Update__Batch_Size(this->number_data_k_fold) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Update__Batch_Size(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 this->number_data_k_fold,
                                 __LINE__);

        return;
    }

    // Validation.
    if(ptr_CNeural_Network_received->Update__Thread_Size(this->ptr_Validation_Dataset->number_data_k_fold) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Update__Thread_Size(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 this->ptr_Validation_Dataset->number_data_k_fold,
                                 __LINE__);

        return;
    }
    
    T_ tmp_summation_loss(0_T),
         tmp_summation_accurancy(0_T);
    
    if(this->use_shuffle) { this->Cross_Validation_K_Fold__Shuffle(); }
        
    for(size_t j(0u), k; j != this->number_k_fold; ++j)
    {
        if(this->Cross_Validation_K_Fold__Increment_Fold(j))
        {
            for(k = 0u; k != this->number_k_sub_fold; ++k)
            {
                if(this->Cross_Validation_K_Fold__Increment_Sub_Fold(k))
                {
                    this->Train_Epoch_Cross_Validation_K_Fold(ptr_CNeural_Network_received);

                    ptr_CNeural_Network_received->Update_Parameter(this->number_data_k_fold, this->Get__Total_Data());
                }
                else
                {
                    PRINT_FORMAT("%s: ERROR: From \"Cross_Validation_K_Fold__Increment_Sub_Fold\"." NEW_LINE,
                                            __FUNCTION__);
                        
                    return;
                }
            }
                
            tmp_summation_loss += this->ptr_Validation_Dataset->Test_Epoch_Cross_Validation_K_Fold(ptr_CNeural_Network_received);
            tmp_summation_accurancy += ptr_CNeural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE);
        }
        else
        {
            PRINT_FORMAT("%s: ERROR: From \"Cross_Validation_K_Fold__Increment_Fold\"." NEW_LINE,
                                    __FUNCTION__);
                        
            return;
        }
    }

    this->Cross_Validation_K_Fold__Reset();
    
    ++ptr_CNeural_Network_received->epoch_time_step;

    ref_loss_received = tmp_summation_loss /= static_cast<T_>(this->number_k_fold);
    ref_accuracy_received = tmp_summation_accurancy /= static_cast<T_>(this->number_k_fold);

    ptr_CNeural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_summation_loss);
    ptr_CNeural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_summation_accurancy);
}
    
template<typename T>
__global__ void kernel__Dataset_device__Testing(T_ *const ptr_loss_received,
                                                                              T_ *const ptr_accuracy_received,
                                                                              class CUDA_Neural_Network *const ptr_CNeural_Network_received,
                                                                              class Dataset_device<T> *const ptr_Dataset_device_received)
{
    ptr_Dataset_device_received->device__Testing(*ptr_loss_received,
                                                                             *ptr_accuracy_received,
                                                                             ptr_CNeural_Network_received);
}
    
template<typename T>
__device__ void Dataset_device<T>::device__Testing(T_ &ref_loss_received,
                                                                                  T_ &ref_accuracy_received,
                                                                                  class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    if(this->Check_Topology(ptr_CNeural_Network_received->number_inputs, ptr_CNeural_Network_received->number_outputs) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Check_Topology(%u, %u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 ptr_CNeural_Network_received->number_inputs,
                                 ptr_CNeural_Network_received->number_outputs,
                                 __LINE__);

        ref_loss_received = 1.0f;

        return;
    }

    ptr_CNeural_Network_received->Reset__Loss();

    size_t const tmp_number_examples(this->Get__Total_Data()),
                               tmp_maximum_batch_size(ptr_CNeural_Network_received->batch_size),
                               tmp_number_batchs(static_cast<size_t>(ceil(static_cast<double>(tmp_number_examples) / static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_batch_size,
                      i;
    
    if(ptr_CNeural_Network_received->Update__Thread_Size(tmp_number_examples) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Update__Thread_Size(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 tmp_number_examples,
                                 __LINE__);

        return;
    }
    
    for(i = 0u; i != tmp_number_batchs; ++i)
    {
        tmp_batch_size = i + 1u != tmp_number_batchs ? tmp_maximum_batch_size : tmp_number_examples - i * tmp_maximum_batch_size;

        ptr_CNeural_Network_received->Forward_Pass(tmp_batch_size, this->Get__Input_Array() + i * tmp_maximum_batch_size);
        
        ptr_CNeural_Network_received->Test(tmp_batch_size, this->Get__Output_Array() + i * tmp_maximum_batch_size);
    }

    *ptr_CNeural_Network_received->ptr_array_number_loss = tmp_number_examples * this->Get__Number_Outputs();
    ptr_CNeural_Network_received->number_accuracy_trial = tmp_number_examples * this->Get__Number_Outputs();

    // Synchronize the computed error before merging between threads.
    CUDA__Check_Error();

    ptr_CNeural_Network_received->Merge__Post__Training();

    ref_loss_received = ptr_CNeural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE);
    ref_accuracy_received = ptr_CNeural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_NONE);

    ptr_CNeural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, ref_loss_received);
    ptr_CNeural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, ref_accuracy_received);
}
    
template<typename T>
__host__ T_ Dataset_device<T>::Testing(class Neural_Network *const ptr_Neural_Network_received)
{
    T_ tmp_loss,
            tmp_accuracy,
            *tmp_ptr_device_loss,
            *tmp_ptr_device_accuracy;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_loss, sizeof(T_)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_accuracy, sizeof(T_)));
    
    kernel__Dataset_device__Testing<T> <<< 1u, 1u >>> (tmp_ptr_device_loss,
                                                                                         tmp_ptr_device_accuracy,
                                                                                         ptr_Neural_Network_received->ptr_device_Neural_Network,
                                                                                         this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&tmp_loss,
                                                    tmp_ptr_device_loss,
                                                    sizeof(T_),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&tmp_accuracy,
                                                    tmp_ptr_device_accuracy,
                                                    sizeof(T_),
                                                    cudaMemcpyDeviceToHost));
        
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_loss)); // T_
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_accuracy)); // T_
        
    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, tmp_loss);
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, tmp_accuracy);

    return(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));
}

template<typename T>
__host__ __device__ MyEA::Common::ENUM_TYPE_DATASET_PROCESS Dataset_device<T>::Get__Type_Dataset_Process(void) const { return(this->p_type_dataset_process); }
    
template<typename T>
__device__ T Dataset_device<T>::Get__Input_At(size_t const index_received, size_t const sub_index_received) const { return(this->p_ptr_array_inputs_array[index_received][sub_index_received]); }

template<typename T>
__device__ T Dataset_device<T>::Get__Output_At(size_t const index_received, size_t const sub_index_received) const { return(this->p_ptr_array_outputs_array[index_received][sub_index_received]); }
    
template<typename T>
__device__ T* Dataset_device<T>::Get__Input_At(size_t const index_received) const { return(this->p_ptr_array_inputs_array[index_received]); }

template<typename T>
__device__ T* Dataset_device<T>::Get__Output_At(size_t const index_received) const { return(this->p_ptr_array_outputs_array[index_received]); }
    
template<typename T>
__device__ T** Dataset_device<T>::Get__Input_Array(void) const { return(this->p_ptr_array_inputs_array); }

template<typename T>
__device__ T** Dataset_device<T>::Get__Output_Array(void) const { return(this->p_ptr_array_outputs_array); }
    
template<typename T>
__global__ void kernel__Dataset_device__Get__Sizeof(size_t *const ptr_size_t_received, class Dataset_device<T> const *const ptr_Dataset_device_received)
{ *ptr_size_t_received = ptr_Dataset_device_received->Get__Sizeof(); }
template __global__ void kernel__Dataset_device__Get__Sizeof(size_t *const ptr_size_t_received, class Dataset_device<T_> const *const);

template<typename T>
__host__ __device__ size_t Dataset_device<T>::Get__Sizeof(void) const
{
    size_t tmp_total_size_t(0_zu);

#if defined(__CUDA_ARCH__) == false
    size_t *tmp_ptr_device_total_size_t;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_total_size_t, sizeof(size_t)));

    kernel__Dataset_device__Get__Sizeof<T> <<< 1u, 1u >>> (tmp_ptr_device_total_size_t, this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&tmp_total_size_t,
                                                    tmp_ptr_device_total_size_t,
                                                    sizeof(size_t),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_total_size_t));

    return(tmp_total_size_t);
#else
    tmp_total_size_t += sizeof(class Dataset_device<T>); // this

    if(this->_reference == false && this->p_ptr_array_inputs_array != nullptr)
    {
        tmp_total_size_t += this->p_number_examples * sizeof(T*);
        tmp_total_size_t += this->p_number_examples * this->p_number_inputs * sizeof(T);
    }

    if(this->_reference == false && this->p_ptr_array_outputs_array != nullptr)
    {
        tmp_total_size_t += this->p_number_examples * sizeof(T*);
        tmp_total_size_t += this->p_number_examples * this->p_number_outputs * sizeof(T);
    }
    
    if(this->ptr_array_dim3_grid_batch != NULL) { tmp_total_size_t += sizeof(struct dim3); }
    if(this->ptr_array_dim3_block_batch != NULL) { tmp_total_size_t += sizeof(struct dim3); }
    
    if(this->ptr_array_dim3_grid_batch_fold != NULL) { tmp_total_size_t += sizeof(struct dim3); }
    if(this->ptr_array_dim3_block_batch_fold != NULL) { tmp_total_size_t += sizeof(struct dim3); }
    
    if(this->ptr_array_dim3_grid_shuffle != NULL) { tmp_total_size_t += sizeof(struct dim3); }
    if(this->ptr_array_dim3_block_shuffle != NULL) { tmp_total_size_t += sizeof(struct dim3); }
    
    // Mini-Batch Stochastic
    if(this->ptr_array_stochastic_index != nullptr) { tmp_total_size_t += this->p_number_examples * sizeof(size_t); }
    
    if(this->ptr_array_inputs_array_stochastic != nullptr) { tmp_total_size_t += this->p_number_data_last_iteration * sizeof(T*); }
    if(this->ptr_array_outputs_array_stochastic != nullptr) { tmp_total_size_t += this->p_number_data_last_iteration * sizeof(T*); }
    // - Mini-Batch Stochastic -

    // Cross Validation k-fold
    // TODO: Sizeof training || Sizeof validating
    if(this->ptr_array_inputs_array_k_fold != nullptr) { tmp_total_size_t += this->number_data_training * sizeof(T*); }
    if(this->ptr_array_outputs_array_k_fold != nullptr) { tmp_total_size_t += this->number_data_training * sizeof(T*); }
    // - Cross Validation k-fold -

    // cuRAND.
    if(this->ptr_array_cuRAND_State_MTGP32_shuffle != nullptr)
    {
        tmp_total_size_t += this->p_number_cuRAND_State_MTGP32_shuffle * sizeof(struct curandStateMtgp32);
        tmp_total_size_t += this->p_number_cuRAND_State_MTGP32_shuffle * sizeof(struct mtgp32_kernel_params);
    }
    // |END| cuRAND. |END|

    return(tmp_total_size_t);
#endif
}

template<typename T>
__host__ __device__ CUDA_Dataset_Manager<T>::CUDA_Dataset_Manager(void) { }

template<typename T>
__host__ __device__ CUDA_Dataset_Manager<T>::~CUDA_Dataset_Manager(void)
{ this->Deallocate(); }
    
template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Copy(size_t const number_data_received,
                                                                                    size_t const number_inputs_received,
                                                                                    size_t const number_outputs_received,
                                                                                    size_t const number_recurrent_depth_received,
                                                                                    T *const ptr_array_inputs_received,
                                                                                    T *const ptr_array_outputs_received,
                                                                                    class CUDA_Device_Information *const ptr_Class_Device_Information_received,
                                                                                    class CUDA_Dataset_Manager<T> *const ptr_CUDA_Dataset_Manager_received)
{
    ptr_CUDA_Dataset_Manager_received->device_Copy(number_data_received,
                                                                                  number_inputs_received,
                                                                                  number_outputs_received,
                                                                                  number_recurrent_depth_received,
                                                                                  ptr_array_inputs_received,
                                                                                  ptr_array_outputs_received,
                                                                                  ptr_Class_Device_Information_received);
}
    
template<typename T>
__host__ bool CUDA_Dataset_Manager<T>::Copy(class Dataset_Manager<T> *const ptr_Dataset_Manager_received)
{
    size_t const tmp_number_examples(ptr_Dataset_Manager_received->Get__Number_Examples()),
                               tmp_number_inputs(ptr_Dataset_Manager_received->Get__Number_Inputs()),
                               tmp_number_outputs(ptr_Dataset_Manager_received->Get__Number_Outputs()),
                               tmp_number_time_predictions(ptr_Dataset_Manager_received->Get__Number_Recurrent_Depth());
    int tmp_index_device(0);
        
    T *tmp_ptr_device_array_inputs_array,
        *tmp_ptr_device_array_outputs_array;
        
    class CUDA_Device_Information *tmp_ptr_Class_Device_Information;
        
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_Class_Device_Information, sizeof(class CUDA_Device_Information)));

    CUDA__Safe_Call(cudaGetDevice(&tmp_index_device));

    tmp_ptr_Class_Device_Information->Initialize(tmp_index_device);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_inputs_array, tmp_number_inputs * tmp_number_examples * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_outputs_array, tmp_number_outputs * tmp_number_examples * sizeof(T)));

    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_inputs_array,
                                                    ptr_Dataset_Manager_received->Get__Input_Array(),
                                                    tmp_number_inputs * tmp_number_examples * sizeof(T),
                                                    cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_outputs_array,
                                                    ptr_Dataset_Manager_received->Get__Output_Array(),
                                                    tmp_number_outputs * tmp_number_examples * sizeof(T),
                                                    cudaMemcpyHostToDevice));
        
    kernel__CUDA_Dataset_Manager__Copy<T> <<< 1u, 1u >>>(tmp_number_examples, // size_t
                                                                                              tmp_number_inputs, // size_t
                                                                                              tmp_number_outputs, // size_t
                                                                                              tmp_number_time_predictions, // size_t
                                                                                              tmp_ptr_device_array_inputs_array, // T
                                                                                              tmp_ptr_device_array_outputs_array, // T
                                                                                              tmp_ptr_Class_Device_Information, // class CUDA_Device_Information
                                                                                              this); // class

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaFree(tmp_ptr_Class_Device_Information)); // class CUDA_Device_Information
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_inputs_array)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_outputs_array)); // T

    if(this->Initialize_CUDA_Device() == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Initialize_CUDA_Device()\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }
    else if(this->Initialize_cuRAND(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count())) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Initialize_cuRAND(random)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 __LINE__);

        return(false);
    }

    return(true);
}

template<typename T>
__device__ bool CUDA_Dataset_Manager<T>::device_Copy(size_t const number_data_received,
                                                                                       size_t const number_inputs_received,
                                                                                       size_t const number_outputs_received,
                                                                                       size_t const number_recurrent_depth_received,
                                                                                       T const *ptr_array_inputs_received,
                                                                                       T const *ptr_array_outputs_received,
                                                                                       class CUDA_Device_Information *const ptr_Class_Device_Information_received)
{
    T *tmp_ptr_array_inputs,
       *tmp_ptr_array_outputs;

    this->p_number_examples = number_data_received;

    this->p_number_inputs = number_inputs_received;
    this->p_number_outputs = number_outputs_received;
    this->p_number_recurrent_depth = number_recurrent_depth_received;

    this->p_ptr_array_inputs_array = new T*[number_data_received];
    if(this->p_ptr_array_inputs_array == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 static_cast<size_t>(number_data_received) * sizeof(T*),
                                 __LINE__);

        this->Deallocate();

        return(false);
    }

    this->p_ptr_array_outputs_array = new T*[number_data_received];
    if(this->p_ptr_array_outputs_array == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 static_cast<size_t>(number_data_received) * sizeof(T*),
                                 __LINE__);

        this->Deallocate();

        return(false);
    }
        
    tmp_ptr_array_inputs = new T[number_inputs_received * number_data_received];
    if(tmp_ptr_array_inputs == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 static_cast<size_t>(number_inputs_received * number_data_received) * sizeof(T),
                                 __LINE__);

        this->Deallocate();

        return(false);
    }
        
    tmp_ptr_array_outputs = new T[number_outputs_received * number_data_received];
    if(tmp_ptr_array_outputs == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 static_cast<size_t>(number_outputs_received * number_data_received) * sizeof(T),
                                 __LINE__);

        this->Deallocate();

        return(false);
    }
    
    // Memcpy array inputs.
    struct dim3 tmp_dim3_grid,
                     tmp_dim3_block;

    ptr_Class_Device_Information_received->Grid_Block_1Dimensions(number_inputs_received * number_data_received,
                                                                                                    0u,
                                                                                                    tmp_dim3_grid,
                                                                                                    tmp_dim3_block);
    
    Memory::Memory_Copy_1D<T>(number_inputs_received * number_data_received,
                                                   tmp_ptr_array_inputs,
                                                   ptr_array_inputs_received,
                                                   &tmp_dim3_grid,
                                                   &tmp_dim3_block);
    // |END| Memcpy array inputs. |END|
    
    // Memcpy array outputs.
    ptr_Class_Device_Information_received->Grid_Block_1Dimensions(number_outputs_received * number_data_received,
                                                                                                    0u,
                                                                                                    tmp_dim3_grid,
                                                                                                    tmp_dim3_block);
    
    Memory::Memory_Copy_1D<T>(number_outputs_received * number_data_received,
                                                   tmp_ptr_array_outputs,
                                                   ptr_array_outputs_received,
                                                   &tmp_dim3_grid,
                                                   &tmp_dim3_block);
    // |END| Memcpy array outputs. |END|
    
    ptr_Class_Device_Information_received->Grid_Block_1Dimensions(number_data_received,
                                                                                                    0u,
                                                                                                    tmp_dim3_grid,
                                                                                                    tmp_dim3_block);

    Two_Memory_Assign_1D_to_2D<T>(number_data_received,
                                                         number_inputs_received,
                                                         number_outputs_received,
                                                         this->p_ptr_array_inputs_array,
                                                         this->p_ptr_array_outputs_array,
                                                         tmp_ptr_array_inputs,
                                                         tmp_ptr_array_outputs,
                                                         &tmp_dim3_grid,
                                                         &tmp_dim3_block);

    return(true);
}

template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Deallocate(class CUDA_Dataset_Manager<T> *const ptr_CUDA_Dataset_Manager_received)
{ ptr_CUDA_Dataset_Manager_received->Deallocate(); }
template __global__ void kernel__CUDA_Dataset_Manager__Deallocate(class CUDA_Dataset_Manager<T_> *const);

template<typename T>
__host__ __device__ bool CUDA_Dataset_Manager<T>::Deallocate(void)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CUDA_Dataset_Manager__Deallocate<T> <<< 1u, 1u >>> (this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    SAFE_DELETE(this->p_ptr_Class_Device_Information_Array);

    if(this->p_ptr_array_inputs_array != nullptr)
    {
        SAFE_DELETE_ARRAY(this->p_ptr_array_inputs_array[0u]);

        delete[](this->p_ptr_array_inputs_array);
        this->p_ptr_array_inputs_array = nullptr;
    }

    if(this->p_ptr_array_outputs_array != nullptr)
    {
        SAFE_DELETE_ARRAY(this->p_ptr_array_outputs_array[0u]);

        delete[](this->p_ptr_array_outputs_array);
        this->p_ptr_array_outputs_array = nullptr;
    }

    if(this->_ptr_array_Dataset != nullptr)
    {
        switch(this->_type_storage_data)
        {
            case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
                if(this->_ptr_array_Dataset[0u].Deallocate() == false)
                {
                    PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"[0].Deallocate()\" function. At line %d." NEW_LINE,
                                             __FUNCTION__,
                                             __LINE__);
                    
                    return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
                if(this->_ptr_array_Dataset[0u].Deallocate() == false)
                {
                    PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"[0].Deallocate()\" function. At line %d." NEW_LINE,
                                             __FUNCTION__,
                                             __LINE__);
                    
                    return(false);
                }

                if(this->_ptr_array_Dataset[1u].Deallocate() == false)
                {
                    PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"[1].Deallocate()\" function. At line %d." NEW_LINE,
                                             __FUNCTION__,
                                             __LINE__);
                    
                    return(false);
                }
                    break;
            case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
                if(this->_ptr_array_Dataset[0u].Deallocate() == false)
                {
                    PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"[0].Deallocate()\" function. At line %d." NEW_LINE,
                                             __FUNCTION__,
                                             __LINE__);
                    
                    return(false);
                }

                if(this->_ptr_array_Dataset[1u].Deallocate() == false)
                {
                    PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"[1].Deallocate()\" function. At line %d." NEW_LINE,
                                             __FUNCTION__,
                                             __LINE__);
                    
                    return(false);
                }

                if(this->_ptr_array_Dataset[2u].Deallocate() == false)
                {
                    PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"[2].Deallocate()\" function. At line %d." NEW_LINE,
                                             __FUNCTION__,
                                             __LINE__);
                    
                    return(false);
                }
                    break;
            default:
                PRINT_FORMAT("%s: ERROR: Dataset storage type (%u) is not managed in the switch. At line %d." NEW_LINE,
                                            __FUNCTION__,
                                            this->_type_storage_data,
                                            __LINE__);
                    break;
        }

        delete[](this->_ptr_array_Dataset);
        this->_ptr_array_Dataset = nullptr;
    }
    
    // cuRAND.
    if(this->ptr_array_cuRAND_State_MTGP32_shuffle != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_cuRAND_State_MTGP32_shuffle->k);

        delete(this->ptr_array_cuRAND_State_MTGP32_shuffle);
    }
    // |END| cuRAND. |END|

    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Initialize(class CUDA_Dataset_Manager<T> *const ptr_CUDA_Dataset_Manager_received)
{ ptr_CUDA_Dataset_Manager_received->Initialize(); }
template __global__ void kernel__CUDA_Dataset_Manager__Initialize(class CUDA_Dataset_Manager<T_> *const);

template<typename T>
__device__ class CUDA_Device_Information_Array *CUDA_Dataset_Manager<T>::Get__Class_Device_Information_Array(void) const { return(this->p_ptr_Class_Device_Information_Array); }

template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Add_CUDA_Device(int const index_device_received,
                                                                                                        struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received,
                                                                                                        class CUDA_Dataset_Manager<T> *const ptr_Dataset_device_received)
{ ptr_Dataset_device_received->Add_CUDA_Device(index_device_received, ptr_struct_cudaDeviceProp_received); }
    
template<typename T>
__device__ bool CUDA_Dataset_Manager<T>::Add_CUDA_Device(int const index_device_received, struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received)
{
    if(this->p_ptr_Class_Device_Information_Array == nullptr)
    { this->p_ptr_Class_Device_Information_Array = new class CUDA_Device_Information_Array; }

    return(this->p_ptr_Class_Device_Information_Array->Push_Back(index_device_received, ptr_struct_cudaDeviceProp_received));
}

template<typename T>
__host__ bool CUDA_Dataset_Manager<T>::Initialize_CUDA_Device(void)
{
    int tmp_index_device(0),
        tmp_number_CUDA_devices;
        
    struct cudaDeviceProp tmp_struct_cudaDeviceProp,
                                     *tmp_ptr_device_struct_cudaDeviceProp(NULL);

    CUDA__Safe_Call(cudaGetDeviceCount(&tmp_number_CUDA_devices));
        
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_struct_cudaDeviceProp, sizeof(struct cudaDeviceProp)));

    for(; tmp_index_device != tmp_number_CUDA_devices; ++tmp_index_device)
    {
        CUDA__Safe_Call(cudaGetDeviceProperties(&tmp_struct_cudaDeviceProp, tmp_index_device));

        CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_struct_cudaDeviceProp,
                                                        &tmp_struct_cudaDeviceProp,
                                                        sizeof(struct cudaDeviceProp),
                                                        cudaMemcpyKind::cudaMemcpyHostToDevice));

        kernel__CUDA_Dataset_Manager__Add_CUDA_Device <<< 1u, 1u >>> (tmp_index_device,
                                                                                                                      tmp_ptr_device_struct_cudaDeviceProp,
                                                                                                                      this);
            
    #if defined(COMPILE_DEBUG)
        CUDA__Check_Error();
    #endif
    }

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_struct_cudaDeviceProp));

    return(true);
}

template<typename T>
__host__ __device__ bool CUDA_Dataset_Manager<T>::Initialize(void)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CUDA_Dataset_Manager__Initialize<T> <<< 1u, 1u >>> (this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    this->p_number_examples = 0u;
    this->p_number_recurrent_depth = 0u;
    this->p_number_inputs = 0u;
    this->p_number_outputs = 0u;

    this->p_ptr_array_inputs_array = nullptr;
    this->p_ptr_array_outputs_array = nullptr;
        
    this->_type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE;

    this->p_ptr_Class_Device_Information_Array = nullptr;

    this->_ptr_array_Dataset = nullptr;
    
    // cuRAND.
    this->p_number_cuRAND_State_MTGP32_shuffle = 0u;

    this->ptr_array_cuRAND_State_MTGP32_shuffle = nullptr;
    // |END| cuRAND. |END|
    
    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Initialize(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received,
                                                                                        enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_gradient_descent_received,
                                                                                        class CUDA_Dataset_Manager<T> *const ptr_CUDA_Dataset_Manager_received)
{ ptr_CUDA_Dataset_Manager_received->Initialize(type_dataset_received, type_gradient_descent_received); }
template __global__ void kernel__CUDA_Dataset_Manager__Initialize(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received,
                                                                                                         enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const,
                                                                                                         class CUDA_Dataset_Manager<T_> *const);

template<typename T>
__host__ __device__ bool CUDA_Dataset_Manager<T>::Initialize(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_gradient_descent_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CUDA_Dataset_Manager__Initialize<T> <<< 1u, 1u >>> (type_dataset_received,
                                                                                                                    type_gradient_descent_received,
                                                                                                                    this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    class Dataset_device<T> *const tmp_ptr_Dataset_device(this->Get__Dataset_At(type_dataset_received));

    if(tmp_ptr_Dataset_device == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Get__Dataset_At(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 type_dataset_received,
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_Dataset_device->Initialize(type_gradient_descent_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Initialize(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 type_gradient_descent_received,
                                 __LINE__);

        return(false);
    }

    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Initialize_Mini_Batch_Stochastic_Gradient_Descent(bool const use_shuffle_received,
                                                                                                                                                       size_t const desired_number_data_per_mini_batch_received,
                                                                                                                                                       size_t const number_mini_batch_maximum_received,
                                                                                                                                                       class CUDA_Dataset_Manager<T> *const ptr_CUDA_Dataset_Manager_received)
{
    ptr_CUDA_Dataset_Manager_received->Initialize_Mini_Batch_Stochastic_Gradient_Descent(use_shuffle_received,
                                                                                                                                          desired_number_data_per_mini_batch_received,
                                                                                                                                          number_mini_batch_maximum_received);
}
template __global__ void kernel__CUDA_Dataset_Manager__Initialize_Mini_Batch_Stochastic_Gradient_Descent(bool const,
                                                                                                                                                                    size_t const,
                                                                                                                                                                    size_t const,
                                                                                                                                                                    class CUDA_Dataset_Manager<T_> *const);

template<typename T>
__host__ __device__ bool CUDA_Dataset_Manager<T>::Initialize_Mini_Batch_Stochastic_Gradient_Descent(bool const use_shuffle_received,
                                                                                                                                                              size_t const desired_number_data_per_mini_batch_received,
                                                                                                                                                              size_t const number_mini_batch_maximum_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CUDA_Dataset_Manager__Initialize_Mini_Batch_Stochastic_Gradient_Descent<T> <<< 1u, 1u >>> (use_shuffle_received,
                                                                                                                                                                   desired_number_data_per_mini_batch_received,
                                                                                                                                                                   number_mini_batch_maximum_received,
                                                                                                                                                                   this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    class Dataset_device<T> *tmp_ptr_Dataset_device(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));

    if(tmp_ptr_Dataset_device == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Get__Dataset_At(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_Dataset_device->Get__Type_Dataset_Process() != MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH)
    {
        PRINT_FORMAT("%s: ERROR: The dataset process (%u) differs from the mini-batch process (%u). At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 tmp_ptr_Dataset_device->Get__Type_Dataset_Process(),
                                 MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH,
                                 __LINE__);

        return(false);
    }

    tmp_ptr_Dataset_device->Initialize_Mini_Batch_Stochastic_Gradient_Descent(use_shuffle_received,
                                                                                                                      desired_number_data_per_mini_batch_received,
                                                                                                                      number_mini_batch_maximum_received);

    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Initialize_Cross_Validation_K_Fold(bool const use_shuffle_received,
                                                                                                                                  size_t const number_k_fold_received,
                                                                                                                                  size_t const number_k_sub_fold_received,
                                                                                                                                  class CUDA_Dataset_Manager<T> *const ptr_CUDA_Dataset_Manager_received)
{
    ptr_CUDA_Dataset_Manager_received->Initialize__Cross_Validation(use_shuffle_received,
                                                                                                                     number_k_fold_received,
                                                                                                                     number_k_sub_fold_received);
}
template __global__ void kernel__CUDA_Dataset_Manager__Initialize_Cross_Validation_K_Fold(bool const,
                                                                                                                                                size_t const,
                                                                                                                                                size_t const,
                                                                                                                                                class CUDA_Dataset_Manager<T_> *const);

template<typename T>
__host__ __device__ bool CUDA_Dataset_Manager<T>::Initialize__Cross_Validation(bool const use_shuffle_received,
                                                                                                                                         size_t const number_k_fold_received,
                                                                                                                                         size_t const number_k_sub_fold_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CUDA_Dataset_Manager__Initialize_Cross_Validation_K_Fold<T> <<< 1u, 1u >>> (use_shuffle_received,
                                                                                                                                              number_k_fold_received,
                                                                                                                                              number_k_sub_fold_received,
                                                                                                                                              this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    class Dataset_device<T> *tmp_ptr_Dataset_device(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));

    if(tmp_ptr_Dataset_device == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Get__Dataset_At(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                 __LINE__);

        return(false);
    }
    else  if(number_k_fold_received < 2u)
    {
        PRINT_FORMAT("%s: ERROR: Not enough K-fold." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else if(tmp_ptr_Dataset_device->Get__Type_Dataset_Process() != MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION)
    {
        PRINT_FORMAT("%s: ERROR: The dataset process (%u) differs from the cross validating k-fold process (%u). At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 tmp_ptr_Dataset_device->Get__Type_Dataset_Process(),
                                 MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION,
                                 __LINE__);

        return(false);
    }

    tmp_ptr_Dataset_device->Initialize__Cross_Validation(use_shuffle_received,
                                                                                                  number_k_fold_received,
                                                                                                  number_k_sub_fold_received,
                                                                                                  this);

    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Initialize_Cross_Validation_K_Fold(class CUDA_Dataset_Manager<T> *const ptr_CUDA_Dataset_Manager_received)
{ ptr_CUDA_Dataset_Manager_received->Initialize__Cross_Validation(); }
template __global__ void kernel__CUDA_Dataset_Manager__Initialize_Cross_Validation_K_Fold(class CUDA_Dataset_Manager<T_> *const);

template<typename T>
__host__ __device__ bool CUDA_Dataset_Manager<T>::Initialize__Cross_Validation(void)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CUDA_Dataset_Manager__Initialize_Cross_Validation_K_Fold<T> <<< 1u, 1u >>> (this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    class Dataset_device<T> *tmp_ptr_Dataset_device(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION));

    if(tmp_ptr_Dataset_device == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Get__Dataset_At(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION,
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_Dataset_device->Get__Type_Dataset_Process() != MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION)
    {
        PRINT_FORMAT("%s: ERROR: The dataset process (%u) differs from the cross validating k-fold process (%u). At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 tmp_ptr_Dataset_device->Get__Type_Dataset_Process(),
                                 MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION,
                                 __LINE__);

        return(false);
    }

    tmp_ptr_Dataset_device->Initialize__Cross_Validation(this);

    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Set__Type_Gradient_Descent(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received,
                                                                                                                                        enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_gradient_descent_received,
                                                                                                                                        class CUDA_Dataset_Manager<T> *const ptr_CUDA_Dataset_Manager_received)
{ ptr_CUDA_Dataset_Manager_received->Set__Type_Gradient_Descent(type_dataset_received, type_gradient_descent_received); }
template __global__ void kernel__CUDA_Dataset_Manager__Set__Type_Gradient_Descent(enum MyEA::Common::ENUM_TYPE_DATASET const,
                                                                                                                                        enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const,
                                                                                                                                        class CUDA_Dataset_Manager<T_> *const);

template<typename T>
__host__ __device__ bool CUDA_Dataset_Manager<T>::Set__Type_Gradient_Descent(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_gradient_descent_received)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CUDA_Dataset_Manager__Set__Type_Gradient_Descent<T> <<< 1u, 1u >>> (type_dataset_received,
                                                                                                                                                    type_gradient_descent_received,
                                                                                                                                                    this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    class Dataset_device<T> *const tmp_ptr_Dataset_device(this->Get__Dataset_At(type_dataset_received));

    if(tmp_ptr_Dataset_device == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Get__Dataset_At(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 type_dataset_received,
                                 __LINE__);

        return(false);
    }
    else if(tmp_ptr_Dataset_device->Set__Type_Gradient_Descent(type_gradient_descent_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Set__Type_Gradient_Descent(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 type_gradient_descent_received,
                                 __LINE__);

        return(false);
    }

    return(true);
#endif
}

template<typename T>
__device__ T Get__Limit(T const value_received,
                                  T const minimum_received,
                                  T const maximum_received)
{
    if(value_received < minimum_received) { return(minimum_received); }
    else if(value_received > maximum_received) { return(maximum_received); }
    else { return(value_received); }
}

template<typename T>
__device__ T Get__Minimum(T const value_received, T const minimum_received)
{
    if(value_received < minimum_received) { return(minimum_received); }
    else { return(value_received); }
}

template<typename T>
__device__ T Get__Maximum(T const value_received, T const maximum_received)
{
    if(value_received > maximum_received) { return(maximum_received); }
    else { return(value_received); }
}

template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Prepare_Storage(class CUDA_Dataset_Manager<T> *const ptr_CUDA_Dataset_Manager_received)
{ ptr_CUDA_Dataset_Manager_received->Prepare_Storage(); }
template __global__ void kernel__CUDA_Dataset_Manager__Prepare_Storage(class CUDA_Dataset_Manager<T_> *const);
    
template<typename T>
__host__ __device__ bool CUDA_Dataset_Manager<T>::Prepare_Storage(void)
{
#if defined(__CUDA_ARCH__) == false
    kernel__CUDA_Dataset_Manager__Prepare_Storage<T> <<< 1u, 1u >>> (this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    if(this->Get__Number_Examples() == 0u)
    {
        PRINT_FORMAT("%s: ERROR: Number of data equal to zero." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else if(this->_type_storage_data != MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE)
    {
        PRINT_FORMAT("%s: ERROR: Can not prepare storage multiple time." NEW_LINE, __FUNCTION__);

        return(false);
    }

    this->_ptr_array_Dataset = new class Dataset_device<T>[1u];

    this->_ptr_array_Dataset[0u].Reference(this->Get__Number_Examples(),
                                                                                    this->p_number_inputs,
                                                                                    this->p_number_outputs,
                                                                                    this->p_number_recurrent_depth,
                                                                                    this->p_ptr_array_inputs_array,
                                                                                    this->p_ptr_array_outputs_array,
                                                                                    this->p_number_cuRAND_State_MTGP32_shuffle,
                                                                                    this->ptr_array_cuRAND_State_MTGP32_shuffle,
                                                                                    this->p_ptr_Class_Device_Information_Array);
        
    this->_type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING;

    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__Dataset_Manager__Prepare_Storage(size_t const number_data_training_received,
                                                                                              size_t const number_data_testing_received,
                                                                                              class CUDA_Dataset_Manager<T> *const ptr_Dataset_Manager_received)
{ ptr_Dataset_Manager_received->Prepare_Storage(number_data_training_received, number_data_testing_received); }
template __global__ void kernel__Dataset_Manager__Prepare_Storage(size_t const,
                                                                                                            size_t const,
                                                                                                            class CUDA_Dataset_Manager<T_> *const);

template<typename T>
__host__ __device__ bool CUDA_Dataset_Manager<T>::Prepare_Storage(size_t const number_data_training_received, size_t const number_data_testing_received)
{
    if(number_data_training_received == 0u)
    {
        PRINT_FORMAT("%s: ERROR: Number of training data equal to zero." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else if(number_data_testing_received == 0u)
    {
        PRINT_FORMAT("%s: ERROR: Number of testing data equal to zero." NEW_LINE, __FUNCTION__);

        return(false);
    }

#if defined(__CUDA_ARCH__) == false
    kernel__Dataset_Manager__Prepare_Storage<T> <<< 1u, 1u >>> (number_data_training_received,
                                                                                                         number_data_testing_received,
                                                                                                         this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    if(number_data_training_received + number_data_testing_received != this->Get__Number_Examples())
    {
        PRINT_FORMAT("%s: ERROR: training(%d) + testing(%d) != data(%d)" NEW_LINE,
                                __FUNCTION__,
                                number_data_training_received,
                                number_data_testing_received,
                                this->Get__Number_Examples());

        return(false);
    }
    else if(this->Get__Number_Examples() < 2u)
    {
        PRINT_FORMAT("%s: ERROR: Number of data (%u) < 2" NEW_LINE,
                                 __FUNCTION__,
                                 this->Get__Number_Examples());

        return(false);
    }
    else if(this->_type_storage_data != MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE)
    {
        PRINT_FORMAT("%s: ERROR: Can not prepare storage multiple time." NEW_LINE, __FUNCTION__);

        return(false);
    }

    T **tmp_ptr_array_inputs_array(this->p_ptr_array_inputs_array),
        **tmp_ptr_array_outputs_array(this->p_ptr_array_outputs_array);
        
    this->_ptr_array_Dataset = new class Dataset_device<T>[2u];

    this->_ptr_array_Dataset[0u].Reference(number_data_training_received,
                                                                                    this->p_number_inputs,
                                                                                    this->p_number_outputs,
                                                                                    this->p_number_recurrent_depth,
                                                                                    tmp_ptr_array_inputs_array,
                                                                                    tmp_ptr_array_outputs_array,
                                                                                    this->p_number_cuRAND_State_MTGP32_shuffle,
                                                                                    this->ptr_array_cuRAND_State_MTGP32_shuffle,
                                                                                    this->p_ptr_Class_Device_Information_Array);
                
    tmp_ptr_array_inputs_array += number_data_training_received;
    tmp_ptr_array_outputs_array += number_data_training_received;

    this->_ptr_array_Dataset[1u].Reference(number_data_testing_received,
                                                                                    this->p_number_inputs,
                                                                                    this->p_number_outputs,
                                                                                    this->p_number_recurrent_depth,
                                                                                    tmp_ptr_array_inputs_array,
                                                                                    tmp_ptr_array_outputs_array,
                                                                                    this->p_number_cuRAND_State_MTGP32_shuffle,
                                                                                    this->ptr_array_cuRAND_State_MTGP32_shuffle,
                                                                                    this->p_ptr_Class_Device_Information_Array);

    this->_type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING;

    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__Dataset_Manager__Prepare_Storage(size_t const number_data_training_received,
                                                                                          size_t const number_data_validation_received,
                                                                                          size_t const number_data_testing_received,
                                                                                          class CUDA_Dataset_Manager<T> *const ptr_Dataset_Manager_received)
{
    ptr_Dataset_Manager_received->Prepare_Storage(number_data_training_received,
                                                                                 number_data_validation_received,
                                                                                 number_data_testing_received);
}
template __global__ void kernel__Dataset_Manager__Prepare_Storage(size_t const,
                                                                                                            size_t const,
                                                                                                            size_t const,
                                                                                                            class CUDA_Dataset_Manager<T_> *const);

template<typename T>
__host__ __device__ bool CUDA_Dataset_Manager<T>::Prepare_Storage(size_t const number_data_training_received,
                                                                                                                size_t const number_data_validation_received,
                                                                                                                size_t const number_data_testing_received)
{
    if(number_data_training_received == 0u)
    {
        PRINT_FORMAT("%s: ERROR: Number of training data equal to zero." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else if(number_data_validation_received == 0u)
    {
        PRINT_FORMAT("%s: ERROR: Number of validating data equal to zero." NEW_LINE, __FUNCTION__);

        return(false);
    }
    else if(number_data_testing_received == 0u)
    {
        PRINT_FORMAT("%s: ERROR: Number of testing data equal to zero." NEW_LINE, __FUNCTION__);

        return(false);
    }

#if defined(__CUDA_ARCH__) == false
    kernel__Dataset_Manager__Prepare_Storage<T> <<< 1u, 1u >>> (number_data_training_received,
                                                                                                                        number_data_validation_received,
                                                                                                                        number_data_testing_received,
                                                                                                                        this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    if(number_data_training_received + number_data_validation_received + number_data_testing_received != this->Get__Number_Examples())
    {
        PRINT_FORMAT("%s: ERROR: training(%d) + validation(%d) + testing(%d) != data(%d)" NEW_LINE,
                                __FUNCTION__,
                                number_data_training_received,
                                number_data_validation_received,
                                number_data_testing_received,
                                this->Get__Number_Examples());

        return(false);
    }
    else if(this->Get__Number_Examples() < 3u)
    {
        PRINT_FORMAT("%s: ERROR: Number of data (%u) < 3" NEW_LINE,
                                 __FUNCTION__,
                                 this->Get__Number_Examples());

        return(false);
    }
    else if(this->_type_storage_data != MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE)
    {
        PRINT_FORMAT("%s: ERROR: Can not prepare storage multiple time." NEW_LINE, __FUNCTION__);

        return(false);
    }

    T **tmp_ptr_array_inputs_array(this->p_ptr_array_inputs_array),
        **tmp_ptr_array_outputs_array(this->p_ptr_array_outputs_array);
        
    this->_ptr_array_Dataset = new class Dataset_device<T>[3u];

    this->_ptr_array_Dataset[0u].Reference(number_data_training_received,
                                                                                    this->p_number_inputs,
                                                                                    this->p_number_outputs,
                                                                                    this->p_number_recurrent_depth,
                                                                                    tmp_ptr_array_inputs_array,
                                                                                    tmp_ptr_array_outputs_array,
                                                                                    this->p_number_cuRAND_State_MTGP32_shuffle,
                                                                                    this->ptr_array_cuRAND_State_MTGP32_shuffle,
                                                                                    this->p_ptr_Class_Device_Information_Array);
        
    tmp_ptr_array_inputs_array += number_data_training_received;
    tmp_ptr_array_outputs_array += number_data_training_received;

    this->_ptr_array_Dataset[1u].Reference(number_data_validation_received,
                                                                                    this->p_number_inputs,
                                                                                    this->p_number_outputs,
                                                                                    this->p_number_recurrent_depth,
                                                                                    tmp_ptr_array_inputs_array,
                                                                                    tmp_ptr_array_outputs_array,
                                                                                    this->p_number_cuRAND_State_MTGP32_shuffle,
                                                                                    this->ptr_array_cuRAND_State_MTGP32_shuffle,
                                                                                    this->p_ptr_Class_Device_Information_Array);
        
    tmp_ptr_array_inputs_array += number_data_validation_received;
    tmp_ptr_array_outputs_array += number_data_validation_received;

    this->_ptr_array_Dataset[2u].Reference(number_data_testing_received,
                                                                                    this->p_number_inputs,
                                                                                    this->p_number_outputs,
                                                                                    this->p_number_recurrent_depth,
                                                                                    tmp_ptr_array_inputs_array,
                                                                                    tmp_ptr_array_outputs_array,
                                                                                    this->p_number_cuRAND_State_MTGP32_shuffle,
                                                                                    this->ptr_array_cuRAND_State_MTGP32_shuffle,
                                                                                    this->p_ptr_Class_Device_Information_Array);

    this->_type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING;

    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__Dataset_Manager__Prepare_Storage(T_ const number_data_percent_training_received,
                                                                                              T_ const number_data_percent_testing_received,
                                                                                              class CUDA_Dataset_Manager<T> *const ptr_Dataset_Manager_received)
{ ptr_Dataset_Manager_received->Prepare_Storage(number_data_percent_training_received, number_data_percent_testing_received); }
template __global__ void kernel__Dataset_Manager__Prepare_Storage(T_ const,
                                                                                                            T_ const,
                                                                                                            class CUDA_Dataset_Manager<T_> *const);

template<typename T>
__host__ __device__ bool CUDA_Dataset_Manager<T>::Prepare_Storage(T_ const number_data_percent_training_received, T_ const number_data_percent_testing_received)
{
    if(number_data_percent_training_received + number_data_percent_testing_received != 100_T)
    {
        PRINT_FORMAT("%s: ERROR: training(%f%%) + testing(%f%%) != 100.0%%" NEW_LINE,
                    __FUNCTION__,
                    number_data_percent_training_received,
                    number_data_percent_testing_received);

        return(false);
    }
    else if(number_data_percent_training_received == 0_T)
    {
        PRINT_FORMAT("%s: ERROR: training(%f%%) == 0.0%%" NEW_LINE,
                    __FUNCTION__,
                    number_data_percent_training_received);

        return(false);
    }
    else if(number_data_percent_testing_received == 0_T)
    {
        PRINT_FORMAT("%s: ERROR: testing(%f%%) == 0.0%%" NEW_LINE,
                    __FUNCTION__,
                    number_data_percent_testing_received);

        return(false);
    }
        
#if defined(__CUDA_ARCH__) == false
    kernel__Dataset_Manager__Prepare_Storage<T> <<< 1u, 1u >>> (number_data_percent_training_received,
                                                                                                         number_data_percent_testing_received,
                                                                                                         this);

#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    if(this->Get__Number_Examples() < 2u)
    {
        PRINT_FORMAT("%s: ERROR: Number of data (%u) < 2" NEW_LINE,
                                 __FUNCTION__,
                                 this->Get__Number_Examples());

        return(false);
    }
    else if(this->_type_storage_data != MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE)
    {
        PRINT_FORMAT("%s: ERROR: Can not prepare storage multiple time." NEW_LINE, __FUNCTION__);

        return(false);
    }
    
    size_t const tmp_number_data_training(Get__Minimum<size_t>(static_cast<size_t>(round(static_cast<double>(this->Get__Number_Examples()) * number_data_percent_training_received / 100.0)), 1u)),
                                tmp_number_data_testing(this->Get__Number_Examples() - tmp_number_data_training);

    T **tmp_ptr_array_inputs_array(this->p_ptr_array_inputs_array),
        **tmp_ptr_array_outputs_array(this->p_ptr_array_outputs_array);
        
    this->_ptr_array_Dataset = new class Dataset_device<T>[2u];

    this->_ptr_array_Dataset[0u].Reference(tmp_number_data_training,
                                                                                    this->p_number_inputs,
                                                                                    this->p_number_outputs,
                                                                                    this->p_number_recurrent_depth,
                                                                                    tmp_ptr_array_inputs_array,
                                                                                    tmp_ptr_array_outputs_array,
                                                                                    this->p_number_cuRAND_State_MTGP32_shuffle,
                                                                                    this->ptr_array_cuRAND_State_MTGP32_shuffle,
                                                                                    this->p_ptr_Class_Device_Information_Array);
        
    tmp_ptr_array_inputs_array += tmp_number_data_training;
    tmp_ptr_array_outputs_array += tmp_number_data_training;

    this->_ptr_array_Dataset[1u].Reference(tmp_number_data_testing,
                                                                                    this->p_number_inputs,
                                                                                    this->p_number_outputs,
                                                                                    this->p_number_recurrent_depth,
                                                                                    tmp_ptr_array_inputs_array,
                                                                                    tmp_ptr_array_outputs_array,
                                                                                    this->p_number_cuRAND_State_MTGP32_shuffle,
                                                                                    this->ptr_array_cuRAND_State_MTGP32_shuffle,
                                                                                    this->p_ptr_Class_Device_Information_Array);

    this->_type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING;

    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__Dataset_Manager__Prepare_Storage(T_ const number_data_percent_training_received,
                                                                                            T_ const number_data_percent_validation_received,
                                                                                            T_ const number_data_percent_testing_received,
                                                                                            class CUDA_Dataset_Manager<T> *const ptr_Dataset_Manager_received)
{
    ptr_Dataset_Manager_received->Prepare_Storage(number_data_percent_training_received,
                                                                                number_data_percent_validation_received,
                                                                                number_data_percent_testing_received);
}
template __global__ void kernel__Dataset_Manager__Prepare_Storage(T_ const,
                                                                                                            T_ const,
                                                                                                            T_ const,
                                                                                                            class CUDA_Dataset_Manager<T_> *const);

template<typename T>
__host__ __device__ bool CUDA_Dataset_Manager<T>::Prepare_Storage(T_ const number_data_percent_training_received,
                                                                                                                T_ const number_data_percent_validation_received,
                                                                                                                T_ const number_data_percent_testing_received)
{
    if(number_data_percent_training_received + number_data_percent_validation_received + number_data_percent_testing_received != 100_T)
    {
        PRINT_FORMAT("%s: ERROR: training(%f%%) + validation(%f%%) + testing(%f%%) != 100.0%%" NEW_LINE,
                                __FUNCTION__,
                                number_data_percent_training_received,
                                number_data_percent_validation_received,
                                number_data_percent_testing_received);

        return(false);
    }
    else if(number_data_percent_training_received == 0_T)
    {
        PRINT_FORMAT("%s: ERROR: training(%f%%) == 0.0%%" NEW_LINE,
                                __FUNCTION__,
                                number_data_percent_training_received);

        return(false);
    }
    else if(number_data_percent_validation_received == 0_T)
    {
        PRINT_FORMAT("%s: ERROR: validation(%f%%) == 0.0%%" NEW_LINE,
                                __FUNCTION__,
                                number_data_percent_validation_received);

        return(false);
    }
    else if(number_data_percent_testing_received == 0_T)
    {
        PRINT_FORMAT("%s: ERROR: testing(%f%%) == 0.0%%" NEW_LINE,
                                __FUNCTION__,
                                number_data_percent_testing_received);

        return(false);
    }
        
#if defined(__CUDA_ARCH__) == false
    kernel__Dataset_Manager__Prepare_Storage<T> <<< 1u, 1u >>> (number_data_percent_training_received,
                                                                                                                        number_data_percent_validation_received,
                                                                                                                        number_data_percent_testing_received,
                                                                                                                        this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    return(true);
#else
    if(this->Get__Number_Examples() < 3u)
    {
        PRINT_FORMAT("%s: ERROR: Number of data (%u) < 3" NEW_LINE,
                                 __FUNCTION__,
                                 this->Get__Number_Examples());

        return(false);
    }
    else if(this->_type_storage_data != MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE)
    {
        PRINT_FORMAT("%s: ERROR: Can not prepare storage multiple time." NEW_LINE, __FUNCTION__);

        return(false);
    }
        
    size_t const tmp_number_data_training(Get__Limit<size_t>(static_cast<size_t>(round(static_cast<double>(this->Get__Number_Examples()) * number_data_percent_training_received / 100.0)),
                                                                                                          1u,
                                                                                                          this->Get__Number_Examples() - 2u)),
                                tmp_number_data_validation(Get__Limit<size_t>(static_cast<size_t>(round(static_cast<double>(this->Get__Number_Examples()) * number_data_percent_validation_received / 100.0)),
                                                                                                             1u,
                                                                                                             this->Get__Number_Examples() - tmp_number_data_training - 1u)),
                                tmp_number_data_testing(Get__Minimum<size_t>(this->Get__Number_Examples() - tmp_number_data_training - tmp_number_data_validation, 1u));

    T **tmp_ptr_array_inputs_array(this->p_ptr_array_inputs_array),
        **tmp_ptr_array_outputs_array(this->p_ptr_array_outputs_array);
        
    this->_ptr_array_Dataset = new class Dataset_device<T>[3u];

    this->_ptr_array_Dataset[0u].Reference(tmp_number_data_training,
                                                                                    this->p_number_inputs,
                                                                                    this->p_number_outputs,
                                                                                    this->p_number_recurrent_depth,
                                                                                    tmp_ptr_array_inputs_array,
                                                                                    tmp_ptr_array_outputs_array,
                                                                                    this->p_number_cuRAND_State_MTGP32_shuffle,
                                                                                    this->ptr_array_cuRAND_State_MTGP32_shuffle,
                                                                                    this->p_ptr_Class_Device_Information_Array);
        
    tmp_ptr_array_inputs_array += tmp_number_data_training;
    tmp_ptr_array_outputs_array += tmp_number_data_training;

    this->_ptr_array_Dataset[1u].Reference(tmp_number_data_validation,
                                                                                    this->p_number_inputs,
                                                                                    this->p_number_outputs,
                                                                                    this->p_number_recurrent_depth,
                                                                                    tmp_ptr_array_inputs_array,
                                                                                    tmp_ptr_array_outputs_array,
                                                                                    this->p_number_cuRAND_State_MTGP32_shuffle,
                                                                                    this->ptr_array_cuRAND_State_MTGP32_shuffle,
                                                                                    this->p_ptr_Class_Device_Information_Array);
        
    tmp_ptr_array_inputs_array += tmp_number_data_validation;
    tmp_ptr_array_outputs_array += tmp_number_data_validation;

    this->_ptr_array_Dataset[2u].Reference(tmp_number_data_testing,
                                                                                    this->p_number_inputs,
                                                                                    this->p_number_outputs,
                                                                                    this->p_number_recurrent_depth,
                                                                                    tmp_ptr_array_inputs_array,
                                                                                    tmp_ptr_array_outputs_array,
                                                                                    this->p_number_cuRAND_State_MTGP32_shuffle,
                                                                                    this->ptr_array_cuRAND_State_MTGP32_shuffle,
                                                                                    this->p_ptr_Class_Device_Information_Array);
        
    this->_type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING;

    return(true);
#endif
}
    
template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Get__Number_Data(size_t *ptr_number_data_received, class CUDA_Dataset_Manager<T> const *const ptr_CUDA_Dataset_Manager_received)
{ *ptr_number_data_received = ptr_CUDA_Dataset_Manager_received->Get__Number_Examples(); }

template<typename T>
__host__ __device__ size_t CUDA_Dataset_Manager<T>::Get__Number_Examples(void) const
{
#if defined(__CUDA_ARCH__) == false
    size_t tmp_number_examples,
                        *tmp_ptr_device_number_data;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_number_data, sizeof(size_t)));

    kernel__CUDA_Dataset_Manager__Get__Number_Data<T> <<< 1u, 1u >>> (tmp_ptr_device_number_data, this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&tmp_number_examples,
                                                            tmp_ptr_device_number_data,
                                                            sizeof(size_t),
                                                            cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_data));

    return(tmp_number_examples);
#else
    return(this->p_number_examples);
#endif
}

template<typename T>
__host__ __device__ size_t CUDA_Dataset_Manager<T>::Get__Number_Inputs(void) const { return(this->p_number_inputs); }

template<typename T>
__host__ __device__ size_t CUDA_Dataset_Manager<T>::Get__Number_Outputs(void) const { return(this->p_number_outputs); }

template<typename T>
__host__ __device__ size_t CUDA_Dataset_Manager<T>::Get__Number_Recurrent_Depth(void) const { return(this->p_number_recurrent_depth); }
    
template<typename T>
__device__ void CUDA_Dataset_Manager<T>::Training(T_ &ref_loss_received,
                                                                                    T_ &ref_accuracy_received,
                                                                                    class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    class Dataset_device<T> *const tmp_ptr_Dataset_device(this->Get__Dataset_At(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
    
    if(tmp_ptr_Dataset_device == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Get__Dataset_At(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                 __LINE__);

        return;
    }
    else if(tmp_ptr_Dataset_device->Check_Topology(ptr_CNeural_Network_received->number_inputs, ptr_CNeural_Network_received->number_outputs) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Check_Topology(%u, %u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 ptr_CNeural_Network_received->number_inputs,
                                 ptr_CNeural_Network_received->number_outputs,
                                 __LINE__);

        return;
    }

    ptr_CNeural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_TRAINING;
    
    switch(MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_BATCH)
    {
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_BATCH:
            tmp_ptr_Dataset_device->device__Training_Process_Batch(ref_loss_received,
                                                                                                   ref_accuracy_received,
                                                                                                   ptr_CNeural_Network_received);
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_MINI_BATCH:
            tmp_ptr_Dataset_device->device__Training_Process_Mini_Batch_Stochastic(ref_loss_received,
                                                                                                                            ref_accuracy_received,
                                                                                                                            ptr_CNeural_Network_received);
                break;
        case MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_CROSS_VALIDATION:
            tmp_ptr_Dataset_device->device__Training_Process_Cross_Validation_K_Fold(ref_loss_received,
                                                                                                                              ref_accuracy_received,
                                                                                                                              ptr_CNeural_Network_received);
                break;
        default:
            ref_loss_received = 1_T;
            ref_accuracy_received = 0_T;
                break;
    }
    
    ptr_CNeural_Network_received->type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_INFERENCE;
}

template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Training(T_ *const ptr_error_received,
                                                                                            T_ *const ptr_accuracy_received,
                                                                                            class CUDA_Neural_Network *const ptr_CNeural_Network_received,
                                                                                            class CUDA_Dataset_Manager<T> *const ptr_CUDA_Dataset_Manager_received)
{
    ptr_CUDA_Dataset_Manager_received->Training(*ptr_error_received,
                                                                               *ptr_accuracy_received,
                                                                               ptr_CNeural_Network_received);
}

template<typename T>
__host__ T_ CUDA_Dataset_Manager<T>::Training(class Neural_Network *const ptr_Neural_Network_received)
{
    T_ tmp_loss,
            tmp_accuracy,
            *tmp_ptr_device_loss,
            *tmp_ptr_device_accuracy;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_loss, sizeof(T_)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_accuracy, sizeof(T_)));
    
    kernel__CUDA_Dataset_Manager__Training<T> <<< 1u, 1u >>> (tmp_ptr_device_loss,
                                                                                                       tmp_ptr_device_accuracy,
                                                                                                       ptr_Neural_Network_received->ptr_device_Neural_Network,
                                                                                                       this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&tmp_loss,
                                                    tmp_ptr_device_loss,
                                                    sizeof(T_),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&tmp_accuracy,
                                                    tmp_ptr_device_accuracy,
                                                    sizeof(T_),
                                                    cudaMemcpyDeviceToHost));
        
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_loss)); // T_
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_accuracy)); // T_

    ptr_Neural_Network_received->is_update_from_device = false;

    ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_loss);
    ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_accuracy);

    return(ptr_Neural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING));
}

template<typename T>
__device__ void CUDA_Dataset_Manager<T>::device__Type_Testing(T_ &ref_loss_received,
                                                                                                    T_ &ref_accuracy_received,
                                                                                                    enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received,
                                                                                                    class CUDA_Neural_Network *const ptr_CNeural_Network_received)
{
    class Dataset_device<T> *const tmp_ptr_Dataset_device(this->Get__Dataset_At(type_dataset_received));
    
    if(tmp_ptr_Dataset_device == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Get__Dataset_At(%u)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING,
                                 __LINE__);
        
        return;
    }
    
    T_ const tmp_previous_loss(ptr_CNeural_Network_received->Get__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)),
                    tmp_previous_accuracy(ptr_CNeural_Network_received->Get__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING));
        
    tmp_ptr_Dataset_device->device__Testing(ref_loss_received,
                                                                  ref_accuracy_received,
                                                                  ptr_CNeural_Network_received);

    switch(type_dataset_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING:
            ptr_CNeural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ref_loss_received);
            ptr_CNeural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, ref_accuracy_received);
                break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION:
            ptr_CNeural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, ref_loss_received);
            ptr_CNeural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, ref_accuracy_received);
                break;
    }

    // Reset testing loss/accuracy.
    if(type_dataset_received != MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING)
    {
        ptr_CNeural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, tmp_previous_loss);
        ptr_CNeural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, tmp_previous_accuracy);
    }
    // |END| Reset testing loss/accuracy. |END|
}

template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Type_Testing(T_ *const ptr_loss_received,
                                                                                               T_ *const ptr_accuray_received,
                                                                                               enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received,
                                                                                               class CUDA_Neural_Network *const ptr_CNeural_Network_received,
                                                                                               class CUDA_Dataset_Manager<T> *const ptr_CUDA_Dataset_Manager_received)
{
    ptr_CUDA_Dataset_Manager_received->device__Type_Testing(*ptr_loss_received,
                                                                                              *ptr_accuray_received,
                                                                                              type_dataset_received,
                                                                                              ptr_CNeural_Network_received);
}

template<typename T>
__host__ T_ CUDA_Dataset_Manager<T>::Type_Testing(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, class Neural_Network *const ptr_Neural_Network_received)
{
    T_ tmp_loss(0_T),
         tmp_accuracy(0_T),
         *tmp_ptr_device_loss,
         *tmp_ptr_device_accuracy;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_loss, sizeof(T_)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_accuracy, sizeof(T_)));
    
    kernel__CUDA_Dataset_Manager__Type_Testing<T> <<< 1u, 1u >>> (tmp_ptr_device_loss,
                                                                                                          tmp_ptr_device_accuracy,
                                                                                                          type_dataset_received,
                                                                                                          ptr_Neural_Network_received->ptr_device_Neural_Network,
                                                                                                          this);
    
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&tmp_loss,
                                                    tmp_ptr_device_loss,
                                                    sizeof(T_),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&tmp_accuracy,
                                                    tmp_ptr_device_accuracy,
                                                    sizeof(T_),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_loss)); // T_
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_accuracy)); // T_

    switch(type_dataset_received)
    {
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING:
            ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_loss);
            ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING, tmp_accuracy);
                break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION:
            ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, tmp_loss);
            ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION, tmp_accuracy);
                break;
        case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING:
            ptr_Neural_Network_received->Set__Loss(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, tmp_loss);
            ptr_Neural_Network_received->Set__Accuracy(MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING, tmp_accuracy);
                break;
        default:
            PRINT_FORMAT("%s: ERROR: Undefined data type (%u)." NEW_LINE,
                        __FUNCTION__,
                        type_dataset_received);
                break;
    }

    return(tmp_loss);
}
    
template<typename T>
__host__ __device__ MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE CUDA_Dataset_Manager<T>::Get__Type_Storage(void) const { return(this->_type_storage_data); }
    
template<typename T>
__device__ T CUDA_Dataset_Manager<T>::Get__Input_At(size_t const index_received, size_t const sub_index_received) const { return(this->p_ptr_array_inputs_array[index_received][sub_index_received]); }

template<typename T>
__device__ T CUDA_Dataset_Manager<T>::Get__Output_At(size_t const index_received, size_t const sub_index_received) const { return(this->p_ptr_array_outputs_array[index_received][sub_index_received]); }
    
template<typename T>
__device__ T* CUDA_Dataset_Manager<T>::Get__Input_At(size_t const index_received) const { return(this->p_ptr_array_inputs_array[index_received]); }

template<typename T>
__device__ T* CUDA_Dataset_Manager<T>::Get__Output_At(size_t const index_received) const { return(this->p_ptr_array_outputs_array[index_received]); }
    
template<typename T>
__device__ T** CUDA_Dataset_Manager<T>::Get__Input_Array(void) const { return(this->p_ptr_array_inputs_array); }

template<typename T>
__device__ T** CUDA_Dataset_Manager<T>::Get__Output_Array(void) const { return(this->p_ptr_array_outputs_array); }
    
template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Get__Sizeof(size_t *const ptr_size_t_received, class CUDA_Dataset_Manager<T> const *const ptr_CUDA_Dataset_Manager_received)
{ *ptr_size_t_received = ptr_CUDA_Dataset_Manager_received->Get__Sizeof(); }
template __global__ void kernel__CUDA_Dataset_Manager__Get__Sizeof(size_t *const, class CUDA_Dataset_Manager<T_> const *const);

template<typename T>
__host__ __device__ size_t CUDA_Dataset_Manager<T>::Get__Sizeof(void) const
{
    size_t tmp_total_size_t(0_zu);

#if defined(__CUDA_ARCH__) == false
    size_t *tmp_ptr_device_total_size_t;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_total_size_t, sizeof(size_t)));

    kernel__CUDA_Dataset_Manager__Get__Sizeof<T> <<< 1u, 1u >>> (tmp_ptr_device_total_size_t, this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaMemcpy(&tmp_total_size_t,
                                                    tmp_ptr_device_total_size_t,
                                                    sizeof(size_t),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_total_size_t));

    return(tmp_total_size_t);
#else
    tmp_total_size_t += sizeof(class CUDA_Dataset_Manager<T>); // this

    if(this->p_ptr_array_inputs_array != nullptr)
    {
        tmp_total_size_t += this->p_number_examples * sizeof(T*);
        tmp_total_size_t += this->p_number_examples * this->p_number_inputs * sizeof(T);
    }

    if(this->p_ptr_array_outputs_array != nullptr)
    {
        tmp_total_size_t += this->p_number_examples * sizeof(T*);
        tmp_total_size_t += this->p_number_examples * this->p_number_outputs * sizeof(T);
    }

    // TODO: Create into CUDA_Device_Information_Array a function returning sizeof called Get__Sizeof().
    if(this->p_ptr_Class_Device_Information_Array != nullptr)
    {
        tmp_total_size_t += sizeof(class CUDA_Device_Information_Array);

        if(this->p_ptr_Class_Device_Information_Array->Get__Number_CUDA_Devices() != 0u)
        {
            tmp_total_size_t += sizeof(class CUDA_Device_Information); // _ptr_Class_Device_Information_sum
            tmp_total_size_t += sizeof(class CUDA_Device_Information); // _ptr_Class_Device_Information_higher
            tmp_total_size_t += sizeof(class CUDA_Device_Information); // _ptr_Class_Device_Information_lower
            tmp_total_size_t += this->p_ptr_Class_Device_Information_Array->Get__Number_CUDA_Devices() * sizeof(class CUDA_Device_Information); // _ptr_array_Class_Device_Information
        }
    }

    if(this->_ptr_array_Dataset != nullptr)
    {
        switch(this->_type_storage_data)
        {
            case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
                tmp_total_size_t += this->_ptr_array_Dataset[0u].Get__Sizeof();
                    break;
            case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
                tmp_total_size_t += this->_ptr_array_Dataset[0u].Get__Sizeof();
                tmp_total_size_t += this->_ptr_array_Dataset[1u].Get__Sizeof();
                    break;
            case MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
                tmp_total_size_t += this->_ptr_array_Dataset[0u].Get__Sizeof();
                tmp_total_size_t += this->_ptr_array_Dataset[1u].Get__Sizeof();
                tmp_total_size_t += this->_ptr_array_Dataset[2u].Get__Sizeof();
                    break;
            default:
                PRINT_FORMAT("%s: ERROR: Undefined storage type (%u)." NEW_LINE,
                            __FUNCTION__,
                            this->_type_storage_data);
                    break;
        }
    }
    
    // cuRAND.
    if(this->ptr_array_cuRAND_State_MTGP32_shuffle != nullptr)
    {
        tmp_total_size_t += this->p_number_cuRAND_State_MTGP32_shuffle * sizeof(struct curandStateMtgp32);
        tmp_total_size_t += this->p_number_cuRAND_State_MTGP32_shuffle * sizeof(struct mtgp32_kernel_params);
    }
    // |END| cuRAND. |END|

    return(tmp_total_size_t);
#endif
}

template<typename T>
__device__ class Dataset_device<T>* CUDA_Dataset_Manager<T>::Get__Dataset_At(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received) const
{
    if(this->_type_storage_data == MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING)
    { return(&this->_ptr_array_Dataset[0u]); }
    else if(this->_type_storage_data == MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING)
    {
        switch(type_dataset_received)
        {
            case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING: return(&this->_ptr_array_Dataset[0u]);
            case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION: return(&this->_ptr_array_Dataset[0u]);
            case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: return(&this->_ptr_array_Dataset[1u]);
            default:
            #if defined(COMPILE_COUT)
                PRINT_FORMAT("%s: ERROR: Undefined data type (%u)." NEW_LINE,
                                         __FUNCTION__,
                                         type_dataset_received);
            #endif
                    return(nullptr);
        }
    }
    else if(this->_type_storage_data == MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_VALIDATION_TESTING)
    {
        switch(type_dataset_received)
        {
            case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TRAINING: return(&this->_ptr_array_Dataset[0u]);
            case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_VALIDATION: return(&this->_ptr_array_Dataset[1u]);
            case MyEA::Common::ENUM_TYPE_DATASET::TYPE_DATASET_TESTING: return(&this->_ptr_array_Dataset[2u]);
            default:
            #if defined(COMPILE_COUT)
                PRINT_FORMAT("%s: ERROR: Undefined data type (%u)." NEW_LINE,
                                         __FUNCTION__,
                                         type_dataset_received);
            #endif
                    return(nullptr);
        }
    }

    return(nullptr);
}
    
template<typename T>
__host__ void CUDA_Dataset_Manager<T>::static_Deallocate_CUDA_Dataset_Manager(class CUDA_Dataset_Manager<T_> *&ptr_CUDA_Dataset_Manager_received)
{
#if defined(COMPILE_COUT)
    PRINT_FORMAT("%s: GPU: Data: Deallocate." NEW_LINE, MyEA::String::Get__Time().c_str());
#endif

    if(ptr_CUDA_Dataset_Manager_received != NULL && ptr_CUDA_Dataset_Manager_received->Deallocate())
    {
    #if defined(COMPILE_COUT)
        PRINT_FORMAT("%s: GPU: Data: Free." NEW_LINE, MyEA::String::Get__Time().c_str());
    #endif

        CUDA__Safe_Call(cudaFree(ptr_CUDA_Dataset_Manager_received));

        ptr_CUDA_Dataset_Manager_received = NULL;
    }
}

template<typename T>
__global__ void kernel__Dataset_device__Initialize_cuRAND_MTGP32(int const number_states_MTGP32_received,
                                                                                                      struct curandStateMtgp32 *const ptr_curandStateMtgp32_received,
                                                                                                      class Dataset_device<T> *const ptr_Dataset_device_received)
{
    if(ptr_Dataset_device_received->Initialize_cuRAND_MTGP32(number_states_MTGP32_received, ptr_curandStateMtgp32_received) == false)
    {
        PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Initialize_cuRAND_MTGP32(%d, ptr)\" function. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 number_states_MTGP32_received,
                                 __LINE__);
    }
}

template<typename T>
__device__ bool Dataset_device<T>::Initialize_cuRAND_MTGP32(int const number_states_MTGP32_received, struct curandStateMtgp32 *const ptr_curandStateMtgp32_received)
{
    if(number_states_MTGP32_received == 0)
    {
        PRINT_FORMAT("%s: ERROR: Can not initialize cuRAND. Size of the array equal zero." NEW_LINE,
                                __FUNCTION__);

        return(false);
    }

    struct mtgp32_kernel_params *tmp_ptr_array_mtgp32_kernel_params_t;

    // Allocate cuRAND State MTGP32 shuffle.
    struct curandStateMtgp32 *tmp_ptr_array_cuRAND_State_MTGP32_shuffle(new struct curandStateMtgp32[number_states_MTGP32_received]);
    if(tmp_ptr_array_cuRAND_State_MTGP32_shuffle == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 static_cast<size_t>(number_states_MTGP32_received) * sizeof(struct curandStateMtgp32),
                                 __LINE__);

        return(false);
    }
    this->ptr_array_cuRAND_State_MTGP32_shuffle = tmp_ptr_array_cuRAND_State_MTGP32_shuffle;
    // |END| Allocate cuRAND State MTGP32 shuffle. |END|

    // Copy cuRAND State MTGP32 shuffle.
    Memory::Copy_Loop<struct curandStateMtgp32>(ptr_curandStateMtgp32_received,
                                                                            ptr_curandStateMtgp32_received + number_states_MTGP32_received,
                                                                            this->ptr_array_cuRAND_State_MTGP32_shuffle);
    // |END| Copy cuRAND State MTGP32 shuffle. |END|

    // Allocate tmp_ptr_array_mtgp32_kernel_params_t.
    tmp_ptr_array_mtgp32_kernel_params_t = new struct mtgp32_kernel_params[number_states_MTGP32_received];
    if(tmp_ptr_array_mtgp32_kernel_params_t == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 static_cast<size_t>(number_states_MTGP32_received) * sizeof(struct mtgp32_kernel_params),
                                 __LINE__);

        return(false);
    }
    // |END| Allocate tmp_ptr_array_mtgp32_kernel_params_t. |END|
    
    // Assign cuRAND State MTGP32 shuffle variable.
    struct dim3 tmp_dim3_grid(1u, 1u, 1u),
                     tmp_dim3_block(1u, 1u, 1u);

    if(USE_PARALLEL && number_states_MTGP32_received >= warpSize)
    {
        this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(static_cast<size_t>(number_states_MTGP32_received),
                                                                                                                                               0_zu,
                                                                                                                                               tmp_dim3_grid,
                                                                                                                                               tmp_dim3_block);
    }

    cuRAND__Memcpy_cuRAND_State_MTGP32(number_states_MTGP32_received,
                                                                        tmp_ptr_array_cuRAND_State_MTGP32_shuffle,
                                                                        ptr_curandStateMtgp32_received,
                                                                        tmp_ptr_array_mtgp32_kernel_params_t,
                                                                        &tmp_dim3_grid,
                                                                        &tmp_dim3_block);

    this->p_number_cuRAND_State_MTGP32_shuffle = number_states_MTGP32_received;
    // |END| Assign cuRAND State MTGP32 shuffle variable. |END|

    return(true);
}

template<typename T>
__global__ void kernel__Dataset_device__Total_Blocks_cuRAND_MTGP32(int *const ptr_number_states_MTGP32_received, class Dataset_device<T> *ptr_Dataset_device_received)
{
    double const tmp_number_blocks(ceil(static_cast<double>(ptr_Dataset_device_received->Get__Total_Data()) / static_cast<double>(ptr_Dataset_device_received->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Get__Warp_Size())));

    if(tmp_number_blocks > (std::numeric_limits<int>::max)())
    {
        PRINT_FORMAT("%s: ERROR: Overflow conversion (%f) to int (%d). At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 tmp_number_blocks,
                                 (std::numeric_limits<int>::max)(),
                                 __LINE__);
    }

    *ptr_number_states_MTGP32_received = static_cast<int>(ceil(tmp_number_blocks / 256.0));
}

template<typename T>
__host__ bool Dataset_device<T>::Initialize_cuRAND(size_t const seed_received)
{
    int tmp_number_states_MTGP32,
         *tmp_ptr_device_number_states_MTGP32(nullptr);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_number_states_MTGP32, sizeof(int)));
    
    kernel__Dataset_device__Total_Blocks_cuRAND_MTGP32<T> <<< 1u, 1u >>> (tmp_ptr_device_number_states_MTGP32, this);
    
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
            PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Allocate_cuRAND_MTGP32(%d, %zu, ptr, ptr)\" function. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     tmp_number_states_MTGP32,
                                     seed_received,
                                     __LINE__);
            
            CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));

            return(false);
        }
        
        kernel__Dataset_device__Initialize_cuRAND_MTGP32 <<< 1u, 1u >>> (tmp_number_states_MTGP32,
                                                                                                                tmp_ptr_curandStateMtgp32_t,
                                                                                                                this);
            
    #if defined(COMPILE_DEBUG)
        CUDA__Check_Error();
    #endif

        Cleanup_cuRAND_MTGP32(tmp_ptr_mtgp32_kernel_params, tmp_ptr_curandStateMtgp32_t);
    }
    
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));
    
    return(true);
}

template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Initialize_cuRAND_MTGP32(int const number_states_MTGP32_received,
                                                                                                                    struct curandStateMtgp32 *const ptr_curandStateMtgp32_received,
                                                                                                                    class CUDA_Dataset_Manager<T> *const ptr_CUDA_Dataset_Manager_received)
{
    if(ptr_CUDA_Dataset_Manager_received->Initialize_cuRAND_MTGP32(number_states_MTGP32_received, ptr_curandStateMtgp32_received) == false)
    {
            PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Initialize_cuRAND_MTGP32(%d, ptr)\" function. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     number_states_MTGP32_received,
                                     __LINE__);
    }
}

template<typename T>
__device__ bool CUDA_Dataset_Manager<T>::Initialize_cuRAND_MTGP32(int const number_states_MTGP32_received, struct curandStateMtgp32 *const ptr_curandStateMtgp32_received)
{
    if(number_states_MTGP32_received == 0u)
    {
        PRINT_FORMAT("%s: ERROR: Can not initialize cuRAND. Size of the array equal zero." NEW_LINE,
                                __FUNCTION__);

        return(false);
    }

    struct mtgp32_kernel_params *tmp_ptr_array_mtgp32_kernel_params_t;

    // Allocate cuRAND State MTGP32 shuffle.
    struct curandStateMtgp32 *tmp_ptr_array_cuRAND_State_MTGP32_shuffle(new struct curandStateMtgp32[number_states_MTGP32_received]);
    if(tmp_ptr_array_cuRAND_State_MTGP32_shuffle == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 static_cast<size_t>(number_states_MTGP32_received) * sizeof(struct curandStateMtgp32),
                                 __LINE__);

        return(false);
    }
    this->ptr_array_cuRAND_State_MTGP32_shuffle = tmp_ptr_array_cuRAND_State_MTGP32_shuffle;
    // |END| Allocate cuRAND State MTGP32 shuffle. |END|
    
    // Copy cuRAND State MTGP32 shuffle.
    Memory::Copy_Loop<struct curandStateMtgp32>(ptr_curandStateMtgp32_received,
                                                                            ptr_curandStateMtgp32_received + number_states_MTGP32_received,
                                                                            this->ptr_array_cuRAND_State_MTGP32_shuffle);
    // |END| Copy cuRAND State MTGP32 shuffle. |END|
    
    // Allocate tmp_ptr_array_mtgp32_kernel_params_t.
    tmp_ptr_array_mtgp32_kernel_params_t = new struct mtgp32_kernel_params[number_states_MTGP32_received];
    if(tmp_ptr_array_mtgp32_kernel_params_t == nullptr)
    {
        PRINT_FORMAT("%s: ERROR: Can not allocate %zu bytes. At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 static_cast<size_t>(number_states_MTGP32_received) * sizeof(struct mtgp32_kernel_params),
                                 __LINE__);

        return(false);
    }
    // |END| Allocate tmp_ptr_array_mtgp32_kernel_params_t. |END|
    
    // Assign cuRAND State MTGP32 shuffle variable.
    struct dim3 tmp_dim3_grid(1u, 1u, 1u),
                     tmp_dim3_block(1u, 1u, 1u);

    if(USE_PARALLEL && number_states_MTGP32_received >= warpSize)
    {
        this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(static_cast<size_t>(number_states_MTGP32_received),
                                                                                                                                               0_zu,
                                                                                                                                                tmp_dim3_grid,
                                                                                                                                                tmp_dim3_block);
    }
    
    cuRAND__Memcpy_cuRAND_State_MTGP32(number_states_MTGP32_received,
                                                                        tmp_ptr_array_cuRAND_State_MTGP32_shuffle,
                                                                        ptr_curandStateMtgp32_received,
                                                                        tmp_ptr_array_mtgp32_kernel_params_t,
                                                                        &tmp_dim3_grid,
                                                                        &tmp_dim3_block);

    this->p_number_cuRAND_State_MTGP32_shuffle = number_states_MTGP32_received;
    // |END| Assign cuRAND State MTGP32 shuffle variable. |END|
    
    return(true);
}

template<typename T>
__global__ void kernel__CUDA_Dataset_Manager__Total_Blocks_cuRAND_MTGP32(int *const ptr_number_states_MTGP32_received, class CUDA_Dataset_Manager<T> const *const ptr_CUDA_Dataset_Manager_received)
{
    double const tmp_number_blocks(ceil(static_cast<double>(ptr_CUDA_Dataset_Manager_received->Get__Number_Examples()) / static_cast<double>(ptr_CUDA_Dataset_Manager_received->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Get__Warp_Size())));
    
    if(tmp_number_blocks > (std::numeric_limits<int>::max)())
    {
        PRINT_FORMAT("%s: ERROR: Overflow conversion (%f) to int (%d). At line %d." NEW_LINE,
                                 __FUNCTION__,
                                 tmp_number_blocks,
                                 (std::numeric_limits<int>::max)(),
                                 __LINE__);
    }

    *ptr_number_states_MTGP32_received = static_cast<int>(ceil(tmp_number_blocks / 256.0));
}

template<typename T>
__host__ bool CUDA_Dataset_Manager<T>::Initialize_cuRAND(size_t const seed_received)
{
    int tmp_number_states_MTGP32,
         *tmp_ptr_device_number_states_MTGP32(nullptr);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_number_states_MTGP32, sizeof(int)));
    
    kernel__CUDA_Dataset_Manager__Total_Blocks_cuRAND_MTGP32<T> <<< 1u, 1u >>> (tmp_ptr_device_number_states_MTGP32, this);
    
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
            PRINT_FORMAT("%s: ERROR: An error has been triggered from the \"Allocate_cuRAND_MTGP32(%d, %zu, ptr, ptr)\" function. At line %d." NEW_LINE,
                                     __FUNCTION__,
                                     tmp_number_states_MTGP32,
                                     seed_received,
                                     __LINE__);
            
            CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));

            return(false);
        }
        
        kernel__CUDA_Dataset_Manager__Initialize_cuRAND_MTGP32 <<< 1u, 1u >>> (tmp_number_states_MTGP32,
                                                                                                                               tmp_ptr_curandStateMtgp32_t,
                                                                                                                               this);
            
    #if defined(COMPILE_DEBUG)
        CUDA__Check_Error();
    #endif

        Cleanup_cuRAND_MTGP32(tmp_ptr_mtgp32_kernel_params, tmp_ptr_curandStateMtgp32_t);
    }
    
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));
    
    return(true);
}

// template initialization declaration.
template class Dataset_device<T_>;
template class CUDA_Dataset_Manager<T_>;
