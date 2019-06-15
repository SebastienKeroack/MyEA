#pragma once

#include <device_launch_parameters.h>

#include <Tools/Configuration.hpp>
#include <Enums/Enum_Type_Dataset.hpp>
#include <Enums/Enum_Type_Dataset_File.hpp>
#include <Enums/Enum_Type_Dataset_Process.hpp>
#include <Enums/Enum_Type_Dataset_Manager_Storage.hpp>

// Forward declaration
template<typename T> class Dataset_Manager;
template<typename T> class CUDA_Dataset_Manager;
// |END| Forward declaration. |END|

template<typename T>
class Dataset_device
{
    protected:
        size_t p_number_examples = 0;
        size_t p_number_recurrent_depth = 0;
        size_t p_number_inputs = 0;
        size_t p_number_outputs = 0;
        
        T **p_ptr_array_inputs_array = nullptr; // Size[D], Size[I].
        T **p_ptr_array_outputs_array = nullptr; // Size[D], Size[O].
        
        struct dim3 *ptr_array_dim3_grid_batch = nullptr; // Size[1].
        struct dim3 *ptr_array_dim3_block_batch = nullptr; // Size[1].
        struct dim3 *ptr_array_dim3_grid_batch_fold = nullptr; // Size[1].
        struct dim3 *ptr_array_dim3_block_batch_fold = nullptr; // Size[1].
        struct dim3 *ptr_array_dim3_grid_shuffle = nullptr; // Size[1].
        struct dim3 *ptr_array_dim3_block_shuffle = nullptr; // Size[1].
        struct dim3 *ptr_array_dim3_grid_index_transposed = nullptr; // Size[1].
        struct dim3 *ptr_array_dim3_block_index_transposed = nullptr; // Size[1].

        enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS p_type_dataset_process = MyEA::Common::ENUM_TYPE_DATASET_PROCESS::TYPE_DATASET_PROCESS_NONE;
        
        class CUDA_Device_Information_Array *p_ptr_Class_Device_Information_Array = nullptr; // Ptr.
        
        // Mini-Batch Stochastic
        bool use_shuffle = true;

        size_t p_number_mini_batch = 0;
        size_t p_number_examples_mini_batch = 0;
        size_t p_number_examples_per_iteration = 0;
        size_t p_number_examples_last_iteration = 0;
        size_t *ptr_array_stochastic_index = nullptr;

        T **ptr_array_inputs_array_stochastic = nullptr;
        T **ptr_array_outputs_array_stochastic = nullptr;
        // - Mini-Batch Stochastic -

        // Cross Validation k-fold
        size_t number_examples_k_fold = 0;
        size_t number_k_fold = 0;
        size_t number_k_sub_fold = 0;
        size_t number_examples_per_fold = 0;
        size_t number_examples_training = 0;
        size_t number_examples_validating = 0;
        size_t number_examples_per_sub_iteration = 0;
        size_t number_examples_last_sub_iteration = 0;

        T **ptr_array_inputs_array_k_fold = nullptr;
        T **ptr_array_outputs_array_k_fold = nullptr;
        T **ptr_array_inputs_array_k_sub_fold = nullptr;
        T **ptr_array_outputs_array_k_sub_fold = nullptr;

        class Dataset_device<T> *ptr_Validation_Dataset = nullptr;
        // - Cross Validation k-fold -

        // cuRAND.
        size_t p_number_cuRAND_State_MTGP32_shuffle = 0;
        size_t p_number_blocks_shuffle = 0;

        struct curandStateMtgp32 *ptr_array_cuRAND_State_MTGP32_shuffle = nullptr;
        // |END| cuRAND. |END|

    public:
        __host__ __device__ Dataset_device(void);
        __host__ Dataset_device(enum MyEA::Common::ENUM_TYPE_DATASET_FILE const type_data_file_read_received, std::string const &ref_path_received);
        __host__ __device__ ~Dataset_device(void);

        __device__ class Dataset_device<T>& operator=(class Dataset_device<T> const &ref_Dataset_received);

        __device__ void Copy(class Dataset_device<T> const &ref_Dataset_received);
        __device__ void Copy(class CUDA_Dataset_Manager<T> const &ref_Dataset_Manager_received);
        __device__ void Reference(size_t const number_examples_received,
                                                               size_t const number_inputs_received,
                                                               size_t const number_outputs_received,
                                                               size_t const number_recurrent_depth_received,
                                                               T **const ptr_array_inputs_array_received,
                                                               T **const ptr_array_outputs_array_received,
                                                               size_t const number_cuRAND_State_MTGP32_shuffle_received,
                                                               struct curandStateMtgp32 *const ptr_cuRAND_State_MTGP32_received,
                                                               class CUDA_Device_Information_Array *const ptr_Class_Device_Information_Array_received);
        __device__ void Train_Epoch_Batch(class CUDA_Neural_Network *const ptr_CNeural_Network_received);
        __device__ void Train_Batch_Batch(class CUDA_Neural_Network *const ptr_CNeural_Network_received);
        
        // Mini-Batch Stochastic
        __device__ void Train_Epoch_Mini_Batch_Stochastic(class CUDA_Neural_Network *const ptr_CNeural_Network_received);
        __device__ void Train_Batch_Mini_Batch_Stochastic(class CUDA_Neural_Network *const ptr_CNeural_Network_received);
        // - Mini-Batch Stochastic -
        
        // Cross Validation k-fold
        __device__ void Train_Epoch_Cross_Validation_K_Fold(class CUDA_Neural_Network *const ptr_CNeural_Network_received);
        __device__ void Train_Batch_Cross_Validation_K_Fold(class CUDA_Neural_Network *const ptr_CNeural_Network_received);
        // - Cross Validation k-fold -

        __device__ bool device_Allocate(size_t const number_examples_received,
                                                        size_t const number_inputs_received,
                                                        size_t const number_outputs_received,
                                                        size_t const number_recurrent_depth_received,
                                                        T const *ptr_array_inputs_received,
                                                        T const *ptr_array_outputs_received,
                                                        class CUDA_Device_Information *const ptr_Class_Device_Information_received);
        __device__ bool Allocate_Dim3(void);
        __host__ bool Initialize_CUDA_Device(void);
        __host__ bool Initialize_cuRAND(size_t const seed_received);
        __device__ bool Initialize_cuRAND_MTGP32(int const size_received, struct curandStateMtgp32 *const ptr_curandStateMtgp32);
        __device__ bool Add_CUDA_Device(int const index_device_received, struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received);
        __device__ bool Check_Topology(size_t const &ref_number_inputs_received, size_t const &ref_number_outputs_received) const;
        __host__ __device__ bool Initialize(enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_gradient_descent_received);
        __host__ __device__ bool Initialize_Mini_Batch_Stochastic_Gradient_Descent(bool const use_shuffle_received,
                                                                                                                            size_t const desired_number_examples_per_mini_batch_received,
                                                                                                                            size_t const number_mini_batch_maximum_received);
        __host__ __device__ bool Initialize__Cross_Validation(bool const use_shuffle_received,
                                                                                                    size_t const number_k_fold_received,
                                                                                                    size_t const number_k_sub_fold_received,
                                                                                                    class CUDA_Dataset_Manager<T> const *const ptr_CUDA_Dataset_Manager_received);
        __host__ __device__ bool Initialize__Cross_Validation(class CUDA_Dataset_Manager<T> const *const ptr_CUDA_Dataset_Manager_received);
        __host__ __device__ bool Set__Type_Gradient_Descent(enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_gradient_descent_received);
        __host__ __device__ bool Deallocate(void);
            
        __host__ __device__ bool Get__Use__Shuffle(void) const;

        __host__ __device__ size_t Get__Total_Data(void) const;
        __host__ __device__ size_t Get__Number_Examples(void) const;
        __host__ __device__ size_t Get__Number_CV_K_Fold(void) const;
        __host__ __device__ size_t Get__Number_CV_K_Sub_Fold(void) const;
        __host__ __device__ size_t Get__Number_CV_Data_Per_Fold(void) const;
        __host__ __device__ size_t Get__Number_CV_Data_Training(void) const;
        __host__ __device__ size_t Get__Number_CV_Data_Validating(void) const;
        __host__ __device__ size_t Get__Number_CV_Data_Per_Sub_Iteration(void) const;
        __host__ __device__ size_t Get__Number_CV_Data_Last_Sub_Iteration(void) const;
        __host__ __device__ size_t Get__Number_Inputs(void) const;
        __host__ __device__ size_t Get__Number_Outputs(void) const;
        __host__ __device__ size_t Get__Number_Recurrent_Depth(void) const;
            
        __host__ float Training_Process_Batch(class Neural_Network *const ptr_Neural_Network_received);
        __device__ void device__Training_Process_Batch(float &ref_loss_received,
                                                                                float &ref_accuracy_received,
                                                                                class CUDA_Neural_Network *const ptr_CNeural_Network_received);

        // Mini-Batch Stochastic
        __host__ float Training_Process_Mini_Batch_Stochastic(class Neural_Network *const ptr_Neural_Network_received);
        __device__ void device__Training_Process_Mini_Batch_Stochastic(float &ref_loss_received,
                                                                                                        float &ref_accuracy_received,
                                                                                                        class CUDA_Neural_Network *const ptr_CNeural_Network_received);
        // - Mini-Batch Stochastic -

        // Cross Validation k-fold
        __host__ float Training_Process_Cross_Validation_K_Fold(class Neural_Network *const ptr_Neural_Network_received);
        __device__ void device__Training_Process_Cross_Validation_K_Fold(float &ref_loss_received,
                                                                                                            float &ref_accuracy_received,
                                                                                                            class CUDA_Neural_Network *const ptr_CNeural_Network_received);
        // - Cross Validation k-fold -
        __host__ float Testing(class Neural_Network *const ptr_Neural_Network_received);
        __device__ void device__Testing(float &ref_loss_received,
                                                        float &ref_accuracy_received,
                                                        class CUDA_Neural_Network *const ptr_CNeural_Network_received);

        __host__ __device__ enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS Get__Type_Dataset_Process(void) const;

        __device__ T Get__Input_At(size_t const index_received, size_t const sub_index_received) const;
        __device__ T Get__Output_At(size_t const index_received, size_t const sub_index_received) const;
        __device__ T *Get__Input_At(size_t const index_received) const;
        __device__ T *Get__Output_At(size_t const index_received) const;
        __device__ T **Get__Input_Array(void) const;
        __device__ T **Get__Output_Array(void) const;

        __host__ __device__ size_t Get__Sizeof(void) const;

        // Mini-Batch Stochastic
        __device__ void Mini_Batch_Stochastic__Reset(void);
        // - Mini-Batch Stochastic -

        // Cross Validation k-fold
        __device__ void Cross_Validation_K_Fold__Reset(void);
        // - Cross Validation k-fold -
        
        __device__ class CUDA_Device_Information_Array *Get__Class_Device_Information_Array(void) const;

    private:
        // Mini-Batch Stochastic
        __device__ void Mini_Batch_Stochastic__Initialize_Shuffle(void);
        __device__ void Mini_Batch_Stochastic__Shuffle(void);
        __device__ bool Mini_Batch_Stochastic__Increment_Mini_Batch(size_t const mini_batch_iteration_received);
        // - Mini-Batch Stochastic -

        // Cross Validation k-fold
        __device__ void Cross_Validation_K_Fold__Initialize_Shuffle(void);
        __device__ void Cross_Validation_K_Fold__Shuffle(void);
        __device__ bool Cross_Validation_K_Fold__Increment_Fold(size_t const fold_received);
        __device__ bool Cross_Validation_K_Fold__Increment_Sub_Fold(size_t const fold_sub_received);
        __device__ float Test_Epoch_Cross_Validation_K_Fold(class CUDA_Neural_Network *ptr_CNeural_Network_received);
        // - Cross Validation k-fold -

        bool _reference = false;
};

template<typename T>
class CUDA_Dataset_Manager
{
    protected:
        size_t p_number_examples = 0;
        size_t p_number_recurrent_depth = 0;
        size_t p_number_inputs = 0;
        size_t p_number_outputs = 0;

        T **p_ptr_array_inputs_array = nullptr;
        T **p_ptr_array_outputs_array = nullptr;

        // cuRAND.
        int p_number_cuRAND_State_MTGP32_shuffle = 0;

        struct curandStateMtgp32 *ptr_array_cuRAND_State_MTGP32_shuffle = nullptr;
        // |END| cuRAND. |END|
        
        class CUDA_Device_Information_Array *p_ptr_Class_Device_Information_Array = nullptr; // Ptr.
        
    public:
        __host__ __device__ CUDA_Dataset_Manager(void);
        __host__ __device__ ~CUDA_Dataset_Manager(void);

        __host__ static void static_Deallocate_CUDA_Dataset_Manager(class CUDA_Dataset_Manager<T_> *&ptr_CUDA_Dataset_Manager_received);
        
        __host__ bool Copy(class Dataset_Manager<T> *const ptr_Dataset_Manager_received);
        __device__ bool device_Copy(size_t const number_examples_received,
                                                   size_t const number_inputs_received,
                                                   size_t const number_outputs_received,
                                                   size_t const number_recurrent_depth_received,
                                                   T const *ptr_array_inputs_received,
                                                   T const *ptr_array_outputs_received,
                                                   class CUDA_Device_Information *const ptr_Class_Device_Information_received);
        __host__ bool Initialize_CUDA_Device(void);
        __host__ bool Initialize_cuRAND(size_t const seed_received);
        __device__ bool Initialize_cuRAND_MTGP32(int const size_received, struct curandStateMtgp32 *const ptr_curandStateMtgp32);
        __device__ bool Add_CUDA_Device(int const index_device_received, struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received);
        __host__ __device__ bool Deallocate(void);
        __host__ __device__ bool Initialize(void);
        __host__ __device__ bool Initialize(enum MyEA::Common::ENUM_TYPE_DATASET const type_data_received, enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_gradient_descent_received);
        __host__ __device__ bool Initialize_Mini_Batch_Stochastic_Gradient_Descent(bool const use_shuffle_received,
                                                                                                                          size_t const desired_number_examples_per_mini_batch_received,
                                                                                                                          size_t const number_mini_batch_maximum_received);
        __host__ __device__ bool Initialize__Cross_Validation(bool const use_shuffle_received,
                                                                                                 size_t const number_k_fold_received,
                                                                                                 size_t const number_k_sub_fold_received);
        __host__ __device__ bool Initialize__Cross_Validation(void);
        __host__ __device__ bool Set__Type_Gradient_Descent(enum MyEA::Common::ENUM_TYPE_DATASET const type_data_received, enum MyEA::Common::ENUM_TYPE_DATASET_PROCESS const type_gradient_descent_received);
        __host__ __device__ bool Prepare_Storage(void);
        __host__ __device__ bool Prepare_Storage(size_t const number_examples_training_received, size_t const number_examples_testing_received);
        __host__ __device__ bool Prepare_Storage(size_t const number_examples_training_received,
                                                                        size_t const number_examples_validation_received,
                                                                        size_t const number_examples_testing_received);
        __host__ __device__ bool Prepare_Storage(float const number_examples_percent_training_received, float const number_examples_percent_testing_received);
        __host__ __device__ bool Prepare_Storage(float const number_examples_percent_training_received,
                                                                        float const number_examples_percent_validation_received,
                                                                        float const number_examples_percent_testing_received);
            
        __host__ __device__ size_t Get__Number_Examples(void) const;
        __host__ __device__ size_t Get__Number_Inputs(void) const;
        __host__ __device__ size_t Get__Number_Outputs(void) const;
        __host__ __device__ size_t Get__Number_Recurrent_Depth(void) const;

        __device__ void Training(float &ref_loss_received,
                                            float &ref_accuracy_received,
                                            class CUDA_Neural_Network *const ptr_CNeural_Network_received);
        __host__ float Training(class Neural_Network *const ptr_Neural_Network_received);
        __host__ float Type_Testing(enum MyEA::Common::ENUM_TYPE_DATASET const type_data_received, class Neural_Network *const ptr_Neural_Network_received);
        __device__ void device__Type_Testing(float &ref_loss_received,
                                                                float &ref_accuracy_received,
                                                                enum MyEA::Common::ENUM_TYPE_DATASET const type_data_received,
                                                                class CUDA_Neural_Network *const ptr_CNeural_Network_received);
            
        __host__ __device__ enum MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE Get__Type_Storage(void) const;
            
        __device__ T Get__Input_At(size_t const index_received, size_t const sub_index_received) const;
        __device__ T Get__Output_At(size_t const index_received, size_t const sub_index_received) const;
        __device__ T *Get__Input_At(size_t const index_received) const;
        __device__ T *Get__Output_At(size_t const index_received) const;
        __device__ T **Get__Input_Array(void) const;
        __device__ T **Get__Output_Array(void) const;
            
        __host__ __device__ size_t Get__Sizeof(void) const;
            
        __device__ class Dataset_device<T> *Get__Dataset_At(enum MyEA::Common::ENUM_TYPE_DATASET const type_storage_received) const;
        
        __device__ class CUDA_Device_Information_Array *Get__Class_Device_Information_Array(void) const;

    private:
        enum MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE _type_storage_data = MyEA::Common::ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE;
            
        class Dataset_device<T> *_ptr_array_Dataset = nullptr;
};