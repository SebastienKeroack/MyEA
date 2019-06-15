#pragma once

//#include <device_launch_parameters.h>
#include <Tools/CUDA_Configuration.cuh>

#include <Enums/Enum_Type_Networks.hpp>
#include <Enums/Enum_Type_Layer.hpp>
#include <Enums/Enum_Type_Layer_Activation.hpp>
#include <Enums/Enum_Type_Layer_Dropout.hpp>
#include <Enums/Enum_Type_Layer_Normalization.hpp>
#include <Enums/Enum_Type_Loss_Functions.hpp>
#include <Enums/Enum_Type_Optimizer_Functions.hpp>
#include <Enums/Enum_Type_State_Propagation.hpp>
#include <Enums/Enum_Type_Dataset.hpp>

namespace MyEA
{
    namespace Common
    {
        enum ENUM_TYPE_ACTIVATION_FUNCTION : unsigned int;
        enum ENUM_TYPE_LAYER : unsigned int;
    }
}
// |END| Forward declaration. |END|

#define UNIFIED_MEMORY

class CUDA_Device_Information
{
    // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
    // Variable from "cudaDeviceProp" in "driver_types.h"

    protected:
        char p_name[256u];
            
        bool p_device_overlap = false; // [Deprecated]
        bool p_kernel_execute_timeout_enabled = false;
        bool p_integrated = false;
        bool p_can_map_host_memory = false;
        bool p_concurrent_kernels = false;
        bool p_ECC_enabled = false;
        bool p_TCC_driver = false;
        bool p_unified_addressing = false;
        bool p_stream_priorities_supported = false;
        bool p_global_L1_cache_supported = false;
        bool p_local_L1_cache_supported = false;
        bool p_managed_memory = false;
        bool p_is_multi_gpu_board = false;
        bool p_host_native_atomic_supported = false;
        bool p_pageable_memory_access = false;
        bool p_concurrent_managed_access = false;
        bool p_compute_preemption_supported = false;
        bool p_can_use_host_pointer_for_registered_memory = false;
        bool p_cooperative_launch = false;
        bool p_cooperative_multi_device_launch = false;
            
        int p_major_compute_capability = -1;
        int p_minor_compute_capability = -1;
        size_t p_warp_size = 0;
        size_t p_number_multiprocessor = 0;
        size_t p_maximum_threads_per_block = 0;
        size_t p_maximum_threads_per_multiprocessor = 0;
        size_t p_registers_per_block = 0; // 32-bit
        size_t p_registers_per_multiprocessor = 0; // 32-bit
        size_t p_maximum_threads_dimension[3u];
        size_t p_maximum_grid_size[3u];
        size_t p_clock_rate = 0; // Kilohertz.
        int p_compute_mode = 0;
        size_t p_maximum_texture_1D = 0;
        size_t p_maximum_texture_1D_mipmap = 0;
        size_t p_maximum_texture_1D_linear = 0;
        size_t p_maximum_texture_2D[2u];
        size_t p_maximum_texture_2D_mipmap[2u];
        size_t p_maximum_texture_2D_linear[3u];
        size_t p_maximum_texture_2D_gather[2u];
        size_t p_maximum_texture_3D[3u];
        size_t p_maximum_texture_3D_alternate[3u];
        size_t p_maximum_texture_cubemap = 0;
        size_t p_maximum_texture_1D_layered[2u];
        size_t p_maximum_texture_2D_layered[3u];
        size_t p_maximum_texture_cubemap_layered[2u];
        size_t p_maximum_surface_1D = 0;
        size_t p_maximum_surface_2D[2u];
        size_t p_maximum_surface_3D[3u];
        size_t p_maximum_surface_1D_layered[2u];
        size_t p_maximum_surface_2D_layered[3u];
        size_t p_maximum_surface_cubemap = 0;
        size_t p_maximum_surface_cubemap_layered[2u];
        int p_PCI_bus_ID = 0;
        int p_PCI_device_ID = 0;
        int p_PCI_domain_ID = 0;
        size_t p_async_engine_count = 0;
        size_t p_memory_clock_rate = 0; // Kilohertz.
        size_t p_memory_bus_width = 0; // Bits.
        size_t p_L2_cache_size = 0; // Bytes.
        int p_multi_gpu_board_group_ID = 0;
        int p_single_to_double_precision_performance_ratio = 0;

        size_t p_minimum_threads_for_occupancy = 0;
        size_t p_minimum_threads_for_occupancy_custom = 0;
        size_t p_maximum_number_threads = 0;
        size_t p_number_concurrent_kernel = 0;
        size_t p_number_CUDA_cores = 0;
        size_t p_number_CUDA_cores_per_multiprocessor = 0;
        size_t p_maximum_blocks_per_multiprocessor = 0;
        size_t p_maximum_number_warps_per_multiprocessor = 0;
        size_t p_number_shared_memory_banks = 0;

        size_t p_total_global_memory = 0_zu; // Bytes.
        size_t p_total_constant_memory = 0_zu; // Bytes.
        size_t p_shared_memory_per_block = 0_zu; // Bytes.
        size_t p_shared_memory_per_multiprocessor = 0_zu; // Bytes.
        size_t p_shared_memory_per_block_opt_in = 0_zu;
        size_t p_memory_pitch = 0_zu; // Bytes.
        size_t p_texture_alignment = 0_zu;
        size_t p_texture_pitch_alignment = 0_zu;
        size_t p_surface_alignment = 0_zu;

    public:
        __host__ __device__ CUDA_Device_Information(void) { }
        __host__ __device__ ~CUDA_Device_Information(void) { }
        
        __host__ __device__ class CUDA_Device_Information& operator=(class CUDA_Device_Information const &ref_source_CUDA_Device_Information_received);

        __host__ __device__ void Copy(class CUDA_Device_Information const &ref_source_CUDA_Device_Information_received);
        __host__ __device__ void Grid_Block_1Dimensions(size_t const elements_received,
                                                                                    size_t const limit_blocks_received,
                                                                                    struct dim3 &ref_dim3_grid_received,
                                                                                    struct dim3 &ref_dim3_block_received,
                                                                                    size_t const registers_per_thread_received = 32u,
                                                                                    size_t const shared_memory_per_block_received = 0u,
                                                                                    size_t const shared_memory_variable_per_block_received = 0u) const;
        __host__ __device__ void Grid_Block_2Dimensions(size_t const rows_received,
                                                                                    size_t const columns_received,
                                                                                    size_t const limit_blocks_received,
                                                                                    struct dim3 &ref_dim3_grid_received,
                                                                                    struct dim3 &ref_dim3_block_received,
                                                                                    size_t const registers_per_thread_received = 32u,
                                                                                    size_t const shared_memory_per_block_received = 0u,
                                                                                    size_t const shared_memory_variable_per_block_received = 0u) const;
        __host__ __device__ void Grid_Block_Transpose_2Dimensions(size_t const rows_received,
                                                                                                    size_t const columns_received,
                                                                                                    size_t const limit_blocks_received,
                                                                                                    struct dim3 &ref_dim3_grid_received,
                                                                                                    struct dim3 &ref_dim3_block_received,
                                                                                                    size_t const registers_per_thread_received = 32u,
                                                                                                    size_t const shared_memory_per_block_received = 0u,
                                                                                                    size_t const shared_memory_variable_per_block_received = 0u) const;
        __host__ __device__ void Grid_Block_cuRAND_1Dimensions(size_t const elements_received,
                                                                                                 size_t limit_blocks_received,
                                                                                                 struct dim3 &ref_dim3_grid_received,
                                                                                                 struct dim3 &ref_dim3_block_received) const;
        __host__ __device__ void Grid_Block_Dynamic_Parallelisme(size_t const elements_received,
                                                                                                 size_t limit_blocks_received,
                                                                                                 struct dim3 &ref_dim3_grid_received,
                                                                                                 struct dim3 &ref_dim3_block_received) const;
        __host__ __device__ void Grid_Block_Reduce_1Dimensions(size_t const elements_received,
                                                                                                size_t const limit_blocks_received,
                                                                                                struct dim3 &ref_dim3_grid_received,
                                                                                                struct dim3 &ref_dim3_block_received,
                                                                                                size_t const registers_per_thread_received = 32u,
                                                                                                size_t const shared_memory_per_block_received = 0u,
                                                                                                size_t const shared_memory_variable_per_block_received = 0u) const;
        __host__ __device__ void Grid_Block_Reduce_Dynamic_Parallelisme(size_t const elements_received,
                                                                                                             size_t const limit_blocks_received,
                                                                                                             struct dim3 &ref_dim3_grid_received,
                                                                                                             struct dim3 &ref_dim3_block_received) const;
        __host__ __device__ void Set__Minimum_Threads_For_Occupancy(size_t const minimum_threads_per_received);

        __host__ bool Initialize(size_t const index_device_received);
        __host__ __device__ bool Initialize(size_t const index_device_received, struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received);
        __host__ __device__ bool Get__Device_Overlap(void) const; // [Deprecated]
        __host__ __device__ bool Get__Kernel_Execute_Timeout_Enabled(void) const;
        __host__ __device__ bool Get__Integrated(void) const;
        __host__ __device__ bool Get__Can_Map_Host_Memory(void) const;
        __host__ __device__ bool Get__Concurrent_Kernels(void) const;
        __host__ __device__ bool Get__ECC_Enabled(void) const;
        __host__ __device__ bool Get__TCC_Driver(void) const;
        __host__ __device__ bool Get__Unified_Addressing(void) const;
        __host__ __device__ bool Get__Stream_Priorities_Supported(void) const;
        __host__ __device__ bool Get__Global_L1_Cache_Supported(void) const;
        __host__ __device__ bool Get__Local_L1_Cache_Supported(void) const;
        __host__ __device__ bool Get__Managed_Memory(void) const;
        __host__ __device__ bool Get__Is_Multi_GPU_Board(void) const;
        __host__ __device__ bool Get__Host_Native_Atomic_Supported(void) const;
        __host__ __device__ bool Get__Pageable_Memory_Access(void) const;
        __host__ __device__ bool Get__Concurrent_Managed_Access(void) const;
        __host__ __device__ bool Get__Compute_Preemption_Supported(void) const;
        __host__ __device__ bool Get__Can_Use__Host_Pointer_For_Registered_Memory(void) const;
        __host__ __device__ bool Get__Cooperative_Launch(void) const;
        __host__ __device__ bool Get__Cooperative_Multi_Device_Launch(void) const;
            
        __host__ __device__ int Get__Major_Compute_Capability(void) const;
        __host__ __device__ int Get__Minor_Compute_Capability(void) const;
        __host__ __device__ size_t Get__Warp_Size(void) const;
        __host__ __device__ size_t Get__Number_Multiprocessor(void) const;
        __host__ __device__ size_t Get__Maximum_Threads_Per_Block(void) const;
        __host__ __device__ size_t Get__Maximum_Threads_Per_Multiprocessor(void) const;
        __host__ __device__ size_t Get__Registers_Per_Block(void) const; // 32-bit
        __host__ __device__ size_t Get__Registers_Per_Multiprocessor(void) const; // 32-bit
        __host__ __device__ size_t Get__Maximum_Threads_Dimension(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Grid_Size(size_t const index_received) const;
        __host__ __device__ size_t Get__Clock_Rate(void) const; // Kilohertz.
        __host__ __device__ int Get__Compute_Mode(void) const;
        __host__ __device__ size_t Get__Maximum_Texture_1D(void) const;
        __host__ __device__ size_t Get__Maximum_Texture_1D_Mipmap(void) const;
        __host__ __device__ size_t Get__Maximum_Texture_1D_Linear(void) const;
        __host__ __device__ size_t Get__Maximum_Texture_2D(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Texture_2D_Mipmap(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Texture_2D_Linear(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Texture_2D_Gather(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Texture_3D(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Texture_3D_Alternate(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Texture_Cubemap(void) const;
        __host__ __device__ size_t Get__Maximum_Texture_1D_Layered(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Texture_2D_Layered(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Texture_Cubemap_Layered(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Surface_1D(void) const;
        __host__ __device__ size_t Get__Maximum_Surface_2D(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Surface_3D(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Surface_1D_Layered(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Surface_2D_Layered(size_t const index_received) const;
        __host__ __device__ size_t Get__Maximum_Surface_Cubemap(void) const;
        __host__ __device__ size_t Get__Maximum_Surface_Cubemap_Layered(size_t const index_received) const;
        __host__ __device__ int Get__PCI_Bus_ID(void) const;
        __host__ __device__ int Get__PCI_Device_ID(void) const;
        __host__ __device__ int Get__PCI_Domain_ID(void) const;
        __host__ __device__ size_t Get__Async_Engine_Count(void) const;
        __host__ __device__ size_t Get__Memory_Clock_Rate(void) const; // Kilohertz.
        __host__ __device__ size_t Get__Memory_Bus_Width(void) const; // Bits.
        __host__ __device__ size_t Get__L2_Cache_Size(void) const; // Bytes.
        __host__ __device__ int Get__Multi_GPU_Board_Group_ID(void) const;
        __host__ __device__ int Get__Single_To_Double_Precision_Performance_Ratio(void) const;
        __host__ __device__ int Get__ID(void) const;

        __host__ __device__ size_t Get__Minimum_Threads_For_Occupancy(bool const use_default_received) const;
        __host__ __device__ size_t Get__Maximum_Threads(void) const;
        __host__ __device__ size_t Get__Number_Concurrent_Kernel_By_Compute_Capability(void) const;
        __host__ __device__ size_t Get__Number_Concurrent_Kernel(void) const;
        __host__ __device__ size_t Get__Number_CUDA_Cores_By_Compute_Capability(void) const;
        __host__ __device__ size_t CUDA__Number_CUDA_Cores(void) const;
        __host__ __device__ size_t Get__Number_CUDA_Cores_Per_Multiprocessor(void) const;
        __host__ __device__ size_t Get__Maximum_Blocks_Per_Multiprocessor_By_Compute_Capability(void) const;
        __host__ __device__ size_t Get__Maximum_Blocks_Per_Multiprocessor(void) const;
        __host__ __device__ size_t Get__Maximum_Warps_Per_Multiprocessor_By_Compute_Capability(void) const;
        __host__ __device__ size_t Get__Maximum_Warps_Per_Multiprocessor(void) const;
        __host__ __device__ size_t Get__Number_Shared_Memory_Banks_By_Compute_Capability(void) const;
        __host__ __device__ size_t Get__Number_Shared_Memory_Banks(void) const;
        __host__ __device__ size_t Get__Limit_Block_Due_To_Warp_Per_Multiprocessor(size_t const number_warps_received) const;

        __host__ __device__ size_t Get__Total_Global_Memory(void) const;
        __host__ __device__ size_t Get__Total_Constant_Memory(void) const;
        __host__ __device__ size_t Get__Shared_Memory_Per_Block(void) const;
        __host__ __device__ size_t Get__Shared_Memory_Per_Multiprocessor(void) const;
        __host__ __device__ size_t Get__Shared_Memory_Per_Block_Opt_In(void) const;
        __host__ __device__ size_t Get__Memory_Pitch(void) const;
        __host__ __device__ size_t Get__Texture_Alignment(void) const;
        __host__ __device__ size_t Get__Texture_Pitch_Alignment(void) const;
        __host__ __device__ size_t Get__Surface_Alignment(void) const;

        __host__ __device__ double OccupencyOfEachMultiprocessor(size_t const thread_count_received,
                                                                                                   size_t const registers_per_thread_received = 32u,
                                                                                                   size_t const shared_memory_per_block_received = 0u) const;

    private:
        int _ID = -1;
};
    
class CUDA_Device_Information_Array
{
    public:
        __host__ __device__ CUDA_Device_Information_Array(void);
        __host__ __device__ ~CUDA_Device_Information_Array(void);

        __host__ bool Push_Back(int const index_device_received);
        __host__ __device__ bool Push_Back(int const index_device_received, struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received);
        __host__ __device__ bool Update(struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received);
        __host__ __device__ bool Select_CUDA_Device(int const index_received);
        __host__ __device__ bool Deallocate(void);

        __host__ __device__ size_t Get__Number_CUDA_Devices(void) const;
        __host__ __device__ int Get__Selected_CUDA_Device(void) const;

        __host__ __device__ class CUDA_Device_Information* Get__CUDA_Device(void) const;
        __host__ __device__ class CUDA_Device_Information* Get__CUDA_Device(size_t const index_received) const;

    private:
        size_t _number_cuda_devices = 0;

        int _selected_cuda_device = -1;

        class CUDA_Device_Information *_ptr_Class_Device_Information_sum = nullptr;
        class CUDA_Device_Information *_ptr_Class_Device_Information_higher = nullptr;
        class CUDA_Device_Information *_ptr_Class_Device_Information_lower = nullptr;
        class CUDA_Device_Information *_ptr_array_Class_Device_Information = nullptr;
};

struct CUDA_Neuron
{
    // Default constructor.
    __device__ CUDA_Neuron(void) { }

    // N: Number of threads.
    // T: Number of times to predict.
    // P: Number of parameters.

    // Dropout variable.
    bool *ptr_mask_dropout_bernoulli = nullptr; // size[1].
    // |END| Dropout variable. |END|

    size_t *ptr_first_connection_index = nullptr; // size[1].
    size_t *ptr_last_connection_index = nullptr; // size[1].
    size_t *ptr_number_connections = nullptr; // size[1].
    size_t *ptr_reduce_summation_size = nullptr; // size[1].
    size_t *ptr_reduce_error_size = nullptr; // size[1].
    size_t *ptr_reduce_norms_size = nullptr; // size[1].
    size_t *ptr_reduce_batch_size = nullptr; // size[1].
    
    T_ *ptr_array_summations = nullptr; // size[N, T].
    T_ *ptr_activation_steepness = nullptr; // size[1].
    T_ *ptr_array_values = nullptr; // size[N, T].
    T_ *ptr_array_errors = nullptr; // size[N, T].
    T_ **ptr_array_reduce_summation = nullptr; // size[N, T], size[ptr_reduce_summation_size].
    T_ **ptr_array_reduce_error = nullptr; // size[N, T], size[ptr_reduce_error_size].
    T_ **ptr_array_reduce_norms = nullptr; // size[1], size[ptr_reduce_norms_size].
    T_ **ptr_array_reduce_mean = nullptr; // size[1], size[ptr_reduce_batch_size].
    T_ **ptr_array_reduce_variance = nullptr; // size[1], size[ptr_reduce_batch_size].
    
    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION *ptr_type_activation_function = nullptr; // size[1].
        
    // Batch renormalization variable.
    T_ *ptr_array_values_hats = nullptr; // size[N, T].
    T_ *ptr_array_values_normalizes = nullptr; // size[N, T].
    T_ *ptr_scale = nullptr; // size[1].
    T_ *ptr_shift = nullptr; // size[1].
    T_ *ptr_array_derivatives_scales = nullptr; // size[N].
    T_ *ptr_array_derivatives_shifts = nullptr; // size[N].
    T_ *ptr_array_means = nullptr; // size[N, T?].
    T_ *ptr_array_variances = nullptr; // size[N, T?].
    T_ *ptr_array_transposed_mean = nullptr; // size[N, T?].
    T_ *ptr_array_transposed_variance = nullptr; // size[N, T?].
    T_ *ptr_array_derivatives_means = nullptr; // size[N].
    T_ *ptr_array_derivatives_variances = nullptr; // size[N].
    T_ *ptr_r_correction = nullptr; // size[1].
    T_ *ptr_d_correction = nullptr; // size[1].
    T_ *ptr_mean_average = nullptr; // size[1].
    T_ *ptr_variance_average = nullptr; // size[1].
    // |END| Batch renormalization variable. |END|

    struct dim3 *ptr_dim3_grid_connections = NULL; // size[1].
    struct dim3 *ptr_dim3_block_connections = NULL; // size[1].
    struct dim3 *ptr_array_dim3_grid_reduce_summation = NULL; // size[ptr_reduce_summation_size].
    struct dim3 *ptr_array_dim3_block_reduce_summation = NULL; // size[ptr_reduce_summation_size].
    struct dim3 *ptr_array_dim3_grid_reduce_error = NULL; // size[ptr_reduce_error_size].
    struct dim3 *ptr_array_dim3_block_reduce_error = NULL; // size[ptr_reduce_error_size].
    struct dim3 *ptr_array_dim3_grid_reduce_threads = NULL; // size[ptr_reduce_batch_size].
    struct dim3 *ptr_array_dim3_block_reduce_threads = NULL; // size[ptr_reduce_batch_size].
    struct dim3 **ptr_array_2D_dim3_grid_reduce_norms = NULL; // size[ptr_reduce_norms_size].
    struct dim3 **ptr_array_2D_dim3_block_reduce_norms = NULL; // size[ptr_reduce_norms_size].

    // cuRAND.
    struct curandStateMtgp32 *ptr_cuRAND_State_MTGP32 = nullptr;
    // |END| cuRAND. |END|
};
    
enum ENUM_TYPE_DIM3 : unsigned int
{
    TYPE_DIM3_1D = 0u,
    TYPE_DIM3_DYNAMIC_PARALLELISM = 1u
};

class CUDA_Storage_Dim3
{
    public:
        __device__ CUDA_Storage_Dim3(void);
        __device__ ~CUDA_Storage_Dim3(void);

        __device__ bool Get__Dim3(size_t const size_need_received,
                                                struct dim3 &ref_dim3_grid_received,
                                                struct dim3 &ref_dim3_block_received,
                                                class CUDA_Device_Information const *const ptr_Class_Device_Information_received,
                                                enum ENUM_TYPE_DIM3 const type_dim3_received);
        __device__ bool Get__Dim3_1D(size_t const size_need_received,
                                                    struct dim3 &ref_dim3_grid_received,
                                                    struct dim3 &ref_dim3_block_received,
                                                    class CUDA_Device_Information const *const ptr_Class_Device_Information_received);
        __device__ bool Get__Dim3_Memcpy(size_t const new_size_received,
                                                            size_t const old_size_received,
                                                            struct dim3 &ref_dim3_grid_zero_received,
                                                            struct dim3 &ref_dim3_block_zero_received,
                                                            struct dim3 &ref_dim3_grid_copy_received,
                                                            struct dim3 &ref_dim3_block_copy_received,
                                                            class CUDA_Device_Information const *const ptr_Class_Device_Information_received,
                                                            bool const memcpy_received = true);
        __device__ bool Get__Dim3_Dynamic_Parallelisme(size_t const size_need_received,
                                                                                struct dim3 &ref_dim3_grid_received,
                                                                                struct dim3 &ref_dim3_block_received,
                                                                                class CUDA_Device_Information const *const ptr_Class_Device_Information_received);

    private:
        int _size_1D = 0;
        int _size_DP = 0;

        size_t *_ptr_array_cache_dim3_size_1D = nullptr;
        size_t *_ptr_array_cache_dim3_size_DP = nullptr;
        
        struct dim3 *_ptr_array_dim3_grids_1D = NULL;
        struct dim3 *_ptr_array_dim3_blocks_1D = NULL;
        
        struct dim3 *_ptr_array_dim3_grids_DP = NULL;
        struct dim3 *_ptr_array_dim3_blocks_DP = NULL;
};
    
struct CUDA_Layer
{
    // Default constructor.
    __device__ CUDA_Layer(void) { }

    // N: Number of threads.
    // T: Number of times to predict.
    // H: Number of neurons in layer.
    // K: Number of blocks in layer.
    // C: Number of cells in layer.

    bool use_Batch_Stride = false;

    enum MyEA::Common::ENUM_TYPE_LAYER type_layer = MyEA::Common::ENUM_TYPE_LAYER::TYPE_LAYER_NONE;
    enum MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION type_activation = MyEA::Common::ENUM_TYPE_LAYER_ACTIVATION::TYPE_ACTIVATION_NONE;

    // FC layer variable.
    size_t *ptr_first_neuron_index = nullptr; // size[H].
    size_t *ptr_last_neuron_index = nullptr; // size[1].
    size_t *ptr_number_neurons = nullptr; // size[1].

    struct CUDA_Neuron *ptr_array_neuron_units = nullptr; // size[H].
    struct CUDA_Neuron *ptr_last_neuron_unit = nullptr; // size[1].
    // |END| FC layer variable. |END|

    // Dropout layer variable.
    T_ dropout_values[2u] = {0};
    // |END| Dropout layer variable. |END|

    // Batch renormalization layer variable.
    bool use_Batch_Renormalization = false;
    // |END| Batch renormalization layer variable. |END|

    struct dim3 *ptr_dim3_grid_neurons = nullptr; // size[1].
    struct dim3 *ptr_dim3_block_neurons = nullptr; // size[1].
    struct dim3 *ptr_dim3_grid_neurons_DP = nullptr; // size[1].
    struct dim3 *ptr_dim3_block_neurons_DP = nullptr; // size[1].
    struct dim3 *ptr_dim3_grid_neurons_cuRAND = nullptr; // size[1].
    struct dim3 *ptr_dim3_block_neurons_cuRAND = nullptr; // size[1].
    struct dim3 *ptr_dim3_grid_batch_neurons = nullptr; // size[1].
    struct dim3 *ptr_dim3_block_batch_neurons = nullptr; // size[1].
    struct dim3 *ptr_dim3_grid_weights = nullptr; // size[1].
    struct dim3 *ptr_dim3_block_weights = nullptr; // size[1].

    class CUDA_Storage_Dim3 *ptr_Class_Storage_Dim3_Batch = nullptr; // size[1].
};

// Get__Number_Examples() / Get__Number_Recurrent_Depth() : 1u
// total_parameters : 2u
// ((ptr_last_layer - 1) - (ptr_array_layers + 1)) + 1u : 3u
#define TOTAL_KERNEL_PARALLEL 9u

enum ENUM_TYPE_CURAND_GENERATOR : unsigned int
{
    TYPE_CURAND_WEIGHTS = 0u,
    TYPE_CURAND_BERNOULLI = 1u
};

class CUDA_Neural_Network
{
    // N: Number of threads.
    // T: Number of times to predict.
    // L: Number of layers.
    // H: Number of neurons.
    // K: Number of blocks.
    // C: Number of cells.
    // P: Number of parameters.
    // W: Number of weights.

    public:
        __host__ __device__ CUDA_Neural_Network(void);
        __host__ __device__ ~CUDA_Neural_Network(void);

        __host__ void Set__Limit_Device_Runtime_Pending_Launch_Count(size_t limit_device_runtime_pending_launch_count_received = 0u);
        __host__ __device__ void Set__Maximum_Allowable_Memory(size_t const available_memory_mbs_received);
        __host__ __device__ bool Update__Thread_Size(size_t number_threads_received);
        __host__ __device__ bool Update__Batch_Size(size_t const batch_size_received);
        __host__ __device__ void Reset__Loss(void);
        __device__ void Merge__Post__Training(void);
        __device__ void device__Clear_Train_Arrays(void);
        __device__ void Compute__Error(size_t const batch_size_received, T_ **const ptr_array_outputs_received);
        __device__ void FF__Compute__Error(size_t const batch_size_received, T_ **const ptr_array_outputs_received);
        __device__ void FF__Compute__Error__Standard(size_t const batch_size_received, T_ **const ptr_array_outputs_received);
        __device__ void FF__Compute__Error__Binary_Cross_Entropy(size_t const batch_size_received, T_ **const ptr_array_outputs_received);
        __device__ void FF__Compute__Error__Bit_Fail(size_t const batch_size_received, T_ **const ptr_array_outputs_received);
        __device__ void Test(size_t const batch_size_received,
                                        T_ **const ptr_array_outputs_received,
                                        size_t const time_step_index_received = 0u);
        __device__ void FF__Test(size_t const batch_size_received, T_ **const ptr_array_outputs_received);
        __device__ void FF__Test__Standard(size_t const batch_size_received, T_ **const ptr_array_outputs_received);
        __device__ void FF__Test__Binary_Cross_Entropy(size_t const batch_size_received, T_ **const ptr_array_outputs_received);
        __device__ void FF__Test__Bit_Fail(size_t const batch_size_received, T_ **const ptr_array_outputs_received);
        __device__ void RNN__Test(size_t const batch_size_received,
                                                T_ **const ptr_array_outputs_received,
                                                size_t const time_step_index_received = 0u);
        __device__ void Initialize_Candidate_Weights(size_t const first_connection_received,
                                                                                            size_t const last_connection_received,
                                                                                            float const scale_factor_received);
        __device__ void Reset__Link_Connections(void);
        __device__ void Add_Candidate_Neuron(struct CUDA_Layer *ptr_layer_received);
        __device__ void Update_Candidate_Slopes(class CUDA_Neural_Network *ptr_CNeural_Network_received = NULL);
        __device__ void Update_Candidate_Weights(size_t const number_examples_received);
        __device__ bool Set__Probability_Retained_Unit(size_t const index_layer_received,
                                                                            T_ const retention_probability_received,
                                                                            bool const scale_weights_received = true);
        __device__ bool Set__Probability_Retained_Unit(struct CUDA_Layer *ptr_layer_received,
                                                                            T_ const retention_probability_received,
                                                                            bool const scale_weights_received = true);
        __device__ bool Set__Batch_Renormalization(size_t const index_layer_received, bool const Set__received = true);
        __device__ bool Set__Batch_Renormalization(struct CUDA_Layer *const ptr_layer_received, bool const Set__received = true);
        __device__ void Scale_Weight__Dropout(T_ const scale_factor_received, struct CUDA_Layer const *const ptr_layer_it_received);
        __device__ void Scale_Weight__FC__Forward__Dropout(T_ const scale_factor_received, struct CUDA_Layer const *const ptr_layer_it_received);
        __host__ __device__ void Set__Loss_Function(enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS const type_loss_function_received);
        __host__ __device__ void Set__Accuracy_Function(enum MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS const type_accuracy_function_received);
        __host__ __device__ void Set__Bit_Fail_Limit(T_ const bit_fail_limit_received);
        __host__ __device__ void Set__Optimizer_Function(enum MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS const optimizer_function_received);
        __device__ void Deallocate__Parameter__Optimizer(void);
        __device__ void Deallocate__Parameter__Gradient_Descent(void);
        __device__ void Deallocate__Parameter__iRPROP_minus(void);
        __device__ void Deallocate__Parameter__iRPROP_plus(void);
        __device__ void Deallocate__Parameter__Adam(void);
        __device__ void Deallocate__Parameter__AMSGrad(void);
        __device__ void Deallocate__Parameter__Regularization(void);
        __device__ void Deallocate_Cost(void);
        __device__ void Deallocate_Reduce_Batch(void);
        __device__ void Deallocate_Reduce_Cost(void);
        __device__ void Deallocate_Batch_Reduce(void);
        __device__ void Deallocate__Normalized_Unit__Batch_Normalization(void);
        __device__ void Deallocate__Neurons_Reduce_Summation(void);
        __device__ void Deallocate__Neurons_Reduce_Error(void);
        __device__ void Deallocate__Neurons_Reduce_Norms(void);
        __device__ void Deallocate__Neuron__Mask_Dropout_Bernoulli(void);
        __device__ void Deallocate__Cell_Unit__Mask_Dropout_Zoneout(void);
        __device__ void Remove_Batch_Normalization(void);
        __device__ void Clear_Optimizer(void);
        __device__ void Reset__Parameter__Normalized_Unit(void);
        __device__ void Reset__Derivative_Parameter__Normalized_Unit(void);
        __device__ void Update_Parameter(size_t const batch_size_received, size_t const training_size_received);
        __device__ void Update_Parameter__Gradient_Descent(size_t const batch_size_received, size_t const training_size_received, size_t const start_index_received, size_t const end_index_received);
        __device__ void Update_Parameter__Gradient_Descent__CUDA(size_t const batch_size_received, size_t const training_size_received, size_t const start_index_received, size_t const end_index_received);
        __device__ void Update_Parameter__Gradient_Descent_Momentum__CUDA(size_t const batch_size_received, size_t const training_size_received, size_t const start_index_received, size_t const end_index_received);
        __device__ void Update_Parameter__Nesterov_Accelerated_Gradient__CUDA(size_t const batch_size_received, size_t const training_size_received, size_t const start_index_received, size_t const end_index_received);
        __device__ void Update_Parameter__iRPROP_plus(size_t const start_index_received, size_t const end_index_received);
        __device__ void Update_Parameter__iRPROP_plus__CUDA(size_t const start_index_received, size_t const end_index_received);
        __device__ void Update_Parameter__iRPROP_plus__CUDA__Dropout(size_t const start_index_received, size_t const end_index_received);
        __device__ void Update_Parameter__Adam(size_t const batch_size_received, size_t const training_size_received, size_t const start_index_received, size_t const end_index_received);
        __device__ void Update_Parameter__AMSGrad(size_t const batch_size_received, size_t const training_size_received, size_t const start_index_received, size_t const end_index_received);
        __device__ void Update_Weight_Regularization__Max_Norm_Constraints(void);
        __device__ void Update_Weight_Regularization__Max_Norm_Constraints__Neurons(struct CUDA_Layer const *const ptr_layer_it_received, struct CUDA_Layer const *const ptr_last_layer_received);
        __device__ void Merge_Derivatives_Parameters(void);
        __host__ __device__ void Launch_Randomize_Weights(T_ const minimum_weight_received, T_ const maximum_weight_received);
        __host__ __device__ void Set__Accurancy_Variance(float const accurancy_variance_received);
        __host__ __device__ void Set__Number_Time_Delays(size_t const time_delays_received);
        __device__ void Set__Accuracy(enum MyEA::Common::ENUM_TYPE_DATASET const type_accuracy_received, float const accurancy_received);
        __device__ void Set__Loss(enum MyEA::Common::ENUM_TYPE_DATASET const type_error_received, float const loss_received);
        __device__ void Indexing_Regularization_Parameters(void);
        __device__ void Indexing_Regularization__Weights__FC__Forward(struct CUDA_Layer const *const ptr_layer_it_received);
        __device__ void Update_Derivative_Weight__Regularization__L1(size_t const batch_size_received);
        __device__ void Update_Derivative_Weight__Regularization__L2(size_t const batch_size_received);
        __device__ void Transpose_Layer_Forward__Batch_Normalization(struct CUDA_Layer *const ptr_layer_it_received);
        __device__ void Transpose_Layer_Backward__Batch_Normalization(struct CUDA_Layer *const ptr_layer_it_received);
        __device__ void Transpose_Weights(void);
        __device__ void Prepare__Global__Grids_Blocks_Dimensions(void);
        __device__ bool Prepare__Layers__Grids_Blocks_Dimensions(void);
        __device__ bool Prepare__Neurons__Grids_Blocks_Dimensions(void);
        __device__ void Prepare__Parameters__Grids_Blocks_Dimensions(void);
        __device__ void Prepare__Threads__Grids_Blocks_Dimensions(size_t const number_threads_received);
        __device__ void Prepare__Threads_Parameters__Grids_Blocks_Dimensions(size_t const number_threads_received);
        __device__ void Prepare__Batch__Grids_Blocks_Dimensions(size_t const batch_size_received);
        __device__ void Prepare__Batch_Layers__Grids_Blocks_Dimensions(size_t const batch_size_received);
        __device__ void Prepare__Batch_Neurons__Grids_Blocks_Dimensions(size_t const batch_size_received);
        __device__ void Copy__Neuron_Unit(struct CUDA_Neuron *const ptr_copy_neuron_received,
                                                    size_t const neuron_first_connection_index_received,
                                                    size_t const neuron_last_connection_index_received,
                                                    T_ const neuron_steepness_received,
                                                    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const neuron_activation_function_received);
        __device__ void Copy__Neurons(size_t const *ptr_array_neuron_first_connection_index_received,
                                                     size_t const *ptr_array_neuron_last_connection_index_received,
                                                     T_ const *ptr_array_neuron_steepness_received,
                                                     enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const *ptr_array_neuron_activation_function_received,
                                                     struct CUDA_Neuron *const ptr_array_copy_first_neuron_received,
                                                     struct CUDA_Neuron *const ptr_array_copy_last_neuron_received);
        __device__ void Copy__FC_to_FC(struct CUDA_Neuron *ptr_copy_neuron_it_received,
                                                                    struct CUDA_Neuron const *const ptr_copy_last_neuron_received,
                                                                    struct CUDA_Neuron *const ptr_copy_first_neuron_received,
                                                                    size_t const *&ptr_array_neuron_units_first_connection_index_received,
                                                                    size_t const *&ptr_array_neuron_units_last_connection_index_received,
                                                                    T_ const *&ptr_array_neuron_units_steepness_received,
                                                                    enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const *&ptr_array_neuron_units_activation_function_received);
        __device__ void Reset__Parameter__Mask_Dropout(bool *ptr_array_neuron_units_mask_dropout_received);
        __device__ void Dropout(void);
        __device__ void Dropout__FC_to(size_t &ref_sync_code_received,
                                                            bool const use_parameters_dropout_received,
                                                            struct CUDA_Layer *const ptr_layer_it_received,
                                                            struct CUDA_Layer const *const ptr_previous_layer_it_received);
        __device__ void Dropout_Bernoulli__FC_to_FC(size_t &ref_sync_code_received,
                                                                        bool const use_parameters_dropout_received,
                                                                        struct CUDA_Layer *const ptr_layer_it_received,
                                                                        struct CUDA_Layer const *const ptr_previous_layer_it_received);
        __device__ void Dropout__FC_to__Batch_Normalization(size_t &ref_sync_code_received,
                                                                                              struct CUDA_Layer *const ptr_layer_it_received,
                                                                                              struct CUDA_Layer const *const ptr_previous_layer_it_received);
        __device__ void Dropout_Bernoulli__FC_to_FC__Batch_Renormalization(size_t &ref_sync_code_received,
                                                                                                         struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                         struct CUDA_Layer const *const ptr_previous_layer_it_received);
        __device__ void Assign_Inputs(bool &ref_synchronized_received,
                                                     size_t const thread_index_received,
                                                     T_ const *ptr_array_inputs_received);
        __device__ void Forward_Pass(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received);
        __device__ void FF__Forward_Pass_Batch(size_t const batch_size_received, T_ const *const *const ptr_matrix_inputs_received);
        __device__ void Assign_Inputs_Batch(bool &ref_synchronized_received,
                                                               size_t const batch_size_received,
                                                               T_ const *const *const ptr_matrix_inputs_received);
        __device__ void Forward_Pass__FC_to(bool &ref_synchronized_received,
                                                                      size_t const batch_size_received,
                                                                      struct CUDA_Layer *const ptr_layer_it_received,
                                                                      struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                      struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                      struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to__Dropout_Bernoulli__Testing(bool &ref_synchronized_received,
                                                                                                  size_t const batch_size_received,
                                                                                                  struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                  struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                  struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                  struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to__Dropout(bool &ref_synchronized_received,
                                                                                                  size_t const batch_size_received,
                                                                                                  struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                  struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                  struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                  struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to__Batch_Renormalization__Loop(bool &ref_synchronized_received,
                                                                                                                        size_t const batch_size_received,
                                                                                                                        struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                        struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to__Batch_Renormalization__Dropout_Bernoulli__Testing(bool &ref_synchronized_received,
                                                                                                                                        size_t const batch_size_received,
                                                                                                                                        struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                        struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to__Batch_Renormalization__Training(bool &ref_synchronized_received,
                                                                                                                        size_t const batch_size_received,
                                                                                                                        struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                        struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to__Batch_Renormalization__Dropout(bool &ref_synchronized_received,
                                                                                                                                      size_t const batch_size_received,
                                                                                                                                      struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                      struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                                      struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                      struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to_FC(bool &ref_synchronized_received,
                                                                                 size_t const batch_size_received,
                                                                                 struct CUDA_Layer *const ptr_layer_it_received,
                                                                                 struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                 struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                 struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to_FC__Softmax(bool &ref_synchronized_received,
                                                                                                size_t const batch_size_received,
                                                                                                struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing(bool &ref_synchronized_received,
                                                                                                            size_t const batch_size_received,
                                                                                                            struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                            struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                            struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                            struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to_FC__Dropout(bool &ref_synchronized_received,
                                                                                                             size_t const thread_index_received,
                                                                                                             struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                             struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                             struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                             struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to_FC__Batch_Renormalization__Loop(bool &ref_synchronized_received,
                                                                                                                                  size_t const batch_size_received,
                                                                                                                                  struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                  struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                                  struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                  struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing(bool &ref_synchronized_received,
                                                                                                                                                size_t const batch_size_received,
                                                                                                                                                struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                                struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                                                struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                                struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to_FC__Batch_Renormalization__Training(bool &ref_synchronized_received,
                                                                                                                                    size_t const batch_size_received,
                                                                                                                                    struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                    struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                                    struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                    struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout(bool &ref_synchronized_received,
                                                                                                                                                    size_t const batch_size_received,
                                                                                                                                                    struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                                    struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                                                    struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                                    struct dim3 const *const ptr_dim3_batch_size_block_received);
        //__device__ void Compute__Error_Tanh_FF(size_t const thread_index_received, T_ const *const ptr_array_desireds_outputs_received);
        __device__ void Backward_Pass(size_t const batch_size_received);
        __device__ void FF__Backward_Pass_Batch(size_t const batch_size_received);
        __device__ void Backward_Pass__FC_to(bool &ref_synchronized_received,
                                                                        size_t const batch_size_received,
                                                                        struct CUDA_Layer *const ptr_layer_it_received,
                                                                        struct CUDA_Layer *const ptr_next_layer_received,
                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Backward_Pass__FC_to__Dropout(bool &ref_synchronized_received,
                                                                                        size_t const batch_size_received,
                                                                                        struct CUDA_Layer *const ptr_layer_it_received,
                                                                                        struct CUDA_Layer *const ptr_next_layer_received,
                                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Backward_Pass__FC_to__Batch_Renormalization(bool &ref_synchronized_received,
                                                                                                            size_t const batch_size_received,
                                                                                                            struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                            struct CUDA_Layer *const ptr_previous_layer_it_received,
                                                                                                            struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                            struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Backward_Pass__FC_to__Batch_Renormalization__Dropout(bool &ref_synchronized_received,
                                                                                                                            size_t const batch_size_received,
                                                                                                                            struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                            struct CUDA_Layer *const ptr_previous_layer_it_received,
                                                                                                                            struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                            struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Backward_Pass__FC_to_FC(bool &ref_synchronized_received,
                                                                                    size_t const batch_size_received,
                                                                                    struct CUDA_Layer *const ptr_layer_it_received,
                                                                                    struct CUDA_Layer *const ptr_next_layer_received,
                                                                                    struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                    struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Backward_Pass__FC_to_FC__Dropout(bool &ref_synchronized_received,
                                                                                                    size_t const batch_size_received,
                                                                                                    struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                    struct CUDA_Layer *const ptr_next_layer_received,
                                                                                                    struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                    struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Backward_Pass__FC_to_FC__Batch_Renormalization(bool &ref_synchronized_received,
                                                                                                                        size_t const batch_size_received,
                                                                                                                        struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                        struct CUDA_Layer *const ptr_previous_layer_it_received,
                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Backward_Pass__FC_to_FC__Batch_Renormalization__Dropout(bool &ref_synchronized_received,
                                                                                                                                        size_t const batch_size_received,
                                                                                                                                        struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                                        struct CUDA_Layer *const ptr_previous_layer_it_received,
                                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Update_Derivative_Weight(size_t const batch_size_received, size_t const time_step_index_received = 0u);
        __device__ void FF__Update_Derivative_Weight(size_t const batch_size_received);
        __device__ void Update_Derivative_Weight__FC_to(bool &ref_synchronized_received,
                                                                                        size_t const batch_size_received,
                                                                                        struct CUDA_Layer *const ptr_layer_it_received,
                                                                                        struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Update_Derivative_Weight__FC_to__Dropout(bool &ref_synchronized_received,
                                                                                                      size_t const batch_size_received,
                                                                                                      struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                      struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                      struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                      struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Update_Derivative_Weight__FC_to_FC(bool &ref_synchronized_received,
                                                                                                  size_t const batch_size_received,
                                                                                                  struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                  struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                  struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                  struct dim3 const *const ptr_dim3_batch_size_block_received);
        __device__ void Update_Derivative_Weight__FC_to_FC__Dropout(bool &ref_synchronized_received,
                                                                                                                 size_t const batch_size_received,
                                                                                                                 struct CUDA_Layer *const ptr_layer_it_received,
                                                                                                                 struct CUDA_Layer const *const ptr_previous_layer_it_received,
                                                                                                                 struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                 struct dim3 const *const ptr_dim3_batch_size_block_received);

        __device__ bool Multi_Class_Classification(void) const;
        __host__ bool Initialize_CUDA_Device(void);
        __host__ bool Initialize_cuRAND(size_t const seed_received);
        __device__ bool Initialize_cuRAND_MTGP32(int const size_received,
                                                                          enum ENUM_TYPE_CURAND_GENERATOR const type_curand_generator_received,
                                                                          struct curandStateMtgp32 *const ptr_curandStateMtgp32);
        __host__ __device__ bool Allocate__Structure(size_t const number_layers_received, size_t const maximum_allowable_memory_received);
        __device__ bool Add_CUDA_Device(int const index_device_received, struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received);
        __device__ bool Reallocate_Connections(size_t const total_connections_received);
        __device__ bool Reallocate_Neurons(size_t const total_neuron_units_received, bool const reSet__neuron_position_received);
        __device__ bool Reallocate_Layers(size_t const total_layers_received);
        __device__ bool Reallocate__Parameter__Regularization(size_t const number_parameters_received);
        __device__ bool Reallocate__Parameter__Dropout_Bernoulli(size_t const number_parameters_received);
        __device__ bool Reallocate__Parameter__Optimizer(size_t const number_parameters_received);
        __device__ bool Reallocate__Parameter__Gradient_Descent(size_t const number_parameters_received);
        __device__ bool Reallocate__Parameter__iRPROP_minus(size_t const number_parameters_received);
        __device__ bool Reallocate__Parameter__iRPROP_plus(size_t const number_parameters_received);
        __device__ bool Reallocate__Parameter__Adam(size_t const number_parameters_received);
        __device__ bool Reallocate__Parameter__AMSGrad(size_t const number_parameters_received);
        __device__ bool Allocate_Weights_Transposed(void);
        __device__ bool Allocate__Parameter(void);
        __device__ bool Allocate__Parameter__Optimizer(void);
        __device__ bool Allocate__Parameter__Gradient_Descent(void);
        __device__ bool Allocate__Parameter__iRPROP_minus(void);
        __device__ bool Allocate__Parameter__iRPROP_plus(void);
        __device__ bool Allocate__Parameter__Adam(void);
        __device__ bool Allocate__Parameter__AMSGrad(void);
        __device__ bool Allocate__Parameter__Regularization(void);
        __device__ bool Allocate__Batch_Normalization(void);
        __device__ bool Allocate_Reduce_Threads(void);
        __device__ bool Allocate_Reduce_Threads_Dim(void);
        __device__ bool Allocate_Reduce_Threads_Dim_DP(void);
        __device__ bool Allocate_Reduce_Cost(void);
        __device__ bool Allocate__Neuron_Units(void);
        __device__ bool Allocate__Neurons_Reduce_Norms(void);
        __device__ bool Allocate__Neurons_Reduce_Summation(void);
        __device__ bool Allocate__Neurons_Reduce_Error(void);
        __device__ bool Allocate__Neurons_Reduce_Batch_Normalization(void);
        __device__ bool Allocate__Neuron__Mask_Dropout_Bernoulli(void);
        __device__ bool Allocate__Normalized_Unit__Batch_Renormalization(void);
        __device__ bool Allocate__Neuron__Batch_Renormalization_Transpose(void);
        __device__ bool Reallocate__Thread(size_t const number_threads_received);
        __device__ bool Reallocate__Batch(size_t const batch_size_received);
        __device__ bool Reallocate__Thread__Cost(size_t const batch_size_received);
        __device__ bool Reallocate_Reduce_Threads(size_t const batch_size_received);
        __device__ bool Reallocate_Reduce_Threads_Dim(size_t const batch_size_received);
        __device__ bool Reallocate_Reduce_Threads_Dim_DP(size_t const batch_size_received);
        __device__ bool Reallocate_Reduce_Cost(size_t const total_reduce_batch_size_received);
        __device__ bool Reallocate__Batch__Neuron_Unit(size_t const batch_size_received);
        __device__ bool Reallocate__Batch__Neuron_Reduce_Summation(size_t const batch_size_received);
        __device__ bool Reallocate__Batch__Neuron_Reduce_Error(size_t const batch_size_received);
        __device__ bool Reallocate__Normalized_Unit__Batch_Normalization(size_t const batch_size_received);
        __device__ bool Reallocate__Batch__Neuron_Batch_Normalization_Transpose(size_t const batch_size_received);
        __device__ bool Reallocate__Batch__Neuron_Batch_Normalization_Reduce(size_t const batch_size_received);
        __device__ bool Reallocate__Thread__Parameter(size_t const batch_size_received);
        __device__ bool Reallocate__Parameter(size_t const batch_size_received);
        __host__ bool Copy__Host_To_Device(class Neural_Network const *const ptr_host_Neural_Network_received, size_t const maximum_allowable_memory_received);
        __host__ __device__ void Copy__Optimizer_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        __host__ __device__ void Copy__Warm_Restarts_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        __host__ __device__ void Copy__Gradient_Descent_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        __host__ __device__ void Copy__QuickProp_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        __host__ __device__ void Copy__RPROP_minus_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        __host__ __device__ void Copy__RPROP_plus_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        __host__ __device__ void Copy__SARProp_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        __host__ __device__ void Copy__Adam_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        __host__ __device__ void Copy__NosAdam_Parameters(class Neural_Network const *const ptr_Neural_Network_received);
        __host__ void Copy__Dropout(class Neural_Network const *const ptr_Neural_Network_received);
        __host__ void Copy__Normalization(class Neural_Network const *const ptr_Neural_Network_received);
        __device__ void device__Copy_Dropout(T_ const *ptr_array_probability_retained_unit_received);
        __device__ void device__Copy__Normalization(enum MyEA::Common::ENUM_TYPE_LAYER_NORMALIZATION const *ptr_array_normalization_by_layers_received);
        __host__ __device__ bool Set__Regularization__L1(T_ const regularization__l1_received);
        __host__ __device__ bool Set__Regularization__L2(T_ const regularization__l2_received);
        __host__ __device__ bool Set__Regularization__Weight_Decay(T_ const regularization__weight_decay_received);
        __host__ __device__ bool Set__Regularization__Max_Norm_Constraints(T_ const regularization__max_norm_constraints_received);
        __host__ __device__ bool Set__Normalization_Momentum_Average(T_ const momentum_average_received);
        __host__ __device__ bool Set__Normalization_Epsilon(T_ const Set__Normalization_Epsilon);
        __host__ __device__ bool Set__Batch_Renormalization_r_Correction_Maximum(T_ const r_correction_maximum_received);
        __host__ __device__ bool Set__Batch_Renormalization_d_Correction_Maximum(T_ const d_correction_maximum_received);
        __device__ bool Allouable__Batch_Size(size_t const batch_size_received,
                                                                 size_t const maximum_threads_received,
                                                                 size_t &ref_batch_size_allouable_received,
                                                                 size_t &ref_number_threads_allouable_received);
        __device__ bool Use__Regularization_Parameter(void) const;
        __host__ __device__ bool Deallocate(void);
        bool use_Dropout = false;
        bool use_Warm_Restarts = false;
        bool use_Nesterov = false;
        bool use_normalized_weight_decay = true;
        bool use_adam_bias_correction = true;
        bool use_Batch_Renormalization = false;

        __device__ int Total_Blocks_cuRAND_MTGP32(enum ENUM_TYPE_CURAND_GENERATOR const type_curand_generator_received);
        __device__ size_t Get__Limit_Device_Runtime_Pending_Launch_Count(void);
        size_t *ptr_array_number_loss = nullptr; // Size[N].
        size_t *ptr_array_reduce_number_loss = nullptr; // Size[total reduce batch size].
        size_t *ptr_array_number_bit_fail = nullptr; // Size[N].
        size_t *ptr_array_reduce_bit_fail_values = nullptr; // Size[total reduce batch size].
        size_t limit_device_runtime_pending_launch_count = 0;
        size_t number_active_threads = 1;
        size_t number_threads = 1;
        size_t cache_number_threads = 1;
        size_t batch_size = 1;
        size_t cache_batch_size = 0;
        size_t number_accuracy_trial = 0;
        size_t number_inputs = 0;
        size_t number_outputs = 0;
        size_t number_time_delays = 0;
        size_t number_recurrent_depth = 0;
        size_t total_neuron_units = 0;
        size_t total_neuron_units_allocated = 0;
        size_t total_block_units = 0;
        size_t total_block_units_allocated = 0;
        size_t total_cell_units = 0;
        size_t total_cell_units_allocated = 0;
        size_t total_parameters = 0;
        size_t total_parameters_allocated = 0;
        size_t total_weights = 0;
        size_t total_weights_allocated = 0;
        size_t total_layers = 0;
        size_t total_reduce_batch_size = 0;
        size_t total_reduce_batch_DP_size = 0;
        size_t neurons_total_reduce_summation_size = 0;
        size_t neurons_total_reduce_error_size = 0;
        size_t neurons_total_reduce_batch_size = 0;
        size_t neurons_total_reduce_norms_size = 0;
        size_t *ptr_array_number_neurons_by_layer = nullptr; // size[L].
        size_t *ptr_array_neuron_units_first_forward_connection_index = nullptr; // size[H].
        size_t *ptr_array_neuron_units_last_forward_connection_index = nullptr; // size[H].
        size_t *ptr_array_neuron_units_number_forward_connections = nullptr; // size[H].
        size_t *ptr_array_neuron_units_reduce_summation_size = nullptr; // size[H].
        size_t *ptr_array_neuron_units_reduce_error_size = nullptr; // size[H].
        size_t *ptr_array_neuron_units_reduce_batch_size = nullptr; // size[H].
        size_t *ptr_array_neuron_units_reduce_norms_size = nullptr; // size[H].
        size_t *ptr_array_neuroyed_number_neurons_in_layer = nullptr; // size[H].
        size_t number_cuRAND_State_MTGP32_neuroyed = 0;
        size_t number_cuRAND_State_MTGP32_weighted = 0;

        __host__ __device__ T_ Get__Accuracy(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received) const;
        __host__ __device__ T_ Get__Loss(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received, size_t const number_digits_received = 9u) const;
        __device__ T_ Get__ME(void) const;
        __device__ T_ Get__Loss_L1(void) const;
        __device__ T_ Get__MAE(void) const;
        __device__ T_ Get__Loss_L2(void) const;
        __device__ T_ Get__MSE(void) const;
        __device__ T_ Get__RMSE(void) const;
        __device__ T_ Get__MAPE(void) const;
        __device__ T_ Get__SMAPE(void) const;
        __device__ T_ Get__MASE(void) const;
        __device__ T_ Get__ACE(void) const;
        __device__ T_ Get__BITFAIL(void) const;
        T_ *ptr_array_loss_values = nullptr; // Size[N].
        T_ *ptr_array_accuracy_values[5u] = {nullptr}; // Size[N].
        T_ *ptr_array_reduce_loss_values = nullptr; // Size[total reduce batch size].
        T_ *ptr_array_reduce_accuracy_values[5u] = {nullptr}; // Size[total reduce batch size].
        T_ loss_training = 1_T;
        T_ loss_validating = 1_T;
        T_ loss_testing = 1_T;
        T_ loss_rprop = 1_T;
        T_ previous_loss_rprop = 1_T;
        T_ accuracy_variance = 0.49_T;
        T_ accuracy_training = 0_T;
        T_ accuracy_validating = 0_T;
        T_ accuracy_testing = 0_T;

        enum MyEA::Common::ENUM_TYPE_NETWORKS type_network = MyEA::Common::ENUM_TYPE_NETWORKS::TYPE_NETWORK_NONE;
        enum MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS type_optimizer_function = MyEA::Common::ENUM_TYPE_OPTIMIZER_FUNCTIONS::TYPE_OPTIMIZER_NONE;
        enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS type_loss_function = MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS::TYPE_LOSS_FUNCTION_NONE;
        enum MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS type_accuracy_function = MyEA::Common::ENUM_TYPE_ACCURACY_FUNCTIONS::TYPE_ACCURACY_FUNCTION_DISTANCE;
        enum MyEA::Common::ENUM_TYPE_STATE_PROPAGATION type_state_propagation = MyEA::Common::ENUM_TYPE_STATE_PROPAGATION::TYPE_STATE_PROPAGATION_INFERENCE; // Dropout variable
        enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION *ptr_array_neuron_units_type_activation_function = nullptr;

        void **ptr_array_ptr_connections;

        struct CUDA_Layer *ptr_array_layers = nullptr; // size[L].
        struct CUDA_Layer *ptr_last_layer = nullptr; // size[1].
            
        /* Grid | Block:
                [0]: Total threads
                [1]: Total parameters
                [2]: Total weights
                [3]: Total neurons
                [4]: (threads - 1) * total parameters
                [5]: Batch * total neurons
                [6]: Max norm constraints
                [7]: Total threads DP
                [8]: Total weights cuRAND MTGP32 */
        struct dim3 *ptr_array_dim3_grid = NULL; // Size[TOTAL_KERNEL_PARALLEL].
        struct dim3 *ptr_array_dim3_block = NULL; // Size[TOTAL_KERNEL_PARALLEL].
        struct dim3 *ptr_array_dim3_grid_reduce_threads = NULL; // Size[total reduce batch size].
        struct dim3 *ptr_array_dim3_block_reduce_threads = NULL; // Size[total reduce batch size].
        struct dim3 *ptr_array_dim3_grid_reduce_threads_DP = NULL; // Size[total reduce batch size].
        struct dim3 *ptr_array_dim3_block_reduce_threads_DP = NULL; // Size[total reduce batch size].
        // Grid | Block: Each layer have a dimensions of X neurons to it.
        struct dim3 *ptr_array_layers_dim3_grid_neurons = NULL; // Size[L].
        struct dim3 *ptr_array_layers_dim3_block_neurons = NULL; // Size[L].
        struct dim3 *ptr_array_layers_dim3_grid_neurons_DP = NULL; // Size[L].
        struct dim3 *ptr_array_layers_dim3_block_neurons_DP = NULL; // Size[L].
        struct dim3 *ptr_array_layers_dim3_grid_neurons_cuRAND = NULL; // Size[L].
        struct dim3 *ptr_array_layers_dim3_block_neurons_cuRAND = NULL; // Size[L].
        // Grid | Block: Each layer have a dimensions of X neurons times batch size to it.
        struct dim3 *ptr_array_layers_dim3_grid_batch_neurons = NULL; // Size[L].
        struct dim3 *ptr_array_layers_dim3_block_batch_neurons = NULL; // Size[L].
        // Grid | Block: Each layer have a dimensions of X weights to it.
        struct dim3 *ptr_array_layers_dim3_grid_weights = NULL; // Size[L].
        struct dim3 *ptr_array_layers_dim3_block_weights = NULL; // Size[L].
        // Grid | Block: Each neuron have a dimensions of X connections to it.
        struct dim3 *ptr_array_neuron_units_dim3_grid_connections = NULL; // Size[H].
        struct dim3 *ptr_array_neuron_units_dim3_block_connections = NULL; // Size[H].
        struct dim3 *ptr_array_neuron_units_dim3_grid_reduce_summation = NULL; // Size[neurons total reduce summation size].
        struct dim3 *ptr_array_neuron_units_dim3_block_reduce_summation = NULL; // Size[neurons total reduce summation size].
        struct dim3 *ptr_array_neuron_units_dim3_grid_reduce_error = NULL; // Size[neurons total reduce error size].
        struct dim3 *ptr_array_neuron_units_dim3_block_reduce_error = NULL; // Size[neurons total reduce error size].
        struct dim3 *ptr_array_neuron_units_dim3_grid_reduce_batch = NULL; // Size[neurons total reduce batch size].
        struct dim3 *ptr_array_neuron_units_dim3_block_reduce_batch = NULL; // Size[neurons total reduce batch size].
        struct dim3 **ptr_array_2D_neurons_dim3_grid_reduce_norms = NULL; // Size[H].
        struct dim3 **ptr_array_2D_neurons_dim3_block_reduce_norms = NULL; // Size[H].
        
        class CUDA_Storage_Dim3 *ptr_Class_Storage_Dim3_Memcpy = nullptr; // size[1].
        class CUDA_Storage_Dim3 *ptr_array_layers_Class_Storage_Dim3_Batch = nullptr; // size[L].

        __device__ T_ const *Get__Outputs(size_t const thread_index_received) const;
        __device__ T_ const *Get__Outputs(size_t const thread_index_received, size_t const time_step_index_received) const;
        __device__ T_ Warm_Restarts_Decay(void);
        __device__ T_ Normalized_Weight_Decay(size_t const batch_size_received, size_t const training_size_received);
        __host__ __device__ T_ Get__Regularization__Max_Norm_Constraints(void) const;
        __host__ __device__ T_ Get__Regularization__L1(void) const;
        __host__ __device__ T_ Get__Regularization__L2(void) const;
        T_ *ptr_array_neuron_units_summations = nullptr; // size[N, T, H].
        T_ *ptr_array_neuron_units_activation_steepness = nullptr; // size[H].
        T_ *ptr_array_neuron_units_values = nullptr; // size[N, T, H].
        T_ *ptr_array_normalized_batch_units_values_hats = nullptr; // size[N, T, H]. Batch renormalization variable.
        T_ *ptr_array_normalized_batch_units_values_normalizes = nullptr; // size[N, T, H]. Batch renormalization variable.
        T_ *ptr_array_normalized_batch_units_scales = nullptr; // size[H]. Batch renormalization variable.
        T_ *ptr_array_normalized_batch_units_shifts = nullptr; // size[H]. Batch renormalization variable.
        T_ *ptr_array_normalized_batch_units_means = nullptr; // size[N, ?, H]. Batch renormalization variable.
        T_ *ptr_array_normalized_batch_units_variances = nullptr; // size[N, ?, H]. Batch renormalization variable.
        T_ *ptr_array_neuron_units_transposed_mean = nullptr; // size[N, ?, H]. Batch renormalization variable.
        T_ *ptr_array_neuron_units_transposed_variance = nullptr; // size[N, ?, H]. Batch renormalization variable.
        T_ *ptr_array_normalized_batch_units_derivatives_means = nullptr; // size[N, ?, H]. Batch renormalization variable.
        T_ *ptr_array_normalized_batch_units_derivatives_variances = nullptr; // size[N, ?, H]. Batch renormalization variable.
        T_ *ptr_array_normalized_batch_units_r_corrections = nullptr; // size[H]. Batch renormalization variable.
        T_ *ptr_array_normalized_batch_units_d_corrections = nullptr; // size[H]. Batch renormalization variable.
        T_ *ptr_array_normalized_batch_units_means_averages = nullptr; // size[H]. Batch renormalization variable.
        T_ *ptr_array_normalized_batch_units_variances_averages = nullptr; // size[H]. Batch renormalization variable.
        T_ *ptr_array_neuron_units_errors = nullptr; // size[N, T, H].
        T_ **ptr_array_2D_neurons_reduce_summation = nullptr; // Size[H], Size[N, T, neurons total reduce summation size].
        T_ **ptr_array_2D_neurons_reduce_error = nullptr; // Size[H], Size[N, T, neurons total reduce error size].
        T_ **ptr_array_2D_neurons_reduce_batch_mean = nullptr; // Size[H], Size[neurons total reduce batch size].
        T_ **ptr_array_2D_neurons_reduce_batch_variance = nullptr; // Size[H], Size[neurons total reduce batch size].
        T_ **ptr_array_2D_neurons_reduce_norms = nullptr; // Size[H], Size[neurons total reduce norms size].
        T_ *ptr_array_transposed_weights = nullptr; // Size[W].
        T_ *ptr_array_parameters = nullptr; // Size[P].
        T_ *ptr_array_derivatives_parameters = nullptr; // Size[N, P].
        T_ *ptr_array_mask_regularized_parameters = nullptr; // Size[P].
        T_ *ptr_array_previous_steps = nullptr; // Size[P].
        T_ *ptr_array_previous_delta_parameters = nullptr; // Size[P].
        T_ *ptr_array_previous_derivatives_parameters = nullptr; // Size[P].
        T_ *ptr_array_previous_biased_first_moment = nullptr; // Size[P].
        T_ *ptr_array_previous_biased_second_moment = nullptr; // Size[P].
        T_ *ptr_array_previous_biased_second_moment_hat = nullptr; // Size[P].
        T_ learning_rate = 0.9_T;
        T_ learning_momentum = 0_T;
        T_ bit_fail_limit = 1_T;
        T_ regularization__max_norm_constraints = 0_T;
        T_ regularization__l1 = 0_T;
        T_ regularization__l2 = 0_T;
        T_ regularization__weight_decay = 0_T;
        T_ adam_learning_rate = 0.001_T;
        T_ adam_beta1 = 0.9_T;
        T_ adam_beta2 = 0.999_T; // {0.99, 0.999}
        T_ adam_previous_beta2 = 0.999_T;
        T_ adam_epsilon = 1.0e-8_T;
        T_ adam_gamma = 0.1_T; // {0.05, 0.1}
        T_ optimizer_time_step = 0_T;
        T_ epoch_time_step = 1_T;
        T_ warm_restarts_decay_learning_rate = 1_T;
        T_ warm_restarts_initial_maximum_learning_rate = 1_T;
        T_ warm_restarts_maximum_learning_rate = 1_T;
        T_ warm_restarts_minimum_learning_rate = 1.0e-7_T;
        T_ warm_restarts_initial_T_i = 1_T;
        T_ warm_restarts_T_i = 1_T;
        T_ warm_restarts_multiplier = 2_T;
        T_ normalization_momentum_average = 0.999_T;
        T_ normalization_epsilon = 1.0e-5_T;
        T_ batch_renormalization_r_correction_maximum = 1_T;
        T_ batch_renormalization_d_correction_maximum = 0_T;

        // Dropout variable.
        bool *ptr_array_units_mask_dropout_bernoulli = nullptr; // Size[H].
        bool *ptr_array_cell_units_mask_dropout_zoneout = nullptr;

        T_ *ptr_array_mask_dropout_parameters = nullptr; // Size[P].
        // |END| Dropout variable. |END|

        float quickprop_decay;
        float quickprop_mu;
            
        float rprop_increase_factor;
        float rprop_decrease_factor;
        float rprop_delta_min;
        float rprop_delta_max;
        float rprop_delta_zero;

        float sarprop_weight_decay_shift;
        float sarprop_step_error_threshold_factor;
        float sarprop_step_error_shift;
        float sarprop_temperature;
        size_t sarprop_epoch;

        __device__ void Printf_Parameters(bool const full_description_received);

        __device__ class CUDA_Device_Information_Array* Get__Class_Device_Information_Array(void) const;
        
        __host__ __device__ size_t Get__Maximum_Allowable_Memory(void) const;
        __host__ __device__ size_t Get__Sizeof(size_t number_threads_received = 0u, size_t batch_size_received = 0u) const;
        __host__ __device__ size_t Get__Batch_Sizeof(size_t batch_size_received = 0u) const;
        __host__ __device__ size_t Get__Threads_Sizeof(size_t number_threads_received = 0u) const;
        size_t maximum_allowable_memory_bytes = 0u; // Bytes.

    private:
        __device__ bool Allocate__Neuron(struct CUDA_Neuron *ptr_neuron_received);
            
        struct curandStateMtgp32 *ptr_array_cuRAND_State_MTGP32_weighted = nullptr; // Size[number_cuRAND_State_MTGP32_weighted], MTGP32.
        struct curandStateMtgp32 *ptr_array_cuRAND_State_MTGP32_neuroyed = nullptr; // Size[number_cuRAND_State_MTGP32_neuroyed], MTGP32.

        class CUDA_Device_Information_Array *_ptr_Class_Device_Information_Array = nullptr; // Ptr.
};

__device__ void Activation_Real(T_ &ref_value_received,
                                              T_ const summation_received,
                                              enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const type_activation_function_received);

__device__ bool cuRAND_Bernoulli(float const probability_received, float const curand_uniform_received);

__device__ void Update_Accuracy(T_ const error_received,
                                                  T_ const bit_fail_limit_received,
                                                  float *const ptr_accuracy_value_received);
__device__ void Update_Accuracy__atomic(T_ const error_received,
                                                                T_ const bit_fail_limit_received,
                                                                float *const ptr_accuracy_value_received);

// Standard, update error.
__device__ void Update_Error(T_ const observed_output_received,
                                            T_ const desired_output_received,
                                            T_ const error_received,
                                            float *const ptr_loss_values_received,
                                            enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS const type_loss_function_received);
__device__ void Update_Error__atomic(T_ const observed_output_received,
                                                         T_ const desired_output_received,
                                                         T_ const error_received,
                                                         float *const ptr_loss_values_received,
                                                         enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS const type_loss_function_received);

// Binary cross entropy, update error.
__device__ void Update_Error__Binary_Cross_Entropy(T_ const observed_output_received,
                                                                                T_ const desired_output_received,
                                                                                float *const ptr_loss_values_received);
__device__ void Update_Error__Binary_Cross_Entropy__atomic(T_ const observed_output_received,
                                                                                             T_ const desired_output_received,
                                                                                             float *const ptr_loss_values_received);

// Bit fail, update error.
__device__ void Update_Error__Bit_Fail(T_ const error_received,
                                                          T_ const bit_fail_limit_received,
                                                          size_t *const ptr_bit_fail_values_received);
__device__ void Update_Error__Bit_Fail__atomic(T_ const error_received,
                                                                       T_ const bit_fail_limit_received,
                                                                       size_t *const ptr_bit_fail_values_received);

__device__ T_ Activation_Derived(T_ const activation_steepness_received,
                                                T_ const summation_received,
                                                T_ const value_received,
                                                enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const activation_function_received);

__device__ T_ Activation_Derived(T_ const activation_steepness_received,
                                                T_ const summation_received,
                                                T_ const value_received,
                                                enum MyEA::Common::ENUM_TYPE_ACTIVATION_FUNCTION const activation_function_received,
                                                enum MyEA::Common::ENUM_TYPE_LOSS_FUNCTIONS const type_loss_function_received);