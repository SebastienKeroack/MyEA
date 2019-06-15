#include <Tools/CUDA_Reallocate.cuh>
#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

__host__ __device__ CUDA_Device_Information_Array::CUDA_Device_Information_Array(void) { }

__global__ void kernel__Class_Device_Information_Array__Push_Back(int const index_device_received,
                                                                                                       struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received,
                                                                                                       class CUDA_Device_Information_Array *const ptr_Class_Device_Information_Array_received)
{ ptr_Class_Device_Information_Array_received->Push_Back(index_device_received, ptr_struct_cudaDeviceProp_received); }

__host__ bool CUDA_Device_Information_Array::Push_Back(int const index_device_received)
{
    struct cudaDeviceProp tmp_struct_cudaDeviceProp,
                                            *tmp_ptr_device_struct_cudaDeviceProp(NULL);

    CUDA__Safe_Call(cudaGetDeviceProperties(&tmp_struct_cudaDeviceProp, index_device_received));

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_struct_cudaDeviceProp, sizeof(struct cudaDeviceProp)));

    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_struct_cudaDeviceProp,
                                                            &tmp_struct_cudaDeviceProp,
                                                            sizeof(struct cudaDeviceProp),
                                                            cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__Class_Device_Information_Array__Push_Back <<< 1u, 1u >>> (index_device_received,
                                                                                                             tmp_ptr_device_struct_cudaDeviceProp,
                                                                                                             this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_struct_cudaDeviceProp));

    return(true);
}
    
__host__ __device__ bool CUDA_Device_Information_Array::Push_Back(int const index_device_received, struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received)
{
    if(ptr_struct_cudaDeviceProp_received == nullptr) { return(false); }

#if defined(__CUDA_ARCH__) == false
    kernel__Class_Device_Information_Array__Push_Back <<< 1u, 1u >>> (index_device_received,
                                                                                                             ptr_struct_cudaDeviceProp_received,
                                                                                                             this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#else
    if(CUDA__Required_Compatibility_Device(*ptr_struct_cudaDeviceProp_received))
    {
        for(size_t i(0); i != this->_number_cuda_devices; ++i)
        {
            if(this->_ptr_array_Class_Device_Information[i].Get__ID() == index_device_received)
            { return(true); }
        }

        if(this->_ptr_array_Class_Device_Information == nullptr)
        {
            this->_ptr_Class_Device_Information_sum = new class CUDA_Device_Information;
            this->_ptr_Class_Device_Information_higher = new class CUDA_Device_Information;
            this->_ptr_Class_Device_Information_lower = new class CUDA_Device_Information;
            this->_ptr_array_Class_Device_Information = new class CUDA_Device_Information[1u];
        }
        else
        {
            class CUDA_Device_Information *tmp_ptr_array_Class_Device_Information(Memory::reallocate_objects_cpp<class CUDA_Device_Information>(this->_ptr_array_Class_Device_Information,
                                                                                                                                                                                                                           this->_number_cuda_devices + 1u,
                                                                                                                                                                                                                           this->_number_cuda_devices));

            if(tmp_ptr_array_Class_Device_Information == nullptr)
            {
                PRINT_FORMAT("%s: ERROR: 'tmp_ptr_array_Class_Device_Information' is a nullptr." NEW_LINE, __FUNCTION__);

                return(false);
            }

            this->_ptr_array_Class_Device_Information = tmp_ptr_array_Class_Device_Information;
        }

        if(this->_ptr_array_Class_Device_Information[this->_number_cuda_devices].Initialize(index_device_received, ptr_struct_cudaDeviceProp_received))
        { this->Update(ptr_struct_cudaDeviceProp_received); }

        this->_selected_cuda_device = this->_number_cuda_devices;

        ++this->_number_cuda_devices;
    }
#endif

    return(true);
}

__global__ void kernel__Class_Device_Information_Array__Refresh(struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received, class CUDA_Device_Information_Array *const ptr_Class_Device_Information_Array_received)
{ ptr_Class_Device_Information_Array_received->Update(ptr_struct_cudaDeviceProp_received); }

__host__ __device__ bool CUDA_Device_Information_Array::Update(struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received)
{
    if(ptr_struct_cudaDeviceProp_received == nullptr) { return(false); }

#if defined(__CUDA_ARCH__) == false
    kernel__Class_Device_Information_Array__Refresh <<< 1u, 1u >>> (ptr_struct_cudaDeviceProp_received, this);
        
#if defined(COMPILE_DEBUG)
    CUDA__Check_Error();
#endif
#else
    PRINT_FORMAT("%s: [FUNCTION DEPRECATED] TODO: Fix \"Update\" algorithm." NEW_LINE, __FUNCTION__);

    // Sum += ptr_struct_cudaDeviceProp_received

    // Higher > ptr_struct_cudaDeviceProp_received

    // Lower < ptr_struct_cudaDeviceProp_received
#endif

    return(true);
}

__host__ __device__ bool CUDA_Device_Information_Array::Deallocate(void)
{
    SAFE_DELETE(this->_ptr_Class_Device_Information_sum);
    SAFE_DELETE(this->_ptr_Class_Device_Information_higher);
    SAFE_DELETE(this->_ptr_Class_Device_Information_lower);
    SAFE_DELETE_ARRAY(this->_ptr_array_Class_Device_Information);

    return(true);
}

__host__ __device__ bool CUDA_Device_Information_Array::Select_CUDA_Device(int const index_received)
{
    if(Get__Number_CUDA_Devices() > index_received)
    {
        this->_selected_cuda_device = index_received;

        return(true);
    }
    else
    {
        PRINT_FORMAT("%s: ERROR: Index overflow." NEW_LINE, __FUNCTION__);
            
        return(false);
    }
}

__host__ __device__ size_t CUDA_Device_Information_Array::Get__Number_CUDA_Devices(void) const { return(this->_number_cuda_devices); }

__host__ __device__ int CUDA_Device_Information_Array::Get__Selected_CUDA_Device(void) const { return(this->_selected_cuda_device); }

__host__ __device__ class CUDA_Device_Information* CUDA_Device_Information_Array::Get__CUDA_Device(void) const
{
    if(static_cast<int>(this->Get__Number_CUDA_Devices()) > this->_selected_cuda_device && this->_selected_cuda_device >= 0) { return(&this->_ptr_array_Class_Device_Information[this->_selected_cuda_device]); }
    else { return(nullptr); }
}

__host__ __device__ class CUDA_Device_Information* CUDA_Device_Information_Array::Get__CUDA_Device(size_t const index_received) const
{
    if(this->Get__Number_CUDA_Devices() > index_received) { return(&this->_ptr_array_Class_Device_Information[index_received]); }
    else { return(nullptr); }
}

__host__ __device__ CUDA_Device_Information_Array::~CUDA_Device_Information_Array(void)
{ this->Deallocate(); }
