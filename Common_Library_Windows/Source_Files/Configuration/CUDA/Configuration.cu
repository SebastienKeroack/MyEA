#include <Configuration/CUDA/Configuration.cuh>
    
#include <iostream>

void CUDA__Initialize__Device(struct cudaDeviceProp const &ref_device_prop_received, size_t const memory_allocate_received)
{
    size_t tmp_bytes_total(0),
           tmp_bytes_free(0);

    CUDA__Safe_Call(cudaMemGetInfo(&tmp_bytes_free, &tmp_bytes_total));

    if(tmp_bytes_free < memory_allocate_received)
    { CUDA__Safe_Call(cudaDeviceSetLimit(cudaLimit::cudaLimitMallocHeapSize, tmp_bytes_free)); }
    else
    { CUDA__Safe_Call(cudaDeviceSetLimit(cudaLimit::cudaLimitMallocHeapSize, memory_allocate_received)); }
}

void CUDA__Set__Device(int const index_device_received)
{
    if(index_device_received >= 0) { CUDA__Safe_Call(cudaSetDevice(index_device_received)); }
    else { PRINT_FORMAT("%s: ERROR: Device index can not be less than one." NEW_LINE, __FUNCTION__); }
}

void CUDA__Set__Synchronization_Depth(size_t const depth_received) { CUDA__Safe_Call(cudaDeviceSetLimit(cudaLimit::cudaLimitDevRuntimeSyncDepth, depth_received)); }

void CUDA__Reset(void) { CUDA__Safe_Call(cudaDeviceReset()); }

void CUDA__Print__Device_Property(struct cudaDeviceProp const &ref_device_prop_received, int const index_device_received)
{
    // 1024 bytes = 1 KB, Kilobyte
    // 1024 KB = 1 MB, Megabyte
    // 1024 MB = 1 GB, Gibabyte
    // 1024 GB = 1 TB, Terabyte
    // 1024 TB = 1 PB, PB = Petabye
    // byte  to kilobye: 1024 bytes
    // byte  to megabye: 1024 bytes ^ 2 = 1 048 576 bytes
    // byte  to gigabyte: 1024 bytes ^ 3 = 1 073 741 824 bytes
    // 1000 Hertz = 1 kHz, kilohertz
    // 1000 kHz = 1 MHz, MegaHertz
    // 1000 MHz = 1 GHz, GigaHertz
    // 1000 GHz = 1 THz, TeraHertz
    // Hertz to kHz: 1000 Hertz
    // Hertz to MHz: 1000 ^ 2 = 1 000 000 Hertz
    // Hertz to GHz: 1000 ^ 3 = 1 000 000 000 Hertz
    
    int tmp_current_device(0);

    CUDA__Safe_Call(cudaGetDevice(&tmp_current_device));

    size_t const tmp_number_cuda_cores(CUDA__Number_CUDA_Cores(ref_device_prop_received));
    int tmp_driver_version(0),
        tmp_runtime_version(0);
    
    size_t tmp_bytes_total(0),
              tmp_bytes_free(0);

    if(index_device_received >= 0)
    {
        CUDA__Safe_Call(cudaSetDevice(index_device_received));

        PRINT_FORMAT("%s: Device [%d] name: \"%s\"" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 index_device_received,
                                 ref_device_prop_received.name);
        
        CUDA__Safe_Call(cudaDriverGetVersion(&tmp_driver_version));
        CUDA__Safe_Call(cudaRuntimeGetVersion(&tmp_runtime_version));
            
        PRINT_FORMAT("%s: CUDA Driver version / Runtime version: %d / %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 tmp_driver_version,
                                 tmp_runtime_version);

        PRINT_FORMAT("%s: CUDA Capability Major/Minor version number: %d.%d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.major,
                                 ref_device_prop_received.minor);
        
        PRINT_FORMAT("%s: Total amount of global memory: %.2f GB | %.2f MB | %zu bytes" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(ref_device_prop_received.totalGlobalMem) / 1073741824.0,
                                 static_cast<double>(ref_device_prop_received.totalGlobalMem) / 1048576.0,
                                 ref_device_prop_received.totalGlobalMem);

        CUDA__Safe_Call(cudaMemGetInfo(&tmp_bytes_free, &tmp_bytes_total));

        PRINT_FORMAT("%s: Total amount of free memory: %.2f GB | %.2f MB | %zu bytes" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(tmp_bytes_free) / 1073741824.0,
                                 static_cast<double>(tmp_bytes_free) / 1048576.0,
                                 tmp_bytes_free);
        PRINT_FORMAT("%s: Total amount of global available memory: %.2f GB | %.2f MB | %zu bytes" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(tmp_bytes_total) / 1073741824.0,
                                 static_cast<double>(tmp_bytes_total) / 1048576.0,
                                 tmp_bytes_total);
        
        PRINT_FORMAT("%s: (%d) Multiprocessors, (%zu) CUDA Cores/MP: %zu CUDA Cores" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.multiProcessorCount,
                                 tmp_number_cuda_cores,
                                 tmp_number_cuda_cores / static_cast<size_t>(ref_device_prop_received.multiProcessorCount));
        
        PRINT_FORMAT("%s: Clock frequency: %.4f GHz | %.4f MHz | %d kHz" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(ref_device_prop_received.clockRate) / 1000000.0,
                                 static_cast<double>(ref_device_prop_received.clockRate) / 1000.0,
                                 ref_device_prop_received.clockRate);
        
        PRINT_FORMAT("%s: Peak memory clock frequency: %.4f GHz | %.4f MHz | %d kHz" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(ref_device_prop_received.memoryClockRate) / 1000000.0,
                                 static_cast<double>(ref_device_prop_received.memoryClockRate) / 1000.0,
                                 ref_device_prop_received.memoryClockRate);
        
        PRINT_FORMAT("%s: Global memory bus width: %d-bit" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.memoryBusWidth);
        
        PRINT_FORMAT("%s: Size of L2 cache: %.2f MB | %.2f KB | %d bytes" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(ref_device_prop_received.l2CacheSize) / 1048576.0,
                                 static_cast<double>(ref_device_prop_received.l2CacheSize) / 1024.0,
                                 ref_device_prop_received.l2CacheSize);
        
        PRINT_FORMAT("%s: Maximum texture size (x, y, z): 1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxTexture1D,
                                 ref_device_prop_received.maxTexture2D[0u],
                                 ref_device_prop_received.maxTexture2D[1u],
                                 ref_device_prop_received.maxTexture3D[0u],
                                 ref_device_prop_received.maxTexture3D[1u],
                                 ref_device_prop_received.maxTexture3D[2u]);
        
        PRINT_FORMAT("%s: Maximum mipmapped texture size (x, y): 1D=(%d), 2D=(%d, %d)" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxTexture1DMipmap,
                                 ref_device_prop_received.maxTexture2DMipmap[0u],
                                 ref_device_prop_received.maxTexture2DMipmap[1u]);
        
        PRINT_FORMAT("%s: Maximum textures bound to linear memory (x, y, pitch): 1D=(%d), 2D=(%d, %d, %d)" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxTexture1DLinear,
                                 ref_device_prop_received.maxTexture2DLinear[0u],
                                 ref_device_prop_received.maxTexture2DLinear[1u],
                                 ref_device_prop_received.maxTexture2DLinear[2u]);
        
        PRINT_FORMAT("%s: Maximum 2D texture dimensions if texture gather operations have to be performed (x, y): (%d, %d)" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxTexture2DGather[0u],
                                 ref_device_prop_received.maxTexture2DGather[1u]);
        
        PRINT_FORMAT("%s: Maximum alternate 3D texture dimensions (x, y, z): (%d, %d, %d)" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxTexture3DAlt[0u],
                                 ref_device_prop_received.maxTexture3DAlt[1u],
                                 ref_device_prop_received.maxTexture3DAlt[2u]);
        
        PRINT_FORMAT("%s: Maximum Cubemap texture dimensions: %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxTextureCubemap);
        
        PRINT_FORMAT("%s: Maximum Cubemap texture dimensions: 1D=(%d), %d layers" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxTextureCubemapLayered[0u],
                                 ref_device_prop_received.maxTextureCubemapLayered[1u]);
        
        PRINT_FORMAT("%s: Maximum Cubemap layered texture dimensions, (num) layers: 1D=(%d), %d layers" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxTexture1DLayered[0u],
                                 ref_device_prop_received.maxTexture1DLayered[1u]);
        
        PRINT_FORMAT("%s: Maximum 2D layered texture dimensions, (num) layers: 2D=(%d, %d), %d layers" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxTexture2DLayered[0u],
                                 ref_device_prop_received.maxTexture2DLayered[1u],
                                 ref_device_prop_received.maxTexture2DLayered[2u]);
        
        PRINT_FORMAT("%s: Maximum surface size (x, y, z): 1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxSurface1D,
                                 ref_device_prop_received.maxSurface2D[0u],
                                 ref_device_prop_received.maxSurface2D[1u],
                                 ref_device_prop_received.maxSurface3D[0u],
                                 ref_device_prop_received.maxSurface3D[1u],
                                 ref_device_prop_received.maxSurface3D[2u]);
        
        PRINT_FORMAT("%s: Maximum 1D layered surface dimensions, (num) layers: 1D=(%d), %d layers" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxSurface1DLayered[0u],
                                 ref_device_prop_received.maxSurface1DLayered[1u]);
        
        PRINT_FORMAT("%s: Maximum 2D layered surface dimensions, (num) layers: 2D=(%d, %d), %d layers" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxSurface2DLayered[0u],
                                 ref_device_prop_received.maxSurface2DLayered[1u],
                                 ref_device_prop_received.maxSurface2DLayered[2u]);
        
        PRINT_FORMAT("%s: Maximum Cubemap surface dimensions: %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxSurfaceCubemap);
        
        PRINT_FORMAT("%s: Maximum Cubemap surface dimensions: 1D=(%d), %d layers" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxSurfaceCubemapLayered[0u],
                                 ref_device_prop_received.maxSurfaceCubemapLayered[1u]);
        
        PRINT_FORMAT("%s: Constant memory available on device: %.2f MB | %.2f KB | %zu bytes" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(ref_device_prop_received.totalConstMem) / 1048576.0,
                                 static_cast<double>(ref_device_prop_received.totalConstMem) / 1024.0,
                                 ref_device_prop_received.totalConstMem);
        
        PRINT_FORMAT("%s: Shared memory available per block: %.2f MB | %.2f KB | %zu bytes" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(ref_device_prop_received.sharedMemPerBlock) / 1048576.0,
                                 static_cast<double>(ref_device_prop_received.sharedMemPerBlock) / 1024.0,
                                 ref_device_prop_received.sharedMemPerBlock);
        
        PRINT_FORMAT("%s: 32-bit registers available per block: %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.regsPerBlock);
        
        PRINT_FORMAT("%s: 32-bit registers available per multiprocessor: %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.regsPerMultiprocessor);
        
        PRINT_FORMAT("%s: Warp size: %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.warpSize);
        
        PRINT_FORMAT("%s: Maximum number of threads per multiprocessor: %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxThreadsPerMultiProcessor);
        
        PRINT_FORMAT("%s: Maximum number of threads per block: %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxThreadsPerBlock);
        
        PRINT_FORMAT("%s: Maximum size of each dimension of a block (x, y, z): (%d, %d, %d)" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxThreadsDim[0u],
                                 ref_device_prop_received.maxThreadsDim[1u],
                                 ref_device_prop_received.maxThreadsDim[2u]);
        
        PRINT_FORMAT("%s: Maximum size of each dimension of a grid (x, y, z): (%d, %d, %d)" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.maxGridSize[0u],
                                 ref_device_prop_received.maxGridSize[1u],
                                 ref_device_prop_received.maxGridSize[2u]);
        
        PRINT_FORMAT("%s: Maximum pitch in bytes allowed by memory copies: %.2f MB | %.2f KB | %zu bytes" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(ref_device_prop_received.memPitch) / 1048576.0,
                                 static_cast<double>(ref_device_prop_received.memPitch) / 1024.0,
                                 ref_device_prop_received.memPitch);
        
        PRINT_FORMAT("%s: Alignment requirement for textures: %.2f MB | %.2f KB | %zu bytes" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(ref_device_prop_received.textureAlignment) / 1048576.0,
                                 static_cast<double>(ref_device_prop_received.textureAlignment) / 1024.0,
                                 ref_device_prop_received.textureAlignment);
        
        PRINT_FORMAT("%s: Pitch alignment requirement for texture references bound to pitched memory: %.2f MB | %.2f KB | %zu bytes" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(ref_device_prop_received.texturePitchAlignment) / 1048576.0,
                                 static_cast<double>(ref_device_prop_received.texturePitchAlignment) / 1024.0,
                                 ref_device_prop_received.texturePitchAlignment);
        
        PRINT_FORMAT("%s: Shared memory available per multiprocessor: %.2f MB | %.2f KB | %zu bytes" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(ref_device_prop_received.sharedMemPerMultiprocessor) / 1048576.0,
                                 static_cast<double>(ref_device_prop_received.sharedMemPerMultiprocessor) / 1024.0,
                                 ref_device_prop_received.sharedMemPerMultiprocessor);
        
        PRINT_FORMAT("%s: Per device maximum shared memory per block usable by special opt in: %.2f MB | %.2f KB | %zu bytes" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(ref_device_prop_received.sharedMemPerBlockOptin) / 1048576.0,
                                 static_cast<double>(ref_device_prop_received.sharedMemPerBlockOptin) / 1024.0,
                                 ref_device_prop_received.sharedMemPerBlockOptin);
        
        PRINT_FORMAT("%s: Number of asynchronous engines: %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.asyncEngineCount);
        
        PRINT_FORMAT("%s: Ratio of single precision performance (in floating-point operations per second) to double precision performance: %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.singleToDoublePrecisionPerfRatio);
        
        PRINT_FORMAT("%s: Alignment requirement for surfaces: %s (%zu)" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.surfaceAlignment != 0_zu ? "Yes" : "No",
                                 ref_device_prop_received.surfaceAlignment);
        
        PRINT_FORMAT("%s: Device can coherently access managed memory concurrently with the CPU: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.concurrentManagedAccess != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: Device can possibly execute multiple kernels concurrently: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.concurrentKernels != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: Device can access host registered memory at the same virtual address as the CPU: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.canUseHostPointerForRegisteredMem != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: Link between the device and the host supports native atomic operations: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.hostNativeAtomicSupported != 0 ? "Yes" : "No");

        PRINT_FORMAT("%s: Device is on a multi-GPU board: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.isMultiGpuBoard != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: Unique identifier for a group of devices on the same multi-GPU board: %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.multiGpuBoardGroupID);
        
        PRINT_FORMAT("%s: Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.canMapHostMemory != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: Device supports stream priorities: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.streamPrioritiesSupported != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: Device supports caching globals in L1: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.globalL1CacheSupported != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: Device supports caching locals in L1: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.localL1CacheSupported != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: Device supports Compute Preemption: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.computePreemptionSupported != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: Device supports allocating managed memory on this system: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.managedMemory != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.cooperativeLaunch != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: Device can participate in cooperative kernels launched via ::cudaLaunchCooperativeKernelMultiDevice: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.cooperativeMultiDeviceLaunch != 0 ? "Yes" : "No");
        
        if(ref_device_prop_received.deviceOverlap != 0) // [Deprecated]
        {
            PRINT_FORMAT("%s: Concurrent copy and execution [Deprecated]: Yes (maximum of %d)" NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     CUDA__Maximum_Concurrent_Kernel(ref_device_prop_received));
        }
        else { PRINT_FORMAT("%s: Concurrent copy and execution [Deprecated]: No" NEW_LINE, MyEA::String::Get__Time().c_str()); }
        
        PRINT_FORMAT("%s: Specified whether there is a run time limit on kernels: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.kernelExecTimeoutEnabled != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: Device is integrated as opposed to discrete: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.integrated != 0 ? "Yes" : "No");

        PRINT_FORMAT("%s: Device supports coherently accessing pageable memory without calling cudaHostRegister on it: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.pageableMemoryAccess != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: Device has ECC support: %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.ECCEnabled != 0 ? "Enabled" : "Disabled");
        
        PRINT_FORMAT("%s: CUDA Device Driver Mode (TCC or WDDM): %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.tccDriver != 0 ? "TCC" : "WDDM (Windows Display Driver Model)");
        
        PRINT_FORMAT("%s: Device shares a unified address space with the host (UVA): %s" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.unifiedAddressing != 0 ? "Yes" : "No");
        
        PRINT_FORMAT("%s: PCI bus ID of the device: %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.pciBusID);
        
        PRINT_FORMAT("%s: PCI device ID of the device: %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.pciDeviceID);
        
        PRINT_FORMAT("%s: PCI domain ID of the device: %d" NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.pciDomainID);
        
        PRINT_FORMAT("%s: Compute mode: [%d] ",
                                 MyEA::String::Get__Time().c_str(),
                                 ref_device_prop_received.computeMode);

        switch(ref_device_prop_received.computeMode)
        {
            case 0: PRINT_FORMAT("Default compute mode (Multiple threads can use cudaSetDevice() with this device)." NEW_LINE); break;
            case 1: PRINT_FORMAT("Compute-exclusive-thread mode (Only one thread in one process will be able to use cudaSetDevice() with this device)." NEW_LINE); break;
            case 2: PRINT_FORMAT("Compute-prohibited mode (No threads can use cudaSetDevice() with this device)." NEW_LINE); break;
            case 3: PRINT_FORMAT("Compute-exclusive-process mode (Many threads in one process will be able to use cudaSetDevice() with this device)." NEW_LINE); break;
        }
        
        // https://devblogs.nvidia.com/parallelforall/how-implement-performance-metrics-cuda-cc/
        // (MHz * ((BusWidth to bytes) * double data rate) / Convert to GB/s)
        PRINT_FORMAT("%s: Theoretical bandwidth (GB/s): %f." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 ((static_cast<double>(ref_device_prop_received.memoryClockRate) * 1000.0) * ((static_cast<double>(ref_device_prop_received.memoryBusWidth) / 8.0) * 2.0 )) / 1e9);
        
        // CudaCores * SMs * ClockRate_GHz * FMA_Instruction(2)
        PRINT_FORMAT("%s: Single precision: Theoretical throughput (GFLOP/s): %f." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(CUDA__Number_CUDA_Cores(ref_device_prop_received)) * (static_cast<double>(ref_device_prop_received.clockRate) / 1e6));
        PRINT_FORMAT("%s: Single precision: Theoretical throughput FMA (GFLOP/s): %f." NEW_LINE,
                                 MyEA::String::Get__Time().c_str(),
                                 static_cast<double>(CUDA__Number_CUDA_Cores(ref_device_prop_received)) * (static_cast<double>(ref_device_prop_received.clockRate) / 1e6) * 2.0);
        // FMA = 1 instruction for 2 flops, Fused multiply-add (X * Y + Z)
        
        if(index_device_received != tmp_current_device) { CUDA__Safe_Call(cudaSetDevice(tmp_current_device)); }
    }
}

int CUDA__Device_Count(void)
{
    int tmp_device_count(0);
        
    CUDA__Safe_Call(cudaGetDeviceCount(&tmp_device_count));
        
    return(tmp_device_count);
}

int CUDA__Maximum_Concurrent_Kernel(struct cudaDeviceProp const &ref_device_prop_received)
{
    unsigned int tmp_max_concurrent_kernel(0u);

    switch(ref_device_prop_received.major)
    {
        case 2: // Fermi.
            tmp_max_concurrent_kernel = 16;
                break;
        case 3: // Kepler.
            if(ref_device_prop_received.minor == 2) { tmp_max_concurrent_kernel = 4; }
            if(ref_device_prop_received.minor == 5 ||
                ref_device_prop_received.minor == 7)
            { tmp_max_concurrent_kernel = 32; }
            else { PRINT_FORMAT("%s: Unknown minor device version." NEW_LINE, MyEA::String::Get__Time().c_str()); }
                break;
        case 5: // Maxwell.
            if(ref_device_prop_received.minor == 0 ||
                ref_device_prop_received.minor == 2)
            { tmp_max_concurrent_kernel = 32; }
            else if(ref_device_prop_received.minor == 3) { tmp_max_concurrent_kernel = 16; }
            else { PRINT_FORMAT("%s: Unknown minor device version." NEW_LINE, MyEA::String::Get__Time().c_str()); }
                break;
        case 6: // Pascal.
            if(ref_device_prop_received.minor == 0) { tmp_max_concurrent_kernel = 128; }
            else if(ref_device_prop_received.minor == 1) { tmp_max_concurrent_kernel = 32; }
            else if(ref_device_prop_received.minor == 2) { tmp_max_concurrent_kernel = 16; }
            else { PRINT_FORMAT("%s: Unknown minor device version." NEW_LINE, MyEA::String::Get__Time().c_str()); }
                break;
        case 7: // Volta.
            if(ref_device_prop_received.minor == 0) { tmp_max_concurrent_kernel = 128; }
            else { PRINT_FORMAT("%s: Unknown minor device version." NEW_LINE, MyEA::String::Get__Time().c_str()); }
                break;
        default: PRINT_FORMAT("%s: Unknown major device version." NEW_LINE, MyEA::String::Get__Time().c_str());  break;
    }
    
    return(tmp_max_concurrent_kernel);
}

size_t CUDA__Number_CUDA_Cores(struct cudaDeviceProp const &ref_device_prop_received)
{
    size_t const tmp_multiprocessor(static_cast<size_t>(ref_device_prop_received.multiProcessorCount));
    size_t tmp_cores(0_zu);

    switch(ref_device_prop_received.major)
    {
        case 2: // Fermi.
            if(ref_device_prop_received.minor == 1) { tmp_cores = tmp_multiprocessor * 48_zu; }
            else { tmp_cores = tmp_multiprocessor * 32_zu; }
                break;
        case 3: // Kepler. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-3-0
            tmp_cores = tmp_multiprocessor * 192_zu;
                break;
        case 5: // Maxwell. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-5-x
            tmp_cores = tmp_multiprocessor * 128_zu;
                break;
        case 6: // Pascal. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-6-x
            if(ref_device_prop_received.minor == 0) { tmp_cores = tmp_multiprocessor * 64_zu; }
            else if(ref_device_prop_received.minor == 1 ||
                     ref_device_prop_received.minor == 2) { tmp_cores = tmp_multiprocessor * 128_zu; }
            else { PRINT_FORMAT("%s: Unknown minor device version." NEW_LINE, MyEA::String::Get__Time().c_str()); }
                break;
        case 7: // Volta. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-7-x
            if(ref_device_prop_received.minor == 0) { tmp_cores = tmp_multiprocessor * 64_zu; }
            else { PRINT_FORMAT("%s: Unknown minor device version." NEW_LINE, MyEA::String::Get__Time().c_str()); }
                break;
        default: PRINT_FORMAT("%s: Unknown major device version." NEW_LINE, MyEA::String::Get__Time().c_str());  break;
    }
    
    return(tmp_cores);
}

bool CUDA__Input__Use__CUDA(int &ref_index_device_received, size_t &ref_maximum_allowable_memory_bytes_received)
{
    bool tmp_use_CUDA(false);

    unsigned int const tmp_number_device(CUDA__Device_Count());

    if(tmp_number_device != 0u)
    {
        struct cudaDeviceProp tmp_cuda_device;

        for(unsigned int i(0u); i != tmp_number_device; ++i)
        {
            CUDA__Safe_Call(cudaGetDeviceProperties(&tmp_cuda_device, static_cast<int>(i)));

            CUDA__Print__Device_Property(tmp_cuda_device, i);

            if(CUDA__Required_Compatibility_Device(tmp_cuda_device))
            {
                tmp_use_CUDA = MyEA::String::NoOrYes(MyEA::String::Get__Time() + ": Do you want to use CUDA with this card: " + std::string(tmp_cuda_device.name) + "?");

                ref_index_device_received = i;

                break;
            }
        }

        if(tmp_use_CUDA)
        {
            CUDA__Set__Device(ref_index_device_received);

            CUDA__Safe_Call(cudaGetDeviceProperties(&tmp_cuda_device, ref_index_device_received));
            
            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Device[%d]: %s." NEW_LINE,
                                     MyEA::String::Get__Time().c_str(),
                                     ref_index_device_received,
                                     tmp_cuda_device.name);

            size_t tmp_memory_total_bytes(0_zu),
                      tmp_memory_free_bytes(0_zu),
                      tmp_maximum_allowable_memory_bytes;

            CUDA__Safe_Call(cudaMemGetInfo(&tmp_memory_free_bytes, &tmp_memory_total_bytes));

            if(tmp_memory_free_bytes / KILOBYTE / KILOBYTE < 1_zu)
            {
                PRINT_FORMAT("%s: %s: ERROR: Not enough memory to use the GPU %zu bytes" NEW_LINE,
                                         MyEA::String::Get__Time().c_str(),
                                         __FUNCTION__,
                                         tmp_memory_free_bytes);

                return(false);
            }

            PRINT_FORMAT("%s" NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s: Maximum allowable memory." NEW_LINE, MyEA::String::Get__Time().c_str());
            PRINT_FORMAT("%s:\tRange[1, %zu] MB(s)." NEW_LINE, MyEA::String::Get__Time().c_str(), tmp_memory_free_bytes / KILOBYTE / KILOBYTE);
            
            ref_maximum_allowable_memory_bytes_received = tmp_maximum_allowable_memory_bytes = MyEA::String::Cin_Number<size_t>(1_zu,
                                                                                                                                                                                                            tmp_memory_free_bytes / KILOBYTE / KILOBYTE,
                                                                                                                                                                                                            MyEA::String::Get__Time() + ": Maximum allowable memory (MBs):");

            CUDA__Initialize__Device(tmp_cuda_device, tmp_maximum_allowable_memory_bytes * KILOBYTE * KILOBYTE);

            CUDA__Set__Synchronization_Depth(3_zu);
        }
    }

    return(tmp_use_CUDA);
}
