#pragma once

#if defined(COMPILE_CUDA)
    bool CUDA__Input__Use__CUDA(int &ref_index_device_received, size_t &ref_maximum_allowable_memory_bytes_received);
#endif
