#include <Tools/CUDA_Configuration.cuh>
#include <Tools/CUDA_Reallocate.cuh>
#include <CUDA/CUDA_Neural_Network.cuh>

__device__ CUDA_Storage_Dim3::CUDA_Storage_Dim3(void) { }

__device__ bool CUDA_Storage_Dim3::Get__Dim3(size_t const size_need_received,
                                                                       struct dim3 &ref_dim3_grid_received,
                                                                       struct dim3 &ref_dim3_block_received,
                                                                       class CUDA_Device_Information const *const ptr_Class_Device_Information_received,
                                                                       enum ENUM_TYPE_DIM3 const type_dim3_received)
{
    switch(type_dim3_received)
    {
        case ENUM_TYPE_DIM3::TYPE_DIM3_1D:
            this->Get__Dim3_1D(size_need_received,
                                        ref_dim3_grid_received,
                                        ref_dim3_block_received,
                                        ptr_Class_Device_Information_received);
                break;
        case ENUM_TYPE_DIM3::TYPE_DIM3_DYNAMIC_PARALLELISM:
            this->Get__Dim3_Dynamic_Parallelisme(size_need_received,
                                                                    ref_dim3_grid_received,
                                                                    ref_dim3_block_received,
                                                                    ptr_Class_Device_Information_received);
                break;
    }

    return(false);
}

__device__ bool CUDA_Storage_Dim3::Get__Dim3_1D(size_t const size_need_received,
                                                                             struct dim3 &ref_dim3_grid_received,
                                                                             struct dim3 &ref_dim3_block_received,
                                                                             class CUDA_Device_Information const *const ptr_Class_Device_Information_received)
{
    int tmp_index_find(-1);

    if(this->_size_1D != 0)
    {
        int i(0);

        do
        {
            if(this->_ptr_array_cache_dim3_size_1D[i] == size_need_received)
            {
                tmp_index_find = i;

                break;
            }
        } while(++i != this->_size_1D);

        if(tmp_index_find == -1)
        {
            tmp_index_find = this->_size_1D++;

            // Reallocate cached size.
            size_t *tmp_ptr_array_cache_dim3_size(Memory::reallocate_cpp<size_t>(this->_ptr_array_cache_dim3_size_1D,
                                                                                                                        this->_size_1D,
                                                                                                                        tmp_index_find));
            if(tmp_ptr_array_cache_dim3_size == nullptr)
            {
                PRINT_FORMAT("%s: ERROR: Can not allocate memory. pointer = new size_t[size(%u)]" NEW_LINE,
                                        __FUNCTION__,
                                        this->_size_1D);

                return(false);
            }
            this->_ptr_array_cache_dim3_size_1D = tmp_ptr_array_cache_dim3_size;
            tmp_ptr_array_cache_dim3_size[tmp_index_find] = size_need_received;
            // |END| Reallocate cached size. |END|
                
            // Reallocate struct dim3.
            struct dim3 *tmp_ptr_array_dim3_grids(Memory::reallocate(this->_ptr_array_dim3_grids_1D,
                                                                                                  this->_size_1D * sizeof(struct dim3),
                                                                                                  tmp_index_find * sizeof(struct dim3)));
            if(tmp_ptr_array_dim3_grids == NULL)
            {
                PRINT_FORMAT("%s: ERROR: Can not allocate memory. pointer = malloc(sizeof(struct dim3))" NEW_LINE,
                                        __FUNCTION__,
                                        sizeof(struct dim3));

                return(false);
            }
            this->_ptr_array_dim3_grids_1D = tmp_ptr_array_dim3_grids;
            
            struct dim3 *tmp_ptr_array_dim3_blocks(Memory::reallocate(this->_ptr_array_dim3_blocks_1D,
                                                                                                    this->_size_1D * sizeof(struct dim3),
                                                                                                    tmp_index_find * sizeof(struct dim3)));
            if(tmp_ptr_array_dim3_blocks == NULL)
            {
                PRINT_FORMAT("%s: ERROR: Can not allocate memory. pointer = malloc(sizeof(struct dim3))" NEW_LINE,
                                        __FUNCTION__,
                                        sizeof(struct dim3));

                return(false);
            }
            this->_ptr_array_dim3_blocks_1D = tmp_ptr_array_dim3_blocks;
            // |END| Reallocate struct dim3. |END|

            ptr_Class_Device_Information_received->Grid_Block_1Dimensions(size_need_received,
                                                                                                            0u,
                                                                                                            tmp_ptr_array_dim3_grids[tmp_index_find],
                                                                                                            tmp_ptr_array_dim3_blocks[tmp_index_find]);
        }
    }
    else
    {
        tmp_index_find = 0;

        // Allocate cached size.
        size_t *tmp_ptr_array_cache_dim3_size(new size_t[1u]);
        if(tmp_ptr_array_cache_dim3_size == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory. pointer = new size_t[1u]" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(size_t));

            return(false);
        }
        this->_ptr_array_cache_dim3_size_1D = tmp_ptr_array_cache_dim3_size;
        *tmp_ptr_array_cache_dim3_size = size_need_received;
        // |END| Allocate cached size. |END|
            
        // Allocate struct dim3.
        struct dim3 *tmp_ptr_array_dim3_grids(static_cast<struct dim3*>(malloc(sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_grids == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory. pointer = malloc(sizeof(struct dim3))" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(struct dim3));

            return(false);
        }
        this->_ptr_array_dim3_grids_1D = tmp_ptr_array_dim3_grids;

        struct dim3 *tmp_ptr_array_dim3_blocks(static_cast<struct dim3*>(malloc(sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_blocks == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory. pointer = malloc(sizeof(struct dim3))" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(struct dim3));

            return(false);
        }
        this->_ptr_array_dim3_blocks_1D = tmp_ptr_array_dim3_blocks;
        // |END| Allocate struct dim3. |END|

        ++this->_size_1D;

        ptr_Class_Device_Information_received->Grid_Block_1Dimensions(size_need_received,
                                                                                                        0u,
                                                                                                        *tmp_ptr_array_dim3_grids,
                                                                                                        *tmp_ptr_array_dim3_blocks);
    }

    ref_dim3_grid_received = this->_ptr_array_dim3_grids_1D[tmp_index_find];
    ref_dim3_block_received = this->_ptr_array_dim3_blocks_1D[tmp_index_find];

    return(true);
}

__device__ bool CUDA_Storage_Dim3::Get__Dim3_Memcpy(size_t const new_size_received,
                                                                                     size_t const old_size_received,
                                                                                     struct dim3 &ref_dim3_grid_zero,
                                                                                     struct dim3 &ref_dim3_block_zero,
                                                                                     struct dim3 &ref_dim3_grid_copy,
                                                                                     struct dim3 &ref_dim3_block_copy,
                                                                                     class CUDA_Device_Information const *const ptr_Class_Device_Information_received,
                                                                                     bool const memcpy_received)
{
    // If use "memcpy"
    if(memcpy_received)
    {
        if(old_size_received < new_size_received)
        {
            if(new_size_received - old_size_received >= warpSize * warpSize)
            {
                if(this->Get__Dim3_1D(new_size_received - old_size_received,
                                               ref_dim3_grid_zero,
                                               ref_dim3_block_zero,
                                               ptr_Class_Device_Information_received) == false)
                {
                    PRINT_FORMAT("%s: ERROR: From \"Get__Dim3_1D\"." NEW_LINE,
                                            __FUNCTION__);

                    return(false);
                }
            }
        
            if(old_size_received >= warpSize * warpSize)
            {
                if(this->Get__Dim3_1D(old_size_received,
                                               ref_dim3_grid_copy,
                                               ref_dim3_block_copy,
                                               ptr_Class_Device_Information_received) == false)
                {
                    PRINT_FORMAT("%s: ERROR: From \"Get__Dim3_1D\"." NEW_LINE,
                                            __FUNCTION__);

                    return(false);
                }
            }
        }
        else if(new_size_received >= warpSize * warpSize)
        {
            if(this->Get__Dim3_1D(new_size_received,
                                           ref_dim3_grid_copy,
                                           ref_dim3_block_copy,
                                           ptr_Class_Device_Information_received) == false)
            {
                PRINT_FORMAT("%s: ERROR: From \"Get__Dim3_1D\"." NEW_LINE,
                                        __FUNCTION__);

                return(false);
            }
        }
    }
    // Else if use "memset"
    else if(new_size_received >= warpSize * warpSize)
    {
        if(this->Get__Dim3_1D(new_size_received,
                                        ref_dim3_grid_zero,
                                        ref_dim3_block_zero,
                                        ptr_Class_Device_Information_received) == false)
        {
            PRINT_FORMAT("%s: ERROR: From \"Get__Dim3_1D\"." NEW_LINE,
                                    __FUNCTION__);

            return(false);
        }
    }

    return(true);
}

__device__ bool CUDA_Storage_Dim3::Get__Dim3_Dynamic_Parallelisme(size_t const size_need_received,
                                                                                                        struct dim3 &ref_dim3_grid_received,
                                                                                                        struct dim3 &ref_dim3_block_received,
                                                                                                        class CUDA_Device_Information const *const ptr_Class_Device_Information_received)
{
    int tmp_index_find(-1);

    if(this->_size_DP != 0)
    {
        int i(0);

        do
        {
            if(this->_ptr_array_cache_dim3_size_DP[i] == size_need_received)
            {
                tmp_index_find = i;

                break;
            }
        } while(++i != this->_size_DP);

        if(tmp_index_find == -1)
        {
            tmp_index_find = this->_size_DP++;

            // Reallocate cached size.
            size_t *tmp_ptr_array_cache_dim3_size(Memory::reallocate_cpp<size_t>(this->_ptr_array_cache_dim3_size_DP,
                                                                                                                        this->_size_DP,
                                                                                                                        tmp_index_find));
            if(tmp_ptr_array_cache_dim3_size == nullptr)
            {
                PRINT_FORMAT("%s: ERROR: Can not allocate memory. pointer = new size_t[size(%u)]" NEW_LINE,
                                        __FUNCTION__,
                                        this->_size_DP);

                return(false);
            }
            this->_ptr_array_cache_dim3_size_DP = tmp_ptr_array_cache_dim3_size;
            tmp_ptr_array_cache_dim3_size[tmp_index_find] = size_need_received;
            // |END| Reallocate cached size. |END|
                
            // Reallocate struct dim3.
            struct dim3 *tmp_ptr_array_dim3_grids(Memory::reallocate(this->_ptr_array_dim3_grids_DP,
                                                                                                  this->_size_1D * sizeof(struct dim3),
                                                                                                  tmp_index_find * sizeof(struct dim3)));
            if(tmp_ptr_array_dim3_grids == NULL)
            {
                PRINT_FORMAT("%s: ERROR: Can not allocate memory. pointer = malloc(sizeof(struct dim3))" NEW_LINE,
                                        __FUNCTION__,
                                        sizeof(struct dim3));

                return(false);
            }
            this->_ptr_array_dim3_grids_DP = tmp_ptr_array_dim3_grids;
            
            struct dim3 *tmp_ptr_array_dim3_blocks(Memory::reallocate(this->_ptr_array_dim3_blocks_DP,
                                                                                                                this->_size_1D * sizeof(struct dim3),
                                                                                                                tmp_index_find * sizeof(struct dim3)));
            if(tmp_ptr_array_dim3_blocks == NULL)
            {
                PRINT_FORMAT("%s: ERROR: Can not allocate memory. pointer = malloc(sizeof(struct dim3))" NEW_LINE,
                                        __FUNCTION__,
                                        sizeof(struct dim3));

                return(false);
            }
            this->_ptr_array_dim3_blocks_DP = tmp_ptr_array_dim3_blocks;
            // |END| Reallocate struct dim3. |END|

            ptr_Class_Device_Information_received->Grid_Block_Dynamic_Parallelisme(size_need_received,
                                                                                                                          0u,
                                                                                                                          tmp_ptr_array_dim3_grids[tmp_index_find],
                                                                                                                          tmp_ptr_array_dim3_blocks[tmp_index_find]);
        }
    }
    else
    {
        tmp_index_find = 0;

        // Allocate cached size.
        size_t *tmp_ptr_array_cache_dim3_size(new size_t[1u]);
        if(tmp_ptr_array_cache_dim3_size == nullptr)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory. pointer = new size_t[1u]" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(size_t));

            return(false);
        }
        this->_ptr_array_cache_dim3_size_DP = tmp_ptr_array_cache_dim3_size;
        *tmp_ptr_array_cache_dim3_size = size_need_received;
        // |END| Allocate cached size. |END|
            
        // Allocate struct dim3.
        struct dim3 *tmp_ptr_array_dim3_grids(static_cast<struct dim3*>(malloc(sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_grids == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory. pointer = malloc(sizeof(struct dim3))" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(struct dim3));

            return(false);
        }
        this->_ptr_array_dim3_grids_DP = tmp_ptr_array_dim3_grids;

        struct dim3 *tmp_ptr_array_dim3_blocks(static_cast<struct dim3*>(malloc(sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_blocks == NULL)
        {
            PRINT_FORMAT("%s: ERROR: Can not allocate memory. pointer = malloc(sizeof(struct dim3))" NEW_LINE,
                                    __FUNCTION__,
                                    sizeof(struct dim3));

            return(false);
        }
        this->_ptr_array_dim3_blocks_DP = tmp_ptr_array_dim3_blocks;
        // |END| Allocate struct dim3. |END|

        ++this->_size_DP;
            
        ptr_Class_Device_Information_received->Grid_Block_Dynamic_Parallelisme(size_need_received,
                                                                                                                     0u,
                                                                                                                     *tmp_ptr_array_dim3_grids,
                                                                                                                     *tmp_ptr_array_dim3_blocks);
    }

    ref_dim3_grid_received = this->_ptr_array_dim3_grids_DP[tmp_index_find];
    ref_dim3_block_received = this->_ptr_array_dim3_blocks_DP[tmp_index_find];

    return(true);
}

__device__ CUDA_Storage_Dim3::~CUDA_Storage_Dim3(void)
{
    if(this->_size_1D != 0)
    {
        SAFE_DELETE_ARRAY(this->_ptr_array_cache_dim3_size_1D);

        SAFE_FREE(this->_ptr_array_dim3_grids_1D);
        SAFE_FREE(this->_ptr_array_dim3_blocks_1D);
    }
    
    if(this->_size_DP != 0)
    {
        SAFE_DELETE_ARRAY(this->_ptr_array_cache_dim3_size_DP);

        SAFE_FREE(this->_ptr_array_dim3_grids_DP);
        SAFE_FREE(this->_ptr_array_dim3_blocks_DP);
    }
}
