#include <Tools/CUDA_Configuration.cuh>
#include <CUDA/CUDA_Shared_Memory.cuh>
    
namespace Transpose
{
    // TODO: Fix. Some dimension like 25x25 have bank conflict.
    template <typename T, size_t TILE_WIDTH>
    __global__ void kernel__Transpose_Square(size_t const size_received,
                                                                    size_t const width_received,
                                                                    T *const ptr_array_outputs_received,
                                                                    T const *const ptr_array_inputs_received)
    {
        size_t const &tmp_thread_block_index_x(threadIdx.x),
                                    &tmp_thread_block_index_y(threadIdx.y),
                                    tmp_thread_block_diagonal((tmp_thread_block_index_x + tmp_thread_block_index_y) % TILE_WIDTH),
                                    tmp_grid_stride_x(gridDim.x * TILE_WIDTH * 2u),
                                    tmp_grid_stride_y(gridDim.y * TILE_WIDTH * 2u);
        size_t tmp_thread_global_index_block_x(blockIdx.x * TILE_WIDTH * 2u),
                            tmp_thread_global_index_block_y(blockIdx.y * TILE_WIDTH * 2u),
                            tmp_thread_global_index_x,
                            tmp_thread_global_index_y,
                            tmp_thread_global_index_offSet__x,
                            tmp_thread_global_index_offSet__y;
    
        __shared__ T tmp_array_tile[TILE_WIDTH * 4u][TILE_WIDTH];
    
        while(tmp_thread_global_index_block_y < width_received)
        {
            while(tmp_thread_global_index_block_x < width_received)
            {
                // Coalesced index X.
                // 0 * 32 * 2 + [0...1...31] = 0 + [0...1...31]
                // 1 * 32 * 2 + [0...1...31] = 64 + [0...1...31]
                tmp_thread_global_index_x = tmp_thread_global_index_block_x + tmp_thread_block_index_x;
                tmp_thread_global_index_offSet__x = tmp_thread_global_index_x + TILE_WIDTH;

                // Coalesced index Y.
                // 0 * 32 * 2 + [0...1...31] = 0 + [0...1...31]
                // 1 * 32 * 2 + [0...1...31] = 64 + [0...1...31]
                tmp_thread_global_index_y = tmp_thread_global_index_block_y + tmp_thread_block_index_y;
                tmp_thread_global_index_offSet__y = tmp_thread_global_index_y + TILE_WIDTH;

                if(tmp_thread_global_index_offSet__x < width_received && tmp_thread_global_index_offSet__y < width_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: [0][0], [0][1], [0][2], [0][3], [0][4], [0][5], [0][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 8 + 0 == 000], [0 * 8 + 1 == 001], [0 * 8 + 2 == 002], [0 * 8 + 3 == 003], [0 * 8 + 4 == 004], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_y * width_received + tmp_thread_global_index_x];
                    // Coalesced Shared Memory:
                    // Y[0]: [0][32], [0][33], [0][34], [0][35], [0][36], [0][37], [0][38], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 8 + 1 == 000], [0 * 8 + 1 == 001], [0 * 8 + 2 == 002], [0 * 8 + 3 == 003], [0 * 8 + 4 == 004], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x + TILE_WIDTH][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_y * width_received + tmp_thread_global_index_offSet__x];
                    // Coalesced Shared Memory:
                    // Y[0 + 32]: [32][0], [32][1], [32][2], [32][3], [32][4], [32][5], [32][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[1]: ... = [1 * 8 + 0 == 416], [1 * 384 + 33 == 417], [1 * 384 + 34 == 418], [1 * 384 + 35 == 419], [1 * 384 + 36 == 420], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x + 2u * TILE_WIDTH][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_offSet__y * width_received + tmp_thread_global_index_x];
                    // Coalesced Shared Memory:
                    // Y[0 + 32]: [32][32], [32][33], [32][34], [32][35], [32][36], [32][37], [32][38], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0 + 32]: ... = [32 * 384 + 32 == 12'320], [32 * 384 + 33 == 12'321], [32 * 384 + 34 == 12'322], [32 * 384 + 35 == 12'323], [32 * 384 + 36 == 12'324], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x + 3u * TILE_WIDTH][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_offSet__y * width_received + tmp_thread_global_index_offSet__x];
                }
                else if(tmp_thread_global_index_offSet__x < width_received && tmp_thread_global_index_y < width_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: [0][0], [0][1], [0][2], [0][3], [0][4], [0][5], [0][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 8 + 0 == 000], [0 * 8 + 1 == 001], [0 * 8 + 2 == 002], [0 * 8 + 3 == 003], [0 * 8 + 4 == 004], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_y * width_received + tmp_thread_global_index_x];
                    // Coalesced Shared Memory:
                    // Y[0]: [0][32], [0][33], [0][34], [0][35], [0][36], [0][37], [0][38], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 8 + 0 == 000], [0 * 8 + 1 == 001], [0 * 8 + 2 == 002], [0 * 8 + 3 == 003], [0 * 8 + 4 == 004], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x + TILE_WIDTH][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_y * width_received + tmp_thread_global_index_offSet__x];
                }
                else if(tmp_thread_global_index_x < width_received && tmp_thread_global_index_offSet__y < width_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: [0][0], [0][1], [0][2], [0][3], [0][4], [0][5], [0][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 0 == 000], [0 * 384 + 1 == 001], [0 * 384 + 2 == 002], [0 * 384 + 3 == 003], [0 * 384 + 4 == 004], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_y * width_received + tmp_thread_global_index_x];
                    // Coalesced Shared Memory:
                    // Y[0 + 32]: [32][0], [32][1], [32][2], [32][3], [32][4], [32][5], [32][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0 + 32]: ... = [32 * 384 + 0 == 12'288], [32 * 384 + 1 == 12'289], [32 * 384 + 2 == 12'290], [32 * 384 + 3 == 12'291], [32 * 384 + 4 == 12'292], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x + 2u * TILE_WIDTH][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_offSet__y * width_received + tmp_thread_global_index_x];
                }
                else if(tmp_thread_global_index_x < width_received && tmp_thread_global_index_y < width_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: [0][0], [0][1], [0][2], [0][3], [0][4], [0][5], [0][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 0 == 000], [0 * 384 + 1 == 001], [0 * 384 + 2 == 002], [0 * 384 + 3 == 003], [0 * 384 + 4 == 004], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_y * width_received + tmp_thread_global_index_x];
                }

                // ThreadBlock synchronization.
                __syncthreads();
                
                // Transpose X.
                tmp_thread_global_index_x = tmp_thread_global_index_block_y + tmp_thread_block_index_x;
                tmp_thread_global_index_offSet__x = tmp_thread_global_index_x + TILE_WIDTH;

                // Transpose Y.
                tmp_thread_global_index_y = tmp_thread_global_index_block_x + tmp_thread_block_index_y;
                tmp_thread_global_index_offSet__y = tmp_thread_global_index_y + TILE_WIDTH;

                if(tmp_thread_global_index_offSet__x < width_received && tmp_thread_global_index_offSet__y < width_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: ... = [0][0], [1][0], [2][0], [3][0], [4][0], [5][0], [6][0], ..., [X][Y]
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 0 == 000], [0 * 384 + 1 == 001], [0 * 384 + 2 == 002], [0 * 384 + 3 == 003], [0 * 384 + 4 == 004], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_y * width_received + tmp_thread_global_index_x] = tmp_array_tile[tmp_thread_block_index_y][tmp_thread_block_diagonal];
                    // Coalesced Shared Memory:
                    // Y[0]: [0][32], [0][33], [0][34], [0][35], [0][36], [0][37], [0][38], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 32 == 032], [0 * 384 + 33 == 033], [0 * 384 + 34 == 034], [0 * 384 + 35 == 035], [0 * 384 + 36 == 036], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_y * width_received + tmp_thread_global_index_offSet__x] = tmp_array_tile[tmp_thread_block_index_y + 2u * TILE_WIDTH][tmp_thread_block_diagonal];
                    // Coalesced Shared Memory:
                    // Y[0 + 32]: [32][0], [32][1], [32][2], [32][3], [32][4], [32][5], [32][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0 + 32]: ... = [32 * 384 + 0 == 12'288], [32 * 384 + 1 == 12'289], [32 * 384 + 2 == 12'290], [32 * 384 + 3 == 12'291], [32 * 384 + 4 == 12'292], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_offSet__y * width_received + tmp_thread_global_index_x] = tmp_array_tile[tmp_thread_block_index_y + TILE_WIDTH][tmp_thread_block_diagonal];
                    // Coalesced Shared Memory:
                    // Y[0 + 32]: [32][32], [32][33], [32][34], [32][35], [32][36], [32][37], [32][38], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0 + 32]: ... = [32 * 384 + 32 == 12'320], [32 * 384 + 33 == 12'321], [32 * 384 + 34 == 12'322], [32 * 384 + 35 == 12'323], [32 * 384 + 36 == 12'324], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_offSet__y * width_received + tmp_thread_global_index_offSet__x] = tmp_array_tile[tmp_thread_block_index_y + 3u * TILE_WIDTH][tmp_thread_block_diagonal];
                }
                else if(tmp_thread_global_index_offSet__x < width_received && tmp_thread_global_index_y < width_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: ... = [0][0], [1][0], [2][0], [3][0], [4][0], [5][0], [6][0], ..., [X][Y]
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 0 == 000], [0 * 384 + 1 == 001], [0 * 384 + 2 == 002], [0 * 384 + 3 == 003], [0 * 384 + 4 == 004], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_y * width_received + tmp_thread_global_index_x] = tmp_array_tile[tmp_thread_block_index_y][tmp_thread_block_diagonal];
                    // Coalesced Shared Memory:
                    // Y[0]: [0][32], [0][33], [0][34], [0][35], [0][36], [0][37], [0][38], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 32 == 032], [0 * 384 + 33 == 033], [0 * 384 + 34 == 034], [0 * 384 + 35 == 035], [0 * 384 + 36 == 036], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_y * width_received + tmp_thread_global_index_offSet__x] = tmp_array_tile[tmp_thread_block_index_y + 2u * TILE_WIDTH][tmp_thread_block_diagonal];
                }
                else if(tmp_thread_global_index_x < width_received && tmp_thread_global_index_offSet__y < width_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: ... = [0][0], [1][0], [2][0], [3][0], [4][0], [5][0], [6][0], ..., [X][Y]
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 0 == 000], [0 * 384 + 1 == 001], [0 * 384 + 2 == 002], [0 * 384 + 3 == 003], [0 * 384 + 4 == 004], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_y * width_received + tmp_thread_global_index_x] = tmp_array_tile[tmp_thread_block_index_y][tmp_thread_block_diagonal];
                    // Coalesced Shared Memory:
                    // Y[0 + 32]: [32][0], [32][1], [32][2], [32][3], [32][4], [32][5], [32][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0 + 32]: ... = [32 * 384 + 0 == 12'288], [32 * 384 + 1 == 12'289], [32 * 384 + 2 == 12'290], [32 * 384 + 3 == 12'291], [32 * 384 + 4 == 12'292], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_offSet__y * width_received + tmp_thread_global_index_x] = tmp_array_tile[tmp_thread_block_index_y + TILE_WIDTH][tmp_thread_block_diagonal];
                }
                else if(tmp_thread_global_index_x < width_received && tmp_thread_global_index_y < width_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: ... = [0][0], [1][0], [2][0], [3][0], [4][0], [5][0], [6][0], ..., [X][Y]
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 0 == 000], [0 * 384 + 1 == 001], [0 * 384 + 2 == 002], [0 * 384 + 3 == 003], [0 * 384 + 4 == 004], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_y * width_received + tmp_thread_global_index_x] = tmp_array_tile[tmp_thread_block_index_y][tmp_thread_block_diagonal];
                }
                
                // ThreadBlock synchronization.
                __syncthreads();
                
                // Increment X.
                tmp_thread_global_index_block_x += tmp_grid_stride_x;
            }

            // Reset X.
            tmp_thread_global_index_block_x = blockIdx.x * TILE_WIDTH * 2u;

            // Increment Y.
            tmp_thread_global_index_block_y += tmp_grid_stride_y;
        }
    }

    template<typename T>
    __device__ void Launch_Transpose_Square(size_t size_received,
                                                                    size_t width_received,
                                                                    T *const ptr_array_outputs_received,
                                                                    T const *const ptr_array_inputs_received,
                                                                    struct dim3 const *const ptr_dimension_grid_recieved,
                                                                    struct dim3 const *const ptr_dimension_block_recieved)
    {
        switch(ptr_dimension_block_recieved->x)
        {
            case 1u:
                kernel__Transpose_Square<T, 1u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 2u:
                kernel__Transpose_Square<T, 2u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 3u:
                kernel__Transpose_Square<T, 3u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 4u:
                kernel__Transpose_Square<T, 4u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 5u:
                kernel__Transpose_Square<T, 5u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 6u:
                kernel__Transpose_Square<T, 6u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 7u:
                kernel__Transpose_Square<T, 7u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 8u:
                kernel__Transpose_Square<T, 8u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 9u:
                kernel__Transpose_Square<T, 9u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 10u:
                kernel__Transpose_Square<T, 10u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 11u:
                kernel__Transpose_Square<T, 11u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 12u:
                kernel__Transpose_Square<T, 12u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 13u:
                kernel__Transpose_Square<T, 13u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 14u:
                kernel__Transpose_Square<T, 14u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 15u:
                kernel__Transpose_Square<T, 15u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 16u:
                kernel__Transpose_Square<T, 16u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 17u:
                kernel__Transpose_Square<T, 17u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 18u:
                kernel__Transpose_Square<T, 18u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 19u:
                kernel__Transpose_Square<T, 19u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 20u:
                kernel__Transpose_Square<T, 20u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 21u:
                kernel__Transpose_Square<T, 21u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 22u:
                kernel__Transpose_Square<T, 22u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 23u:
                kernel__Transpose_Square<T, 23u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 24u:
                kernel__Transpose_Square<T, 24u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 25u:
                kernel__Transpose_Square<T, 25u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 26u:
                kernel__Transpose_Square<T, 26u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 27u:
                kernel__Transpose_Square<T, 27u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 28u:
                kernel__Transpose_Square<T, 28u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 29u:
                kernel__Transpose_Square<T, 29u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 30u:
                kernel__Transpose_Square<T, 30u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 31u:
                kernel__Transpose_Square<T, 31u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            case 32u:
                kernel__Transpose_Square<T, 32u> <<< *ptr_dimension_grid_recieved,
                                                                             *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                  width_received,
                                                                                                                                  ptr_array_outputs_received,
                                                                                                                                  ptr_array_inputs_received);
                    break;
            default:
                PRINT_FORMAT("%s: ERROR: Invalid dimension %ux%u!" NEW_LINE,
                                        __FUNCTION__,
                                        ptr_dimension_block_recieved->x, ptr_dimension_block_recieved->y);
                    break;
        }
    }

    template<typename T>
    __device__ void Transpose_Square(size_t const size_received,
                                                        size_t const width_received,
                                                        T *const ptr_array_outputs_received,
                                                        T const *const ptr_array_inputs_received,
                                                        struct dim3 const *const ptr_dimension_grid_recieved,
                                                        struct dim3 const *const ptr_dimension_block_recieved)
    {
        /* E.g: 
            Width = 2;
                
            Output:
                Index = row * width + column;
                0 * 2 + 0 = [0]; 0 * 2 + 1 = [1];
                1 * 2 + 0 = [2]; 1 * 2 + 1 = [3];
                
            Input:
                Index = column * width + row;
                0 * 2 + 0 = [0]; 1 * 2 + 0 = [2];
                0 * 2 + 1 = [1]; 1 * 2 + 1 = [3]; */

        if(USE_PARALLEL && size_received >= warpSize)
        {
            Launch_Transpose_Square<T>(size_received,
                                                          width_received,
                                                          ptr_array_outputs_received,
                                                          ptr_array_inputs_received,
                                                          ptr_dimension_grid_recieved,
                                                          ptr_dimension_block_recieved);
        }
        else
        {
            for(size_t row(0u),
                                   column(0u); column != width_received; ++column)
            {
                for(row = 0u; row != width_received; ++row)
                { ptr_array_outputs_received[row * width_received + column] = ptr_array_inputs_received[column * width_received + row]; }
            }

            /*
            PRINT_FORMAT("Input %ux%u: " NEW_LINE, width_received, width_received);
            for(size_t column(0u),
                                   row(0u); row != width_received; ++row)
            {
                for(column = 0u; column != width_received; ++column)
                {
                    PRINT_FORMAT("[%f] ", ptr_array_inputs_received[column * width_received + row]);
                }

                PRINT_FORMAT(NEW_LINE);
            }
            
            PRINT_FORMAT("Output %ux%u: " NEW_LINE, width_received, width_received);
            for(size_t column(0u),
                                   row(0u); row != width_received; ++row)
            {
                for(column = 0u; column != width_received; ++column)
                {
                    PRINT_FORMAT("[%f] ", ptr_array_outputs_received[column * width_received + row]);
                }

                PRINT_FORMAT(NEW_LINE);
            }
            */
        }
    }
    
    // TODO: Fix. Some dimension like 25x25 have bank conflict.
    template <typename T, size_t TILE_WIDTH>
    __global__ void kernel__Transpose_Rectangular(size_t const size_received,
                                                                          size_t const rows_received,
                                                                          size_t const columns_received,
                                                                          T *const ptr_array_outputs_received,
                                                                          T const *const ptr_array_inputs_received)
    {
        size_t const &tmp_thread_block_index_x(threadIdx.x),
                                    &tmp_thread_block_index_y(threadIdx.y),
                                    tmp_thread_block_diagonal((tmp_thread_block_index_x + tmp_thread_block_index_y) % TILE_WIDTH),
                                    tmp_grid_stride_x(gridDim.x * TILE_WIDTH * 2u),
                                    tmp_grid_stride_y(gridDim.y * TILE_WIDTH * 2u);
        size_t tmp_thread_global_index_block_x(blockIdx.x * TILE_WIDTH * 2u),
                            tmp_thread_global_index_block_y(blockIdx.y * TILE_WIDTH * 2u),
                            tmp_thread_global_index_x,
                            tmp_thread_global_index_y,
                            tmp_thread_global_index_offSet__x,
                            tmp_thread_global_index_offSet__y;
    
        __shared__ T tmp_array_tile[TILE_WIDTH * 4u][TILE_WIDTH];
    
        while(tmp_thread_global_index_block_y < columns_received)
        {
            while(tmp_thread_global_index_block_x < rows_received)
            {
                // Coalesced index X.
                // 0 * 32 * 2 + [0...1...31] = 0 + [0...1...31]
                // 1 * 32 * 2 + [0...1...31] = 64 + [0...1...31]
                tmp_thread_global_index_x = tmp_thread_global_index_block_x + tmp_thread_block_index_x;
                tmp_thread_global_index_offSet__x = tmp_thread_global_index_x + TILE_WIDTH;

                // Coalesced index Y.
                // 0 * 32 * 2 + [0...1...31] = 0 + [0...1...31]
                // 1 * 32 * 2 + [0...1...31] = 64 + [0...1...31]
                tmp_thread_global_index_y = tmp_thread_global_index_block_y + tmp_thread_block_index_y;
                tmp_thread_global_index_offSet__y = tmp_thread_global_index_y + TILE_WIDTH;

                if(tmp_thread_global_index_offSet__x < rows_received && tmp_thread_global_index_offSet__y < columns_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: [0][0], [0][1], [0][2], [0][3], [0][4], [0][5], [0][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 8 + 0 == 000], [0 * 8 + 1 == 001], [0 * 8 + 2 == 002], [0 * 8 + 3 == 003], [0 * 8 + 4 == 004], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_y * rows_received + tmp_thread_global_index_x];
                    // Coalesced Shared Memory:
                    // Y[0]: [0][32], [0][33], [0][34], [0][35], [0][36], [0][37], [0][38], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 8 + 1 == 000], [0 * 8 + 1 == 001], [0 * 8 + 2 == 002], [0 * 8 + 3 == 003], [0 * 8 + 4 == 004], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x + TILE_WIDTH][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_y * rows_received + tmp_thread_global_index_offSet__x];
                    // Coalesced Shared Memory:
                    // Y[0 + 32]: [32][0], [32][1], [32][2], [32][3], [32][4], [32][5], [32][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[1]: ... = [1 * 8 + 0 == 416], [1 * 384 + 33 == 417], [1 * 384 + 34 == 418], [1 * 384 + 35 == 419], [1 * 384 + 36 == 420], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x + 2u * TILE_WIDTH][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_offSet__y * rows_received + tmp_thread_global_index_x];
                    // Coalesced Shared Memory:
                    // Y[0 + 32]: [32][32], [32][33], [32][34], [32][35], [32][36], [32][37], [32][38], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0 + 32]: ... = [32 * 384 + 32 == 12'320], [32 * 384 + 33 == 12'321], [32 * 384 + 34 == 12'322], [32 * 384 + 35 == 12'323], [32 * 384 + 36 == 12'324], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x + 3u * TILE_WIDTH][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_offSet__y * rows_received + tmp_thread_global_index_offSet__x];
                }
                else if(tmp_thread_global_index_offSet__x < rows_received && tmp_thread_global_index_y < columns_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: [0][0], [0][1], [0][2], [0][3], [0][4], [0][5], [0][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 8 + 0 == 000], [0 * 8 + 1 == 001], [0 * 8 + 2 == 002], [0 * 8 + 3 == 003], [0 * 8 + 4 == 004], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_y * rows_received + tmp_thread_global_index_x];
                    // Coalesced Shared Memory:
                    // Y[0]: [0][32], [0][33], [0][34], [0][35], [0][36], [0][37], [0][38], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 8 + 0 == 000], [0 * 8 + 1 == 001], [0 * 8 + 2 == 002], [0 * 8 + 3 == 003], [0 * 8 + 4 == 004], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x + TILE_WIDTH][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_y * rows_received + tmp_thread_global_index_offSet__x];
                }
                else if(tmp_thread_global_index_x < rows_received && tmp_thread_global_index_offSet__y < columns_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: [0][0], [0][1], [0][2], [0][3], [0][4], [0][5], [0][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 0 == 000], [0 * 384 + 1 == 001], [0 * 384 + 2 == 002], [0 * 384 + 3 == 003], [0 * 384 + 4 == 004], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_y * rows_received + tmp_thread_global_index_x];
                    // Coalesced Shared Memory:
                    // Y[0 + 32]: [32][0], [32][1], [32][2], [32][3], [32][4], [32][5], [32][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0 + 32]: ... = [32 * 384 + 0 == 12'288], [32 * 384 + 1 == 12'289], [32 * 384 + 2 == 12'290], [32 * 384 + 3 == 12'291], [32 * 384 + 4 == 12'292], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x + 2u * TILE_WIDTH][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_offSet__y * rows_received + tmp_thread_global_index_x];
                }
                else if(tmp_thread_global_index_x < rows_received && tmp_thread_global_index_y < columns_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: [0][0], [0][1], [0][2], [0][3], [0][4], [0][5], [0][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 0 == 000], [0 * 384 + 1 == 001], [0 * 384 + 2 == 002], [0 * 384 + 3 == 003], [0 * 384 + 4 == 004], ..., [Y * WIDTH + X]
                    tmp_array_tile[tmp_thread_block_index_x][tmp_thread_block_diagonal] = ptr_array_inputs_received[tmp_thread_global_index_y * rows_received + tmp_thread_global_index_x];
                }

                // ThreadBlock synchronization.
                __syncthreads();
                
                // Transpose X.
                tmp_thread_global_index_x = tmp_thread_global_index_block_y + tmp_thread_block_index_x;
                tmp_thread_global_index_offSet__x = tmp_thread_global_index_x + TILE_WIDTH;

                // Transpose Y.
                tmp_thread_global_index_y = tmp_thread_global_index_block_x + tmp_thread_block_index_y;
                tmp_thread_global_index_offSet__y = tmp_thread_global_index_y + TILE_WIDTH;

                if(tmp_thread_global_index_offSet__x < columns_received && tmp_thread_global_index_offSet__y < rows_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: ... = [0][0], [1][0], [2][0], [3][0], [4][0], [5][0], [6][0], ..., [X][Y]
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 0 == 000], [0 * 384 + 1 == 001], [0 * 384 + 2 == 002], [0 * 384 + 3 == 003], [0 * 384 + 4 == 004], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_y * columns_received + tmp_thread_global_index_x] = tmp_array_tile[tmp_thread_block_index_y][tmp_thread_block_diagonal];
                    // Coalesced Shared Memory:
                    // Y[0]: [0][32], [0][33], [0][34], [0][35], [0][36], [0][37], [0][38], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 32 == 032], [0 * 384 + 33 == 033], [0 * 384 + 34 == 034], [0 * 384 + 35 == 035], [0 * 384 + 36 == 036], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_y * columns_received + tmp_thread_global_index_offSet__x] = tmp_array_tile[tmp_thread_block_index_y + 2u * TILE_WIDTH][tmp_thread_block_diagonal];
                    // Coalesced Shared Memory:
                    // Y[0 + 32]: [32][0], [32][1], [32][2], [32][3], [32][4], [32][5], [32][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0 + 32]: ... = [32 * 384 + 0 == 12'288], [32 * 384 + 1 == 12'289], [32 * 384 + 2 == 12'290], [32 * 384 + 3 == 12'291], [32 * 384 + 4 == 12'292], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_offSet__y * columns_received + tmp_thread_global_index_x] = tmp_array_tile[tmp_thread_block_index_y + TILE_WIDTH][tmp_thread_block_diagonal];
                    // Coalesced Shared Memory:
                    // Y[0 + 32]: [32][32], [32][33], [32][34], [32][35], [32][36], [32][37], [32][38], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0 + 32]: ... = [32 * 384 + 32 == 12'320], [32 * 384 + 33 == 12'321], [32 * 384 + 34 == 12'322], [32 * 384 + 35 == 12'323], [32 * 384 + 36 == 12'324], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_offSet__y * columns_received + tmp_thread_global_index_offSet__x] = tmp_array_tile[tmp_thread_block_index_y + 3u * TILE_WIDTH][tmp_thread_block_diagonal];
                }
                else if(tmp_thread_global_index_offSet__x < columns_received && tmp_thread_global_index_y < rows_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: ... = [0][0], [1][0], [2][0], [3][0], [4][0], [5][0], [6][0], ..., [X][Y]
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 0 == 000], [0 * 384 + 1 == 001], [0 * 384 + 2 == 002], [0 * 384 + 3 == 003], [0 * 384 + 4 == 004], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_y * columns_received + tmp_thread_global_index_x] = tmp_array_tile[tmp_thread_block_index_y][tmp_thread_block_diagonal];
                    // Coalesced Shared Memory:
                    // Y[0]: [0][32], [0][33], [0][34], [0][35], [0][36], [0][37], [0][38], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 32 == 032], [0 * 384 + 33 == 033], [0 * 384 + 34 == 034], [0 * 384 + 35 == 035], [0 * 384 + 36 == 036], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_y * columns_received + tmp_thread_global_index_offSet__x] = tmp_array_tile[tmp_thread_block_index_y + 2u * TILE_WIDTH][tmp_thread_block_diagonal];
                }
                else if(tmp_thread_global_index_x < columns_received && tmp_thread_global_index_offSet__y < rows_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: ... = [0][0], [1][0], [2][0], [3][0], [4][0], [5][0], [6][0], ..., [X][Y]
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 0 == 000], [0 * 384 + 1 == 001], [0 * 384 + 2 == 002], [0 * 384 + 3 == 003], [0 * 384 + 4 == 004], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_y * columns_received + tmp_thread_global_index_x] = tmp_array_tile[tmp_thread_block_index_y][tmp_thread_block_diagonal];
                    // Coalesced Shared Memory:
                    // Y[0 + 32]: [32][0], [32][1], [32][2], [32][3], [32][4], [32][5], [32][6], ..., [Y][X] = ...
                    // Coalesced Global Memory:
                    // Y[0 + 32]: ... = [32 * 384 + 0 == 12'288], [32 * 384 + 1 == 12'289], [32 * 384 + 2 == 12'290], [32 * 384 + 3 == 12'291], [32 * 384 + 4 == 12'292], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_offSet__y * columns_received + tmp_thread_global_index_x] = tmp_array_tile[tmp_thread_block_index_y + TILE_WIDTH][tmp_thread_block_diagonal];
                }
                else if(tmp_thread_global_index_x < columns_received && tmp_thread_global_index_y < rows_received)
                {
                    // Coalesced Shared Memory:
                    // Y[0]: ... = [0][0], [1][0], [2][0], [3][0], [4][0], [5][0], [6][0], ..., [X][Y]
                    // Coalesced Global Memory:
                    // Y[0]: ... = [0 * 384 + 0 == 000], [0 * 384 + 1 == 001], [0 * 384 + 2 == 002], [0 * 384 + 3 == 003], [0 * 384 + 4 == 004], ..., [Y * WIDTH + X]
                    ptr_array_outputs_received[tmp_thread_global_index_y * columns_received + tmp_thread_global_index_x] = tmp_array_tile[tmp_thread_block_index_y][tmp_thread_block_diagonal];
                }
                
                // ThreadBlock synchronization.
                __syncthreads();
                
                // Increment X.
                tmp_thread_global_index_block_x += tmp_grid_stride_x;
            }

            // Reset X.
            tmp_thread_global_index_block_x = blockIdx.x * TILE_WIDTH * 2u;

            // Increment Y.
            tmp_thread_global_index_block_y += tmp_grid_stride_y;
        }
    }
    
    template<typename T>
    __device__ void Launch_Transpose_Rectangular(size_t size_received,
                                                                           size_t const rows_received,
                                                                           size_t const columns_received,
                                                                           T *const ptr_array_outputs_received,
                                                                           T const *const ptr_array_inputs_received,
                                                                           struct dim3 const *const ptr_dimension_grid_recieved,
                                                                           struct dim3 const *const ptr_dimension_block_recieved)
    {
        switch(ptr_dimension_block_recieved->x)
        {
            case 1u:
                kernel__Transpose_Rectangular<T, 1u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 2u:
                kernel__Transpose_Rectangular<T, 2u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 3u:
                kernel__Transpose_Rectangular<T, 3u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 4u:
                kernel__Transpose_Rectangular<T, 4u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 5u:
                kernel__Transpose_Rectangular<T, 5u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 6u:
                kernel__Transpose_Rectangular<T, 6u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 7u:
                kernel__Transpose_Rectangular<T, 7u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 8u:
                kernel__Transpose_Rectangular<T, 8u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 9u:
                kernel__Transpose_Rectangular<T, 9u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 10u:
                kernel__Transpose_Rectangular<T, 10u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 11u:
                kernel__Transpose_Rectangular<T, 11u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 12u:
                kernel__Transpose_Rectangular<T, 12u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 13u:
                kernel__Transpose_Rectangular<T, 13u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 14u:
                kernel__Transpose_Rectangular<T, 14u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 15u:
                kernel__Transpose_Rectangular<T, 15u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 16u:
                kernel__Transpose_Rectangular<T, 16u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 17u:
                kernel__Transpose_Rectangular<T, 17u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 18u:
                kernel__Transpose_Rectangular<T, 18u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 19u:
                kernel__Transpose_Rectangular<T, 19u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 20u:
                kernel__Transpose_Rectangular<T, 20u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 21u:
                kernel__Transpose_Rectangular<T, 21u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 22u:
                kernel__Transpose_Rectangular<T, 22u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 23u:
                kernel__Transpose_Rectangular<T, 23u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 24u:
                kernel__Transpose_Rectangular<T, 24u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 25u:
                kernel__Transpose_Rectangular<T, 25u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 26u:
                kernel__Transpose_Rectangular<T, 26u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 27u:
                kernel__Transpose_Rectangular<T, 27u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 28u:
                kernel__Transpose_Rectangular<T, 28u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 29u:
                kernel__Transpose_Rectangular<T, 29u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 30u:
                kernel__Transpose_Rectangular<T, 30u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 31u:
                kernel__Transpose_Rectangular<T, 31u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            case 32u:
                kernel__Transpose_Rectangular<T, 32u> <<< *ptr_dimension_grid_recieved, *ptr_dimension_block_recieved >>> (size_received,
                                                                                                                                                                                   rows_received,
                                                                                                                                                                                   columns_received,
                                                                                                                                                                                   ptr_array_outputs_received,
                                                                                                                                                                                   ptr_array_inputs_received);
                    break;
            default:
                PRINT_FORMAT("%s: ERROR: Invalid dimension %ux%u!" NEW_LINE,
                                        __FUNCTION__,
                                        ptr_dimension_block_recieved->x, ptr_dimension_block_recieved->y);
                    break;
        }
    }

    template<typename T>
    __device__ void Transpose_Rectangular(size_t const size_received,
                                                              size_t const rows_length_received,
                                                              size_t const columns_length_received,
                                                              T *const ptr_array_outputs_received,
                                                              T const *const ptr_array_inputs_received,
                                                              struct dim3 const *const ptr_dimension_grid_recieved,
                                                              struct dim3 const *const ptr_dimension_block_recieved)
    {
        /* E.g: 
            columns_length = 3;
            rows_length = 2;
                
            Output:
                Index = row * columns_length + column;
                0 * 3 + 0 = [0]; 0 * 3 + 1 = [1]; 0 * 3 + 2= [2];
                1 * 3 + 0 = [3]; 1 * 3 + 1 = [4]; 1 * 3 + 2 = [5];
                
            Input:
                Index = column * rows_length + row;
                0 * 2 + 0 = [0]; 1 * 2 + 0 = [2]; 2 * 2 + 0 = [4];
                0 * 2 + 1 = [1]; 1 * 2 + 1 = [3]; 2 * 2 + 1 = [5]; */

        if(USE_PARALLEL && size_received >= warpSize)
        {
            Launch_Transpose_Rectangular<T>(size_received,
                                                                  rows_length_received,
                                                                  columns_length_received,
                                                                  ptr_array_outputs_received,
                                                                  ptr_array_inputs_received,
                                                                  ptr_dimension_grid_recieved,
                                                                  ptr_dimension_block_recieved);
        }
        else
        {
            for(size_t row(0u),
                                   column(0u); column != columns_length_received; ++column)
            {
                for(row = 0u; row != rows_length_received; ++row)
                { ptr_array_outputs_received[row * columns_length_received + column] = ptr_array_inputs_received[column * rows_length_received + row]; }
            }

            /*
            PRINT_FORMAT("Input %ux%u: " NEW_LINE, rows_length_received, columns_length_received);
            for(size_t column(0u),
                                   row(0u); row != rows_length_received; ++row)
            {
                for(column = 0u; column != columns_length_received; ++column)
                {
                    PRINT_FORMAT("[%f] ", ptr_array_inputs_received[column * rows_length_received + row]);
                }

                PRINT_FORMAT(NEW_LINE);
            }

            PRINT_FORMAT("Output %ux%u: " NEW_LINE, columns_length_received, rows_length_received);
            for(size_t column(0u),
                                   row(0u); row != columns_length_received; ++row)
            {
                for(column = 0u; column != rows_length_received; ++column)
                {
                    PRINT_FORMAT("[%f] ", ptr_array_outputs_received[column * columns_length_received + row]);
                }

                PRINT_FORMAT(NEW_LINE);
            }
            */

            /*
            // Check error.
            size_t tmp_count_error = 0u;
            for(size_t row(0u),
                                   column(0u); column != columns_length_received; ++column)
            {
                for(row = 0u; row != rows_length_received; ++row)
                {
                    //PRINT_FORMAT("Output[%u](%f) != Input[%u](%f)" NEW_LINE,
                    //                        row * columns_length_received + column,
                    //                        ptr_array_outputs_received[row * columns_length_received + column],
                    //                        column * rows_length_received + row,
                    //                        ptr_array_inputs_received[column * rows_length_received + row]);
                    
                    if(ptr_array_outputs_received[row * columns_length_received + column] != ptr_array_inputs_received[column * rows_length_received + row])
                    { ++tmp_count_error; }
                }
            }

            PRINT_FORMAT("sequentialMatrix %ux%u Total error: %u" NEW_LINE, rows_length_received, columns_length_received, tmp_count_error);
            // |END| Check error. |END|
            */
        }
    }

    template<typename T>
    __device__ void Transpose(size_t const size_received,
                                            size_t const columns_length_received,
                                            size_t const rows_length_received,
                                            T *const ptr_array_outputs_received,
                                            T const *const ptr_array_inputs_received,
                                            struct dim3 const *const ptr_dimension_grid_recieved,
                                            struct dim3 const *const ptr_dimension_block_recieved)
    {
        if(rows_length_received == columns_length_received)
        {
            Transpose::Transpose_Square(size_received,
                                                        rows_length_received,
                                                        ptr_array_outputs_received,
                                                        ptr_array_inputs_received,
                                                        ptr_dimension_grid_recieved,
                                                        ptr_dimension_block_recieved);
        }
        else
        {
            Transpose::Transpose_Rectangular(size_received,
                                                                rows_length_received,
                                                                columns_length_received,
                                                                ptr_array_outputs_received,
                                                                ptr_array_inputs_received,
                                                                ptr_dimension_grid_recieved,
                                                                ptr_dimension_block_recieved);
        }
    }
}