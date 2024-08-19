#include <stdio.h>
#include <stdlib.h>

__global__ void idx_print()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_idx = threadIdx.x / warpSize;
    int lane_idx = threadIdx.x & (warpSize - 1);
    
    if ((lane_idx & (warpSize/2 - 1)) == 0)
        //  thread, block, warp, lane"
        printf(" %5d\t%5d\t %2d\t%2d\n", idx, blockIdx.x, warp_idx, lane_idx);
}

int main(int argc, char* argv[])
{
    if (argc == 1) {
        printf("./cuda_thread_block [grid size] [block size]");
        printf(" (e.g. ./cuda_thread_block 4 128)\n");

        exit(1);
    }

    int grid = atoi(argv[1]);
    int block = atoi(argv[2]);

    printf("thread, block, warp, lane\n");
    idx_print<<<grid, block>>>();
    cudaDeviceSynchronize();
}
