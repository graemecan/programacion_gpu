#include <stdio.h>

__global__ void helloFromGPU(void)
{
    printf("Hello World from the GPU!\n");
}

int main(){
    printf("Hello World from the CPU!\n");

    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset(); // Clean all resources associated with this device in this process
    return 0;
}
