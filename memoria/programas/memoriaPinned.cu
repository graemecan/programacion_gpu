int main() {

    // cantidad de datos
    unsigned int N = 1<<22;
    unsigned int nbytes = N * sizeof(float);

    // asignar memoria pinned en el host
    float *h_a;
    cudaMallocHost((float **)&h_a, nbytes);

    // asignar memoria en el device
    float *d_a;
    cudaMalloc((float **)&d_a, nbytes);

    // inicializar datos en el host
    for(unsigned int i=0;i<N;i++) h_a[i] = 0.5f;

    // transferir datos del host al device
    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);

    // transferir datos del device al host
    cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);

    // liberar memoria
    cudaFree(d_a);
    cudaFreeHost(h_a);

    // reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
