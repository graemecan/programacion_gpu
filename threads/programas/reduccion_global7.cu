#include <stdio.h>

__global__ void reduccion_memoria_global2(float *data, int N) {

    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Cada bloque necesita una parte del array total
    float *idata = data + blockIdx.x*blockDim.x * 2;

    // Verificar que estamos todavía dentro del array
    if (idx < N) {

        // "Loop unrolling" para aplicar una suma inmediatamente usando
        // elementos en dos *bloques*
        if (idx + blockDim.x < N) data[idx] += data[idx + blockDim.x];
        __syncthreads();

        // Reducción aplicada dentro de cada bloque
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            // Si estamos todavía dentro del bloque...
            if (threadIdx.x < stride) {
                idata[threadIdx.x] += idata[threadIdx.x + stride];
            }

            // sincronización al nivel del bloque
            __syncthreads();
        }

    }

    // guardar el resultado para este bloque en memoria global
    if (threadIdx.x == 0) data[blockIdx.x] = idata[0];
}

__global__ void reduccion_memoria_global4(float *data, int N) {

    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // Cada bloque necesita una parte del array total
    float *idata = data + blockIdx.x*blockDim.x * 4;

    // Verificar que estamos todavía dentro del array
    if (idx < N) {

        // "Loop unrolling" para aplicar una suma inmediatamente usando
        // elementos en cuatro *bloques*
        if (idx + 3 * blockDim.x < N){
            float a1 = data[idx];
            float a2 = data[idx + blockDim.x];
            float a3 = data[idx + 2*blockDim.x];
            float a4 = data[idx + 3*blockDim.x];
            data[idx] = a1 + a2 + a3 + a4;
        }
        
        __syncthreads();

        // Reducción aplicada dentro de cada bloque
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            // Si estamos todavía dentro del bloque...
            if (threadIdx.x < stride) {
                idata[threadIdx.x] += idata[threadIdx.x + stride];
            }

            // sincronización al nivel del bloque
            __syncthreads();
        }

    }

    // guardar el resultado para este bloque en memoria global
    if (threadIdx.x == 0) data[blockIdx.x] = idata[0];
}

__global__ void reduccion_memoria_global8(float *data, int N) {

    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // Cada bloque necesita una parte del array total
    float *idata = data + blockIdx.x*blockDim.x * 8;

    // Verificar que estamos todavía dentro del array
    if (idx < N) {

        // "Loop unrolling" para aplicar una suma inmediatamente usando
        // elementos en ocho *bloques*
        if (idx + 7 * blockDim.x < N){
            float a1 = data[idx];
            float a2 = data[idx + blockDim.x];
            float a3 = data[idx + 2*blockDim.x];
            float a4 = data[idx + 3*blockDim.x];
            float a5 = data[idx + 4*blockDim.x];
            float a6 = data[idx + 5*blockDim.x];
            float a7 = data[idx + 6*blockDim.x];
            float a8 = data[idx + 7*blockDim.x];
            data[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
        }
        __syncthreads();

        // Reducción aplicada dentro de cada bloque
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            // Si estamos todavía dentro del bloque...
            if (threadIdx.x < stride) {
                idata[threadIdx.x] += idata[threadIdx.x + stride];
            }

            // sincronización al nivel del bloque
            __syncthreads();
        }

    }

    // guardar el resultado para este bloque en memoria global
    if (threadIdx.x == 0) data[blockIdx.x] = idata[0];
}

void inicializar_numeros(float *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        // Generamos números aleatorios pequeños
        // El "0xFF" es un número en hexadecimal y está combinado con
        // rand() usando el "bitwise operator" & para obtener números
        // entre 0 y 255.
        data[i] = (float)(rand() & 0xFF) / (float)RAND_MAX;
    }
}

float resultado_cpu(float *data, int N)
{
    float resultado = 0.f;
    for (int i = 0; i < N; i++){
        resultado += data[i];
    }

    return resultado;
}

int main()
{
    float *h_array;
    // agregamos otro array en el host para conveniencia
    // ya que estamos invocando varios kernels
    float *h_array2;
    float *d_array;

    unsigned int N = 1 << 24;
    unsigned int n_threads = 1024;

    float resultado_host;
    float resultado_gpu;

    srand(2019);

    // Asignar memoria en el host
    h_array  = (float *)malloc(N * sizeof(float));
    h_array2 = (float *)malloc(N * sizeof(float));

    // Iniciliazar valores de h_array con números aleatorios
    inicializar_numeros(h_array, N);

    // Calcular reducción en el CPU (secuencial)
    // Ojo: antes hicimos este calculo después, pero ahora
    // vamos a usar h_array para guardar los resultados
    // parciales del GPU
    resultado_host = resultado_cpu(h_array, N);

    // Asignar memoria en el GPU y copiar datos
    cudaMalloc((void **)&d_array, N * sizeof(float));

    // ----- Kernel con unrolling factor 2 ---------------
    resultado_gpu = 0.0f;
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);
    // Calcular reducción en el GPU
    int n_bloques = (N + n_threads - 1) / n_threads;
    reduccion_memoria_global2<<<n_bloques/2, n_threads>>>(d_array, N);
    // Copiar resultado del GPU
    // Ahora copiamos los resultados parciales de cada bloque y calculamos
    // la suma final en el lado del host
    cudaMemcpy(h_array2, d_array, n_bloques/2*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0; i < n_bloques/2; i++) resultado_gpu += h_array2[i];
    // Imprimir valores a la pantalla
    printf("host: %f, device %f\n", resultado_host, resultado_gpu);
    // ---------------------------------------------------

    // ----- Kernel con unrolling factor 4 ---------------
    resultado_gpu = 0.0f;
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);
    // Calcular reducción en el GPU
    n_bloques = (N + n_threads - 1) / n_threads;
    reduccion_memoria_global4<<<n_bloques/4, n_threads>>>(d_array, N);
    // Copiar resultado del GPU
    // Ahora copiamos los resultados parciales de cada bloque y calculamos
    // la suma final en el lado del host
    cudaMemcpy(h_array2, d_array, n_bloques/4*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0; i < n_bloques/4; i++) resultado_gpu += h_array2[i];
    // Imprimir valores a la pantalla
    printf("host: %f, device %f\n", resultado_host, resultado_gpu);
    // ---------------------------------------------------

    // ----- Kernel con unrolling factor 8 ---------------
    resultado_gpu = 0.0f;
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);
    // Calcular reducción en el GPU
    n_bloques = (N + n_threads - 1) / n_threads;
    reduccion_memoria_global8<<<n_bloques/8, n_threads>>>(d_array, N);
    // Copiar resultado del GPU
    // Ahora copiamos los resultados parciales de cada bloque y calculamos
    // la suma final en el lado del host
    cudaMemcpy(h_array2, d_array, n_bloques/8*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0; i < n_bloques/8; i++) resultado_gpu += h_array2[i];
    // Imprimir valores a la pantalla
    printf("host: %f, device %f\n", resultado_host, resultado_gpu);
    // ---------------------------------------------------

    // Liberar memoria
    cudaFree(d_array);
    free(h_array);

    return 0;
}
