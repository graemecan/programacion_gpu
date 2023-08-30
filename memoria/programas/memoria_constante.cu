 //declarar memoria constante
__constant__ float c_angulo[360];

// Kernel que aplica el calculo con memoria constante
__global__ void kernel_constante(float* darray)
{
      int idx;
    
      //calcular indice global de cada thread
     idx = blockIdx.x * blockDim.x + threadIdx.x;
   
    #pragma unroll 10
   for(int l=0;l<360;l++)
                 darray[idx]= darray[idx] + c_angulo[l] ;
return;

}

// Kernel que aplica el mismo calculo pero con memoria del device
__global__ void kernel_device(float* darray, float* d_angulo)
{
      int idx;
    
     // calcular indice global de cada thread
     idx = blockIdx.x * blockDim.x + threadIdx.x;
   
    #pragma unroll 10
   for(int l=0;l<360;l++)
                 darray[idx]= darray[idx] + d_angulo[l] ;
return;

}

// Copia del "kernel_device" usado para "inicializar" el GPU
__global__ void kernel_inicial(float* darray, float* d_angulo)
{
      int idx;
    
     // calcular indice global de cada thread
     idx = blockIdx.x * blockDim.x + threadIdx.x;
   
    #pragma unroll 10
   for(int l=0;l<360;l++)
                 darray[idx]= darray[idx] + d_angulo[l] ;
return;

}

int main(int argc,char** argv)
{

         int N=3200;
         float* darray;
         float h_angulo[360];
         float* d_angulo;

          //asignar memoria en el device
         cudaMalloc ((void**)&darray,sizeof(float)*N);
         cudaMalloc ((void**)&d_angulo,sizeof(float)*360);

         //inicializar array de angulos en el host
       for(int l=0;l<360;l++)
                    h_angulo[l] = acos( -1.0f )* l/ 180.0f;

        //copiar valores del host a la memoria constante 
       cudaMemcpyToSymbol(c_angulo, h_angulo, sizeof(float)*360);

        //**** Invocamos un kernel para "inicializar" el GPU
         //inicializar memoria asignada
        cudaMemset (darray,0,sizeof(float)*N);
        //copiar valores del host a la memoria global
       cudaMemcpy(d_angulo, h_angulo, sizeof(float)*360, cudaMemcpyHostToDevice);
        kernel_inicial  <<<  N/64  ,64  >>>  (darray, d_angulo);
       // *************************************************

         //inicializar memoria asignada
        cudaMemset (darray,0,sizeof(float)*N);
        //copiar valores del host a la memoria global
       cudaMemcpy(d_angulo, h_angulo, sizeof(float)*360, cudaMemcpyHostToDevice);
        kernel_device  <<<  N/64  ,64  >>>  (darray, d_angulo);

         //inicializar memoria asignada (de nuevo)
        cudaMemset (darray,0,sizeof(float)*N);
        kernel_constante  <<<  N/64  ,64  >>>  (darray);
     
       //free device memory
       cudaFree(darray);
  return 0;
}
