#include <stdio.h> 
  
// estructura 1
struct prueba1  
{ 
    short s; // 2 bytes + 2 padding
    int i;   // 4 bytes
    char c;  // 1 byte + 3 padding
}; 
  
// estructura 2
struct prueba2
{ 
    int i; // 4 bytes
    char c; // 1 byte
    short s; // 2 bytes + 1 padding
}; 
  
// driver program 
int main() 
{ 
    struct prueba1 t1; 
    struct prueba2 t2; 
    printf("Tamaño del struct prueba1 es %lu\n",sizeof(t1)); 
    printf("Tamaño del struct prueba2 es %lu\n",sizeof(t2));
    return 0; 
} 

