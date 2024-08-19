#include <string.h>
#include <math.h>

#define GL_GLEXT_PROTOTYPES
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "nbody.h"
#include "tipsy.h"

//==========================================================================================================

void checkGLError()
{
    GLenum err;
    while((err = glGetError()) != GL_NO_ERROR){
        printf("GLerror %d\n",err);
    }  
}

void checkCUDAerror(){
    // check if kernel invocation generated an error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}

__constant__ float softeningSquared;

__device__ float3 bodyBodyInteraction(float3 ai, float4 bi, float4 bj)
{
    float3 r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += softeningSquared;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float invDist = rsqrtf(distSqr);
    float invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

__device__ float3 computeBodyAccel(float4 bodyPos, float4 *positions, int numTiles)
{
    extern __shared__ float4 sharedPos[];

    float3 acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++)
    {
        sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

        __syncthreads(); // This was a cooperative groups sync

        // This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128

        for (unsigned int counter = 0; counter < blockDim.x; counter++)
        {
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[counter]);
        }

        __syncthreads(); // Again, this was a cg sync
    }

    return acc;
}

// __restrict__ is a C keyword to deal with pointer aliasing
// This is when two pointers point to overlapping memory spaces,
// so the compiler doesn't know if it can reuse loaded data or if it has to
// reload because of modification. Using __restrict__ informs the compiler
// that there is no overlap, thus improving performance.
__global__ void integrateBodies(float4* __restrict__ newPos,
                float4* __restrict__ oldPos,
                float4* vel, float deltaTime, float damping, int numTiles)
{
    // Removing cooperative groups stuff, since the synchronization apparently
    // happens at block level anyway
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // deviceOffset not required for single device
    float4 position = oldPos[index];

    float3 accel = computeBodyAccel(position, oldPos, numTiles);

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in bodyBodyInteraction
    // (because they cancel out).  Thus here force == acceleration
    float4 velocity = vel[index];

    velocity.x += accel.x * deltaTime;
    velocity.y += accel.y * deltaTime;
    velocity.z += accel.z * deltaTime;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * deltaTime
    position.x += velocity.x * deltaTime;
    position.y += velocity.y * deltaTime;
    position.z += velocity.z * deltaTime;

    // store new position and velocity
    newPos[index] = position;
    vel[index]    = velocity;
}

//===============================================================================================================


void drawPoints(){

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, m_pbo[m_currentRead]);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, m_vboColor);
    glColorPointer(4, GL_FLOAT, 0, 0);

    glDrawArrays(GL_POINTS, 0, numBodies);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

}

void displayNBodySystem(){

    // FOR SPRITES:
    glEnable(GL_POINT_SPRITE_ARB);
    glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
    glPointSize(m_spriteSize);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);

    glUseProgram(m_programSprites);
    GLuint texLoc = glGetUniformLocation(m_programSprites, "splatTexture");
    glUniform1i(texLoc, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture);

    glColor3f(1, 1, 1);
    glSecondaryColor3fv(m_baseColor);

    drawPoints();

    glUseProgram(0);

    glDisable(GL_POINT_SPRITE_ARB);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
}

inline float evalHermite(float pA, float pB, float vA, float vB, float u)
{
    float u2=(u*u), u3=u2*u;
    float B0 = 2*u3 - 3*u2 + 1;
    float B1 = -2*u3 + 3*u2;
    float B2 = u3 - 2*u2 + u;
    float B3 = u3 - u;
    return (B0*pA + B1*pB + B2*vA + B3*vB);
}

void initialize()
{

    unsigned int memSize = sizeof(float) * 4 * numBodies;
    glGenBuffers(2, (GLuint *)m_pbo);

    for (int i = 0; i < 2; ++i)
    {
        glBindBuffer(GL_ARRAY_BUFFER, m_pbo[i]);
        glBufferData(GL_ARRAY_BUFFER, memSize, 0, GL_DYNAMIC_DRAW);

        int size = 0;
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint *)&size);
        if ((unsigned)size != memSize)
        {
            printf("%u %u\n",size,memSize);
            fprintf(stderr, "WARNING: Pixel Buffer Object allocation failed!\n");
        }

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        cudaGraphicsGLRegisterBuffer(&m_pGRes[i], m_pbo[i], cudaGraphicsMapFlagsNone);
        checkCUDAerror();
    }

    cudaMalloc((void **)&dVel, memSize);
    checkCUDAerror();
    cudaMemcpyToSymbol(softeningSquared, &softeningSq, sizeof(float), 0, cudaMemcpyHostToDevice);
    checkCUDAerror();

    checkGLError();
}

void setArray(enum NBodyConfig array, float4 *data)
{

    m_currentRead = 0;
    m_currentWrite = 1;

    switch (array)
    {
        default:
        case BODYSYSTEM_POSITION:
            {
                glBindBuffer(GL_ARRAY_BUFFER, m_pbo[m_currentRead]);
                glBufferSubData(GL_ARRAY_BUFFER, 0, 4 * sizeof(float) * numBodies, data);

                int size = 0;
                glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint *)&size);

                if ((unsigned)size != 4 * (sizeof(float) * numBodies))
                {
                   fprintf(stderr, "WARNING: Pixel Buffer Object download failed!n");
                }

                glBindBuffer(GL_ARRAY_BUFFER, 0);
            }
            break;

        case BODYSYSTEM_VELOCITY:
                cudaMemcpy(dVel, data, numBodies * 4 * sizeof(float), cudaMemcpyHostToDevice);
            break;
    }
}

void loadTipsyFile(char *filename){

    float4 positions[numBodies];
    float4 velocities[numBodies];

    read_tipsy_file(positions,
                    velocities,
                    filename, numBodies);

    initialize();

    // Tipsy data is now transferred to device, stored in PBOs (OpenGL)
    setArray(BODYSYSTEM_POSITION, positions);
    setArray(BODYSYSTEM_VELOCITY, velocities);
}

unsigned char *createGaussianMap(int N)
{
    float *M = (float *)malloc(sizeof(float)*2*N*N);
    unsigned char *B = (unsigned char *)malloc(sizeof(char)*4*N*N);
    float X,Y,Y2,Dist;
    float Incr = 2.0f/N;
    int i=0;
    int j = 0;
    Y = -1.0f;

    //float mmax = 0;
    for (int y=0; y<N; y++, Y+=Incr)
    {
        Y2=Y*Y;
        X = -1.0f;

        for (int x=0; x<N; x++, X+=Incr, i+=2, j+=4)
        {
            Dist = (float)sqrtf(X*X+Y2); // Presumably this slows things down a lot??

            if (Dist>1) Dist=1;

            M[i+1] = M[i] = evalHermite(1.0f,0.0,0.0,0.0,Dist);
            B[j+3] = B[j+2] = B[j+1] = B[j] = (unsigned char)(M[i] * 255);
        }
    }

    free(M);
    return (B);
}

void createTexture(int resolution)
{
    unsigned char *data = createGaussianMap(resolution);
    glGenTextures(1, (GLuint *)&m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution, resolution, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data);

}

void initGL(int *argc, char **argv)
{
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(720, 480);
    glutCreateWindow("CUDA n-body system");
    // It seems including this function is not trivial because of how the OpenGL libraries
    // work. So hopefully it's not too important!
    // glxSwapIntervalSGI(0);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);

    m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
    m_vertexShaderPoints = glCreateShader(GL_VERTEX_SHADER);
    m_pixelShader = glCreateShader(GL_FRAGMENT_SHADER);

    const char *v = vertexShader;
    const char *p = pixelShader;
    glShaderSource(m_vertexShader, 1, &v, 0);
    glShaderSource(m_pixelShader, 1, &p, 0);
    const char *vp = vertexShaderPoints;
    glShaderSource(m_vertexShaderPoints, 1, &vp, 0);

    glCompileShader(m_vertexShader);
    glCompileShader(m_vertexShaderPoints);
    glCompileShader(m_pixelShader);

    m_programSprites = glCreateProgram();
    glAttachShader(m_programSprites, m_vertexShader);
    glAttachShader(m_programSprites, m_pixelShader);
    glLinkProgram(m_programSprites);

    m_programPoints = glCreateProgram();
    glAttachShader(m_programPoints, m_vertexShaderPoints);
    glLinkProgram(m_programPoints);

    checkGLError();

    createTexture(32);

    glGenBuffers(1, (GLuint *)&m_vboColor);
    glBindBuffer(GL_ARRAY_BUFFER, m_vboColor);
    glBufferData(GL_ARRAY_BUFFER, numBodies * 4 * sizeof(float), 0, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void display()
{
    updateSimulation();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    for (int c = 0; c < 3; ++c)
    {
        // globals that are used to slightly slow the camera movement when rotating??
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);
    

    displayNBodySystem(); // -> calls the "display" function of the NBodySystem class...
                          // -> which then calls setSpriteSize, setPBO, display in the renderer.

    glutSwapBuffers();
    glutReportErrors();
}

void integrateNbodySystem(float4 **dPos, float4 *dVel, struct cudaGraphicsResource **pgres, unsigned int currentRead, float deltaTime, float damping, unsigned int numBodies, int blockSize)
{
    cudaGraphicsResourceSetMapFlags(pgres[currentRead], cudaGraphicsMapFlagsReadOnly);
    cudaGraphicsResourceSetMapFlags(pgres[1-currentRead], cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsMapResources(2, pgres, 0);
    size_t bytes;

    cudaGraphicsResourceGetMappedPointer((void **)&dPos[currentRead], &bytes, pgres[currentRead]);
    cudaGraphicsResourceGetMappedPointer((void **)&dPos[1-currentRead], &bytes, pgres[1-currentRead]);

    int numBlocks = (numBodies + blockSize-1) / blockSize;
    int numTiles = (numBodies + blockSize - 1) / blockSize;
    int sharedMemSize = blockSize * 4 * sizeof(float); // 4 floats for pos

    integrateBodies<<< numBlocks, blockSize, sharedMemSize >>>(dPos[1-currentRead], dPos[currentRead], dVel, deltaTime, damping, numTiles);
    cudaDeviceSynchronize();

    checkCUDAerror();

    cudaGraphicsUnmapResources(2, pgres, 0);

    checkCUDAerror();
}

// This was a method in the BodySystem class, called from updateSimulation.
// Presumably I'll need to set the timestep deltaTime as a global (0.016 is used in the sim...)
void updateSimulation()
{

    int temp;
    // where the magic happens!
    // the m_currentRead, m_currentWrite flags (pointers?) are used to control which of the two
    // PBO buffers we're using to write data and which to display the model

    // This function is within bodysystemcuda.cu, NOT the header file!
    integrateNbodySystem(dPos, dVel, m_pGRes, m_currentRead, deltaTime, damping, numBodies, blockSize);

    temp = m_currentRead;
    m_currentRead = m_currentWrite;
    m_currentWrite = temp;
}

void idle(void)
{
    glutPostRedisplay();
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 1000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

void finalize(){

  cudaGraphicsUnregisterResource(m_pGRes[0]);
  cudaGraphicsUnregisterResource(m_pGRes[1]);
  glDeleteBuffers(2, (const GLuint *)m_pbo);

}

void mouse(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN)
    {
        buttonState |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT)
    {
        buttonState = 2;
    }
    else if (mods & GLUT_ACTIVE_CTRL)
    {
        buttonState = 3;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx = (float)(x - ox);
    float dy = (float)(y - oy);

    if (buttonState == 3)
    {
        // left+middle = zoom
        camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
    }
    else if (buttonState & 2)
    {
        // middle = translate
        camera_trans[0] += dx / 100.0f;
        camera_trans[1] -= dy / 100.0f;
    }
    else if (buttonState & 1)
    {
        // left = rotate
        camera_rot[0] += dy / 5.0f;
        camera_rot[1] += dx / 5.0f;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

int main(int argc, char **argv) {

    setenv ("DISPLAY", ":0", 0);
    // Need to initialise OpenGL before we can do all the buffers and shit
    initGL(&argc, argv);
    blockSize = 256;

    char fname[15] = "galaxy_20K.bin";

    loadTipsyFile(fname);

    // updateSimulation();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    //glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    glutIdleFunc(idle);

    glutMainLoop();

    finalize();
}
