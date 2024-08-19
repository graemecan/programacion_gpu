#ifndef __NBODY_H__
#define __NBODY_H__

float damping = 1.0; // Velocity damping (as global)
float deltaTime = 0.016;
float softeningSq = 0.1;

unsigned int blockSize;
unsigned int numBodies = 20224;
unsigned int m_pbo[2];
struct cudaGraphicsResource *m_pGRes[2];

// Global arrays for device data
float4 *dPos[2]; // mapped host pointers
float4 *dVel;

unsigned int m_currentRead, m_currentWrite;

// view params
int ox = 0, oy = 0;
int buttonState = 0;
float camera_trans[]     = {0, -2, -150};
float camera_rot[]       = {0, 0, 0};
float camera_trans_lag[] = {0, -2, -150};
float camera_rot_lag[]   = {0, 0, 0};
const float inertia      = 0.1f;

unsigned int m_vertexShader;
unsigned int m_vertexShaderPoints;
unsigned int m_pixelShader;
unsigned int m_programPoints;
unsigned int m_programSprites;
unsigned int m_texture;
unsigned int m_vboColor;
float m_pointSize = 1.0;
float m_spriteSize = 1.0;

float m_baseColor[4] = { 0.8f, 0.8f, 0.3f, 1.0f};

#define REFRESH_DELAY     10 //ms

enum NBodyConfig {
  BODYSYSTEM_POSITION,
  BODYSYSTEM_VELOCITY
};

const char vertexShaderPoints[] =
{
    "void main()                                                            \n"
    "{                                                                      \n"
    "    vec4 vert = vec4(gl_Vertex.xyz, 1.0);  			                      \n"
    "    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vert;                           \n"
    "    gl_FrontColor = gl_Color;                                          \n"
    "}                                                                      \n"
};


const char vertexShader[] =
{
    "void main()                                                            \n"
    "{                                                                      \n"
    "    float pointSize = 500.0 * gl_Point.size;                           \n"
    "    vec4 vert = gl_Vertex;												\n"
    "    vert.w = 1.0;														\n"
    "    vec3 pos_eye = vec3 (gl_ModelViewMatrix * vert);                   \n"
    "    gl_PointSize = max(1.0, pointSize / (1.0 - pos_eye.z));            \n"
    "    gl_TexCoord[0] = gl_MultiTexCoord0;                                \n"
    //"    gl_TexCoord[1] = gl_MultiTexCoord1;                                \n"
    "    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vert;     \n"
    "    gl_FrontColor = gl_Color;                                          \n"
    "    gl_FrontSecondaryColor = gl_SecondaryColor;                        \n"
    "}                                                                      \n"
};

const char pixelShader[] =
{
    "uniform sampler2D splatTexture;                                        \n"

    "void main()                                                            \n"
    "{                                                                      \n"
    "    vec4 color2 = gl_SecondaryColor;                                   \n"
    "    vec4 color = (0.6 + 0.4 * gl_Color) * texture2D(splatTexture, gl_TexCoord[0].st); \n"
    "    gl_FragColor =                                                     \n"
    "         color * color2;\n"//mix(vec4(0.1, 0.0, 0.0, color.w), color2, color.w);\n"
    "}                                                                      \n"
};

// Renderer functions
float evalHermite(float pA, float pB, float vA, float vB, float u);
unsigned char *createGaussianMap(int N);
void createTexture(int resolution);
// void drawPoints();
// void displayNBodySystem();

// Setup functions
void initialize(); //-> initialises PBOs
void setArray(enum NBodyConfig array, float4 *data); //-> copies Tipsy file data to PBOs
void loadTipsyFile(char *filename); //-> reads data from Tipsy file

// OpenGL functions
void initGL(int *argc, char **argv); //-> initialises OpenGL context
void display(); //-> display callback
void idle(void); //-> idle callback (called whenever something is not happening...)
void finalize(); //-> clears memory when window is closed

// Simulation functions
void integrateNbodySystem(float4 **dPos, float4 *dVel, struct cudaGraphicsResource **pgres, unsigned int currentRead, float deltaTime, float damping, unsigned int numBodies, int blockSize); //-> calls kernel to update particle positions
void updateSimulation(); //-> calls integrateNBodySystem and swaps currentRead and currentWrite values to switch pbo buffers.


#endif //__ NBODY_H__
