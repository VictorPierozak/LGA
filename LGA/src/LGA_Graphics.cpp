#include "..\inc\LGA_Graphics.h"
#include<stdio.h>

const char* readFile(const char* filename) {
    FILE* file = fopen(filename, "rb"); // Open the file in binary mode for reading
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END); // Move the file pointer to the end to get file size
    long fileSize = ftell(file); // Get the file size
    fseek(file, 0, SEEK_SET); // Move the file pointer back to the beginning

    // Allocate memory to hold file contents (+1 for null-terminator)
    char* buffer = (char*)malloc(fileSize + 1);
    if (buffer == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    // Read the file contents into the buffer
    size_t bytesRead = fread(buffer, 1, fileSize, file);
    buffer[bytesRead] = '\0'; // Null-terminate the content

    fclose(file); // Close the file

    return buffer; // Return the content as const char*
}

void createVBO(Graphics_Objects* objs, LGA_Config* config)
{
	glGenBuffers(1, &objs->vbo);
	glBindBuffer(GL_ARRAY_BUFFER, objs->vbo);
    float* vbo = generateDomainRepresentation(config);
	glBufferData(GL_ARRAY_BUFFER, VERTEX_SIZE * config->nx * config->ny*sizeof(float), vbo, GL_STATIC_DRAW);
    free(vbo);
}

void createVAO(Graphics_Objects* objs)
{
	glGenVertexArrays(1, &objs->vao);
	glBindVertexArray(objs->vao);

	glEnableVertexAttribArray(POS_LOCATION);
	glVertexAttribPointer(POS_LOCATION, DIMENSION, GL_FLOAT, GL_FALSE, VERTEX_SIZE * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(COLOR_LOCATION);
	glVertexAttribPointer(COLOR_LOCATION, COLOR_ATTRIB, GL_FLOAT, GL_FALSE, VERTEX_SIZE * sizeof(GLfloat), (void*)(DIMENSION * sizeof(GLfloat)));
}

void compileProgram(Graphics_Objects* objs, const char* vertexShader, const char* fragmentShader)
{
    const char* vertexCode = readFile(vertexShader);
    const char* fragmentCode = readFile(fragmentShader);

    if (vertexCode == NULL || fragmentCode == NULL) {
        // Handle error (failed to read shader files)
        free((void*)vertexCode);
        free((void*)fragmentCode);
        return;
    }

    GLuint vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    // Load shader source code into the shaders
    glShaderSource(vertexShaderID, 1, &vertexCode, NULL);
    glShaderSource(fragmentShaderID, 1, &fragmentCode, NULL);

    // Compile shaders
    glCompileShader(vertexShaderID);
    glCompileShader(fragmentShaderID);

    // Check for shader compilation errors
    GLint success;
    GLchar infoLog[512];

    glGetShaderiv(vertexShaderID, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShaderID, sizeof(infoLog), NULL, infoLog);
        fprintf(stderr, "Vertex shader compilation error: %s\n", infoLog);
    }

    glGetShaderiv(fragmentShaderID, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShaderID, sizeof(infoLog), NULL, infoLog);
        fprintf(stderr, "Fragment shader compilation error: %s\n", infoLog);
    }

    // Create a shader program and attach the compiled shaders
    GLuint programID = glCreateProgram();
    glAttachShader(programID, vertexShaderID);
    glAttachShader(programID, fragmentShaderID);

    // Link the shader program
    glLinkProgram(programID);

    // Check for program linking errors
    glGetProgramiv(programID, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(programID, sizeof(infoLog), NULL, infoLog);
        fprintf(stderr, "Shader program linking error: %s\n", infoLog);
    }

    // Clean up: delete shader objects (no longer needed after linking)
    glDeleteShader(vertexShaderID);
    glDeleteShader(fragmentShaderID);

    // Assign the program ID to the graphics object
    objs->shaderProgram = programID;

    // Free the memory allocated for shader source code
    free((void*)vertexCode);
    free((void*)fragmentCode);

}

Graphics_Objects* createGraphicsObjects(LGA_Config* config)
{
    glewExperimental = GL_TRUE;
    glewInit();

    Graphics_Objects* gobjs = (Graphics_Objects*) malloc(sizeof(Graphics_Objects));
    compileProgram(gobjs, DEFAULT_VERTEX_SHADER, DEFAULT_FRAGMENT_SHADER);
    createVBO(gobjs, config);
    createVAO(gobjs);
    glUseProgram(gobjs->shaderProgram);

    // CUDA ///
    if (cudaGraphicsGLRegisterBuffer(&gobjs->cudaResources, gobjs->vbo, cudaGraphicsMapFlagsWriteDiscard) != cudaSuccess)
    {
        printf("\nResources cannot be registered!\n");
    }

    return gobjs;
}

void mapCudaResources(Graphics_Objects* gobjs)
{
    cudaGraphicsMapResources(1, &gobjs->cudaResources, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&gobjs->devPtr, &gobjs->vboSize, gobjs->cudaResources);
}

void unmapCudaGraphicResources(Graphics_Objects* objs)
{
    cudaGraphicsUnmapResources(1, &objs->cudaResources);
}

void releaseOpenGLResources(Graphics_Objects* objs)
{
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glDeleteBuffers(1, &objs->vbo);
    glDeleteVertexArrays(1, &objs->vao);
}

float* generateDomainRepresentation(LGA_Config* config)
{
    float* vbo = (float*)malloc(sizeof(float) * config->nx * config->ny * VERTEX_SIZE);

    for (unsigned int y = 0; y < config->ny; y++)
        for (unsigned int x = 0; x < config->nx; x++)
        {
            int idx = x + y * config->nx;
            vbo[idx * VERTEX_SIZE] = 1.9*( - 0.5 + ((float)x) / ((float)config->nx) );
            vbo[idx * VERTEX_SIZE + 1] = 1.9*( - 0.5 + ((float)y) / ((float)config->ny));

            if (config->domain_Host[idx].type == WALL) {
                vbo[idx * VERTEX_SIZE + DIMENSION] = 0;
                vbo[idx * VERTEX_SIZE + DIMENSION + 1] = 0;
                vbo[idx * VERTEX_SIZE + DIMENSION + 2] = MAX_INTENSITY;
                continue;
            }

            vbo[idx * VERTEX_SIZE + DIMENSION] = (config->domain_Host[idx].outputState[0] + config->domain_Host[idx].outputState[1] +
                config->domain_Host[idx].outputState[2] + config->domain_Host[idx].outputState[3]) * MAX_INTENSITY * 0.25;
            vbo[idx * VERTEX_SIZE + DIMENSION + 1] = rand() / RAND_MAX;
            vbo[idx * VERTEX_SIZE + DIMENSION + 2] = rand() / RAND_MAX;
        }
    return vbo;
}