#pragma once
#include<GL\glew.h>
#include<cuda_gl_interop.h>
#include "LGA_Configuration.h"

typedef struct
{
	GLuint vao;
	GLuint vbo;
	GLuint ebo;
	GLuint shaderProgram;
	cudaGraphicsResource_t cudaResources;
	GLfloat* devPtr;
	size_t vboSize;
}  Graphics_Objects;

void createVBO(Graphics_Objects* objs, LBM_Config* config);
void createVAO(Graphics_Objects* objs);
void compileProgram(Graphics_Objects* objs, const char* vertexShader, const char* fragmentShader);
Graphics_Objects* createGraphicsObjects(LBM_Config* config);
void mapCudaResources(Graphics_Objects* gobjs);
void unmapCudaGraphicResources(Graphics_Objects* objs);
void releaseOpenGLResources(Graphics_Objects* objs);


float* generateDomainRepresentation(LBM_Config* config);