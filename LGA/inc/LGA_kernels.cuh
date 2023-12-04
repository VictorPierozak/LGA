#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"..\inc\LGA_Configuration.h"

__global__ void LGA_K_Input(Field* domain_D);
__global__ void LGA_K_Output(Field* domain_D);
__global__ void LGA_K_Draw(Field* domain_D, float* vbo);

void LGA_run(LGA_Config* configuration);
void LGA_draw(LGA_Config* configuration, float* devPtr);
//void setGlobalNX(unsigned int nx)
//{
//	NX_global = nx;
//}
//void setGlobalNY(unsigned int ny)
//{
//	NY_global = ny;
//}
