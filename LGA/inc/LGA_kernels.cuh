#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"..\inc\LGA_Configuration.h"

__global__ void LGA_K_Input(Field* domain_D);
__global__ void LGA_K_Output(Field* domain_D);
__global__ void LGA_K_Draw(Field* domain_D, float* vbo, Visualisation* visualisation_D);
__global__ void LGA_K_Draw_Density(Field* domain_D, float* vbo, Visualisation* visualisation_D);
__global__ void LGA_K_Draw_Velocity_Norm(Field* domain_D, float* vbo, Visualisation* visualisation_D);
__global__ void LGA_K_Draw_Velocity_Horizontal(Field* domain_D, float* vbo, Visualisation* visualisation_D);
__global__ void LGA_K_Draw_Velocity_Vertical(Field* domain_D, float* vbo, Visualisation* visualisation_D);
__global__ void LGA_K_Equalibrium(Field* domain_D);

void setConstantMemory(LGA_Config* configuration);
void LGA_run(LGA_Config* configuration);
void LGA_draw(LGA_Config* configuration, float* devPtr);

