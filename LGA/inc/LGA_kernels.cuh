#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"..\inc\LGA_Configuration.h"

__global__ void LBM_K_Collsion(Field* domain_D);
__global__ void LBM_K_Streaming(Field* domain_D);
__global__ void LBM_K_Draw(Field* domain_D, float* vbo, Visualisation* visualisation_D);
__global__ void LBM_K_Draw_Density(Field* domain_D, float* vbo, Visualisation* visualisation_D);
__global__ void LBM_K_Draw_Velocity_Norm(Field* domain_D, float* vbo, Visualisation* visualisation_D);
__global__ void LBM_K_Draw_Velocity_Horizontal(Field* domain_D, float* vbo, Visualisation* visualisation_D);
__global__ void LBM_K_Draw_Velocity_Vertical(Field* domain_D, float* vbo, Visualisation* visualisation_D);
__global__ void LBM_K_Equalibrium(Field* domain_D);

__global__ void LBM_K_Boundry_N(Field* domain_D);
__global__ void LBM_K_Boundry_S(Field* domain_D);
__global__ void LBM_K_Boundry_W(Field* domain_D);
__global__ void LBM_K_Boundry_E(Field* domain_D);

__global__ void LBM_K_Boundry_NE(Field* domain_D);
__global__ void LBM_K_Boundry_NW(Field* domain_D);
__global__ void LBM_K_Boundry_SE(Field* domain_D);
__global__ void LBM_K_Boundry_SW(Field* domain_D);


void setConstantMemory(LBM_Config* configuration);
void LBM_run(LBM_Config* configuration);
void LBM_draw(LBM_Config* configuration, float* devPtr);

