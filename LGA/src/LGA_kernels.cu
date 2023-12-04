#pragma once
#include"..\inc\LGA_kernels.cuh"
#include<stdint.h>
#include<glfw3.h>

#ifndef CONSTANT_DEVICE

#define CONSTANT_DEVICE
__constant__ int64_t states = 0b1111110111101100011110100110010010111001010110000011000100100000;

__constant__ unsigned int NX_global = NX;   
__constant__ unsigned int NY_global = NY;

#endif // !CONSTANT_DEVICE


__global__ void LGA_K_Input(Field* domain_D)
{

	int idx = threadIdx.x + blockDim.x * blockIdx.x + 
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y );

	if (domain_D[idx].type == WALL)
	{
		return;
	}

	int state = 4*(domain_D[idx].inputState[0] * 8
		+ domain_D[idx].inputState[1] * 4
		+ domain_D[idx].inputState[2] * 2
		+ domain_D[idx].inputState[3]);
	domain_D[idx].outputState[0] = (states >> state) & 1;
	domain_D[idx].outputState[1] = (states >> (state + 1)) & 1; 
	domain_D[idx].outputState[2] = (states >> (state + 2)) & 1;
	domain_D[idx].outputState[3] = (states >> (state + 3)) & 1;
}

__global__ void LGA_K_Output(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x + 
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);

	if (domain_D[idx].type == WALL) {
		return;
	}

	domain_D[idx].inputState[0] = domain_D[idx - 1].outputState[2] + domain_D[idx].outputState[0] * (domain_D[idx - 1].type == WALL);;
	domain_D[idx].inputState[2] = domain_D[idx + 1].outputState[0] + domain_D[idx].outputState[2] * (domain_D[idx + 1].type == WALL);;
	domain_D[idx].inputState[1] = domain_D[idx - NX_global].outputState[3] + domain_D[idx].outputState[1] * (domain_D[idx - NX_global].type == WALL);
	domain_D[idx].inputState[3] = domain_D[idx + NX_global].outputState[1] + domain_D[idx].outputState[3] * (domain_D[idx + NX_global].type == WALL);
}

void LGA_run(LGA_Config* configuration)
{
	LGA_K_Input <<< configuration->gridSize, configuration->blockSize >>> (configuration->domain_Device);
	cudaDeviceSynchronize();
	LGA_K_Output <<< configuration->gridSize, configuration->blockSize >>> (configuration->domain_Device);
	cudaDeviceSynchronize();
}

__global__ void LGA_K_Draw(Field* domain_D, float* vbo)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);

	if (domain_D[idx].type == WALL) {
		vbo[idx * VERTEX_SIZE + DIMENSION] = 0;
		vbo[idx * VERTEX_SIZE + DIMENSION + 1] = 0;
		vbo[idx * VERTEX_SIZE + DIMENSION + 2] = MAX_INTENSITY;
		return;
	}

	vbo[idx * VERTEX_SIZE + DIMENSION] = (domain_D[idx].outputState[0] + domain_D[idx].outputState[1] +
		domain_D[idx].outputState[2] + domain_D[idx].outputState[3]) * MAX_INTENSITY * 0.25;
	vbo[idx * VERTEX_SIZE + DIMENSION + 1] = 0;//(domain_D[idx].outputState[1] + domain_D[idx].outputState[3]) * 0.25;
	vbo[idx * VERTEX_SIZE + DIMENSION + 2] = 0;// (domain_D[idx].outputState[0] + domain_D[idx].outputState[2]) * 0.25;
}

void LGA_draw(LGA_Config* configuration, float* devPtr)
{
	//NX_global = configuration->nx;
	//NY_global = configuration->ny;
	LGA_K_Draw <<< configuration->gridSize, configuration->blockSize >>> (configuration->domain_Device, devPtr);
	cudaDeviceSynchronize();
}

