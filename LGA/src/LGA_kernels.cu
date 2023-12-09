#pragma once
#include"..\inc\LGA_kernels.cuh"
#include<stdint.h>
#include<glfw3.h>

//__constant__ int64_t states = 0b1111110111101100011110100110010010111001010110000011000100100000;

__constant__ unsigned int NX_global = NX;
__constant__ unsigned int NY_global = NY;
__constant__ float Time_coef = 1.0;
__constant__ float Eq_coef_Q1 = 0.25;
__constant__ float Eq_coef_Q2 = 0.25;
__constant__ float Eq_coef_Q3 = 0.25;
__constant__ float Eq_coef_Q4 = 0.25;




__global__ void LGA_K_Input(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);

	if (domain_D[idx].type == WALL)
	{
		return;
	}
	// ZA£O¯ENIE - MA£A LICZBA KOMÓREK GRANICZY ZE ŒCIANAMI //

	if (domain_D[idx + 1].type == WALL) domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];
	else domain_D[idx].inStreams[0] = domain_D[idx + 1].outStreams[2];

	if (domain_D[idx + NX_global].type == WALL) domain_D[idx].inStreams[1] = domain_D[idx].outStreams[1];
	else domain_D[idx].inStreams[1] = domain_D[idx + NX_global].outStreams[3];

	if (domain_D[idx - 1].type == WALL) domain_D[idx].inStreams[2] = domain_D[idx].outStreams[2];
	else domain_D[idx].inStreams[2] = domain_D[idx - 1].outStreams[0];

	if (domain_D[idx - NX_global].type == WALL) domain_D[idx].inStreams[3] = domain_D[idx].outStreams[3];
	else domain_D[idx].inStreams[3] = domain_D[idx - NX_global].outStreams[1];

	domain_D[idx].C = domain_D[idx].inStreams[0] + domain_D[idx].inStreams[1] + domain_D[idx].inStreams[2] +
		domain_D[idx].inStreams[3];
}

__global__ void LGA_K_Output(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);

	if (domain_D[idx].type == WALL) {
		return;
	}

	domain_D[idx].outStreams[2] = domain_D[idx].inStreams[0] + Time_coef * (
		domain_D[idx].C * Eq_coef_Q1 - domain_D[idx].inStreams[0]);
	domain_D[idx].outStreams[3] = domain_D[idx].inStreams[1] + Time_coef * (
		domain_D[idx].C * Eq_coef_Q2 - domain_D[idx].inStreams[1]);
	domain_D[idx].outStreams[0] = domain_D[idx].inStreams[2] + Time_coef * (
		domain_D[idx].C * Eq_coef_Q3 - domain_D[idx].inStreams[2]);
	domain_D[idx].outStreams[1] = domain_D[idx].inStreams[3] + Time_coef * (
		domain_D[idx].C * Eq_coef_Q4 - domain_D[idx].inStreams[3]);
}

void LGA_run(LGA_Config* configuration)
{
	LGA_K_Output <<< configuration->gridSize, configuration->blockSize >> > (configuration->domain_Device);
	cudaDeviceSynchronize();
	LGA_K_Input <<< configuration->gridSize, configuration->blockSize >>> (configuration->domain_Device);
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

	vbo[idx * VERTEX_SIZE + DIMENSION] = domain_D[idx].C * MAX_INTENSITY;
	vbo[idx * VERTEX_SIZE + DIMENSION + 1] = domain_D[idx].C * MAX_INTENSITY;
	vbo[idx * VERTEX_SIZE + DIMENSION + 2] = domain_D[idx].C * MAX_INTENSITY;
}

void LGA_draw(LGA_Config* configuration, float* devPtr)
{
	LGA_K_Draw <<< configuration->gridSize, configuration->blockSize >>> (configuration->domain_Device, devPtr);
	cudaDeviceSynchronize();
}

void setConstantMemory(LGA_Config* config)
{
	cudaMemcpyToSymbol("NX_global", &config->nx, sizeof(unsigned int));
	cudaMemcpyToSymbol("NY_global", &config->ny, sizeof(unsigned int));
	float time_coef = config->simulationData.dt / config->simulationData.tau;
	cudaMemcpyToSymbol("Time_coef", &time_coef, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("Eq_coef_Q1", &config->simulationData.equalibriumStreams[0], sizeof(float), 0);
	cudaMemcpyToSymbol("Eq_coef_Q2", &config->simulationData.equalibriumStreams[1], sizeof(float), 0);
	cudaMemcpyToSymbol("Eq_coef_Q3", &config->simulationData.equalibriumStreams[2], sizeof(float), 0);
	cudaMemcpyToSymbol("Eq_coef_Q4", &config->simulationData.equalibriumStreams[3], sizeof(float), 0);
}