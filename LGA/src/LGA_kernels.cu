#pragma once
#include"..\inc\LGA_kernels.cuh"
#include<stdint.h>
#include<glfw3.h>

//__constant__ int64_t states = 0b1111110111101100011110100110010010111001010110000011000100100000;

__constant__ unsigned int NX_global = NX;
__constant__ unsigned int NY_global = NY;
__constant__ float Time_coef = 1.0f;


__global__ void LGA_K_Input(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);

	if (domain_D[idx].type == WALL)
	{
		return;
	}
	// ZA£O¯ENIE - MA£A LICZBA KOMÓREK GRANICZY ZE ŒCIANAMI //

	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];

	if (domain_D[idx - 1].type == WALL) domain_D[idx].inStreams[1] = domain_D[idx].outStreams[2];
	else domain_D[idx].inStreams[1] = domain_D[idx - 1].outStreams[1];

	if (domain_D[idx + 1].type == WALL) domain_D[idx].inStreams[2] = domain_D[idx].outStreams[1];
	else domain_D[idx].inStreams[2] = domain_D[idx + 1].outStreams[2];

	if (domain_D[idx + NX_global].type == WALL) domain_D[idx].inStreams[3] = domain_D[idx].outStreams[4];
	else domain_D[idx].inStreams[3] = domain_D[idx + NX_global].outStreams[3];

	if (domain_D[idx - NX_global].type == WALL) domain_D[idx].inStreams[4] = domain_D[idx].outStreams[3];
	else domain_D[idx].inStreams[4] = domain_D[idx - NX_global].outStreams[4];

	if (domain_D[idx + NX_global - 1].type == WALL) domain_D[idx].inStreams[5] = domain_D[idx].outStreams[7];
	else domain_D[idx].inStreams[5] = domain_D[idx + NX_global - 1].outStreams[5];

	if (domain_D[idx + NX_global + 1].type == WALL) domain_D[idx].inStreams[6] = domain_D[idx].outStreams[8];
	else domain_D[idx].inStreams[6] = domain_D[idx + NX_global + 1].outStreams[6];

	if (domain_D[idx - NX_global + 1].type == WALL) domain_D[idx].inStreams[7] = domain_D[idx].outStreams[5];
	else domain_D[idx].inStreams[7] = domain_D[idx - NX_global + 1].outStreams[7];

	if (domain_D[idx - NX_global - 1].type == WALL) domain_D[idx].inStreams[8] = domain_D[idx].outStreams[6];
	else domain_D[idx].inStreams[8] = domain_D[idx - NX_global - 1].outStreams[8];

	domain_D[idx].ro = 0;

	for (int i = 0; i < 9; i += 3)
	{
		domain_D[idx].ro += domain_D[idx].inStreams[i];
		domain_D[idx].ro += domain_D[idx].inStreams[i + 1];
		domain_D[idx].ro += domain_D[idx].inStreams[i + 2];
	}

	float ro_rev =  (domain_D[idx].ro != 0) ? (1.0f / domain_D[idx].ro) : 0.0f;
	
	domain_D[idx].u[0] = (domain_D[idx].inStreams[1] + domain_D[idx].inStreams[5] + domain_D[idx].inStreams[8] -
		domain_D[idx].inStreams[2] - domain_D[idx].inStreams[6] - domain_D[idx].inStreams[7])* ro_rev;

	domain_D[idx].u[1] = (domain_D[idx].inStreams[3] + domain_D[idx].inStreams[5] + domain_D[idx].inStreams[6] -
		domain_D[idx].inStreams[4] - domain_D[idx].inStreams[7] - domain_D[idx].inStreams[8]) * ro_rev;
}

__global__ void LGA_K_Output(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);

	if (domain_D[idx].type == WALL) {
		return;
	}

	for (int i = 0; i < 9; i += 3)
	{
		domain_D[idx].outStreams[i] = domain_D[idx].inStreams[i] + Time_coef * (
			domain_D[idx].eqStreams[i] - domain_D[idx].inStreams[i]);

		domain_D[idx].outStreams[i + 1] = domain_D[idx].inStreams[i + 1] + Time_coef * (
			domain_D[idx].eqStreams[i + 1] - domain_D[idx].inStreams[i + 1]);

		domain_D[idx].outStreams[i + 2] = domain_D[idx].inStreams[i + 2] + Time_coef * (
			domain_D[idx].eqStreams[i + 2] - domain_D[idx].inStreams[i + 2]);
	}

}

void LGA_run(LGA_Config* configuration)
{
	LGA_K_Equalibrium << < configuration->gridSize, configuration->blockSize >> > (configuration->domain_Device);
	cudaDeviceSynchronize();
	LGA_K_Output <<< configuration->gridSize, configuration->blockSize >> > (configuration->domain_Device);
	cudaDeviceSynchronize();
	LGA_K_Input << < configuration->gridSize, configuration->blockSize >> > (configuration->domain_Device);
	cudaDeviceSynchronize();
}

void LGA_init(LGA_Config* configuration)
{
	LGA_K_Equalibrium_Init << < configuration->gridSize, configuration->blockSize >> > (configuration->domain_Device);
	cudaDeviceSynchronize();
	LGA_K_Output << < configuration->gridSize, configuration->blockSize >> > (configuration->domain_Device);
	cudaDeviceSynchronize();
	LGA_K_Input << < configuration->gridSize, configuration->blockSize >> > (configuration->domain_Device);
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

	vbo[idx * VERTEX_SIZE + DIMENSION] = (domain_D[idx].ro)* MAX_INTENSITY;
	vbo[idx * VERTEX_SIZE + DIMENSION + 1] = (domain_D[idx].ro < 0)*  MAX_INTENSITY;
	vbo[idx * VERTEX_SIZE + DIMENSION + 2] = 0; 
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
}

__global__ void LGA_K_Equalibrium(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);
	if (domain_D[idx].type == WALL) {
		return;
	}

	float u_square =  1.5*(domain_D[idx].u[0] * domain_D[idx].u[0] + domain_D[idx].u[1] * domain_D[idx].u[1]);
	domain_D[idx].eqStreams[0] = (4.0f/9.0f)*domain_D[idx].ro*(1.0f - u_square);

	domain_D[idx].eqStreams[1] = (1.0f / 9.0f) * domain_D[idx].ro*(1.0f + 3.0f * domain_D[idx].u[0] + 4.5* domain_D[idx].u[0]* domain_D[idx].u[0] - u_square);
	domain_D[idx].eqStreams[2] = (1.0f/9.0f) * domain_D[idx].ro*(1.0f - 3.0f * domain_D[idx].u[0] + 4.5 * domain_D[idx].u[0] * domain_D[idx].u[0] - u_square);
	domain_D[idx].eqStreams[3] = (1.0f/9.0f) * domain_D[idx].ro * (1.0f + 3.0f * domain_D[idx].u[1] + 4.5 * domain_D[idx].u[1] * domain_D[idx].u[1] - u_square);
	domain_D[idx].eqStreams[4] = (1.0f/9.0f) * domain_D[idx].ro * (1.0f - 3.0f * domain_D[idx].u[1] + 4.5 * domain_D[idx].u[1] * domain_D[idx].u[1] - u_square);
	
	domain_D[idx].eqStreams[5] = (1.0f / 36.0f) * domain_D[idx].ro * (1.0f + 3.0f * (domain_D[idx].u[0] + domain_D[idx].u[1]) + 
		4.5 * (domain_D[idx].u[0] + domain_D[idx].u[1]) * (domain_D[idx].u[0] + domain_D[idx].u[1]) - u_square);

	domain_D[idx].eqStreams[6] = (1.0f / 36.0f) * domain_D[idx].ro * (1.0f + 3.0f * (-domain_D[idx].u[0] + domain_D[idx].u[1]) +
		4.5 * (-domain_D[idx].u[0] + domain_D[idx].u[1]) * (-domain_D[idx].u[0] + domain_D[idx].u[1]) - u_square);

	domain_D[idx].eqStreams[7] = (1.0f / 36.0f) * domain_D[idx].ro * (1.0f + 3.0f * (-domain_D[idx].u[0] - domain_D[idx].u[1]) +
		4.5 * (-domain_D[idx].u[0] - domain_D[idx].u[1]) * (-domain_D[idx].u[0] - domain_D[idx].u[1]) - u_square);

	domain_D[idx].eqStreams[8] = (1.0f / 36.0f) * domain_D[idx].ro * (1.0f + 3.0f * (domain_D[idx].u[0] - domain_D[idx].u[1]) +
		4.5 * (domain_D[idx].u[0] - domain_D[idx].u[1]) * (domain_D[idx].u[0] - domain_D[idx].u[1]) - u_square);
}

__global__ void LGA_K_Equalibrium_Init(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);
	if (domain_D[idx].type == WALL) {
		return;
	}

	float u_square = 1.5 * (domain_D[idx].u[0] * domain_D[idx].u[0] + domain_D[idx].u[1] * domain_D[idx].u[1]);
	domain_D[idx].eqStreams[0] = (4.0f / 9.0f) * domain_D[idx].ro;

	domain_D[idx].eqStreams[1] = 1.0f / 9.0f * domain_D[idx].ro ;
	domain_D[idx].eqStreams[2] = 1.0f / 9.0f * domain_D[idx].ro ;
	domain_D[idx].eqStreams[3] = 1.0f / 9.0f * domain_D[idx].ro ;
	domain_D[idx].eqStreams[4] = 1.0f / 9.0f * domain_D[idx].ro ;

	domain_D[idx].eqStreams[5] = 1.0f / 36.0f * domain_D[idx].ro ;

	domain_D[idx].eqStreams[6] = 1.0f / 36.0f * domain_D[idx].ro ;

	domain_D[idx].eqStreams[7] = 1.0f / 36.0f * domain_D[idx].ro;

	domain_D[idx].eqStreams[8] = 1.0f / 36.0f * domain_D[idx].ro ;
}