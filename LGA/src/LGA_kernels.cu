#pragma once
#include"..\inc\LGA_kernels.cuh"
#include<stdint.h>
#include<glfw3.h>

#define C_1 2.7632
#define C_2 3.8177
#define C_3 1.3816

#define FIELD 1

//__constant__ int64_t states = 0b1111110111101100011110100110010010111001010110000011000100100000;

__constant__ unsigned int NX_global = NX;
__constant__ unsigned int NY_global = NY;
__constant__ float Time_coef = 1.0f;
__constant__ float COEF_1;
__constant__ float COEF_2;
__constant__ float COEF_3;


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
	//domain_D[idx].ro = domain_D[idx].ro * (domain_D[idx].ro > 0);

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

__global__ void LGA_K_Draw(Field* domain_D, float* vbo, Visualisation* visualisation_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);
	for (int v = 0; v < 4; v++)
	{
		if (domain_D[idx].type == WALL) {
			vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION] = 0;
			vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 1] = 0;
			vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 2] = MAX_INTENSITY;
			continue;
		}

		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 2] = visualisation_D->amplifier * domain_D[idx].ro;
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 1] = visualisation_D->amplifier * domain_D[idx].ro;
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION] = visualisation_D->amplifier * domain_D[idx].ro;
	}
}

__global__ void LGA_K_Draw_Density(Field* domain_D, float* vbo, Visualisation* visualisation_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);

	for (int v = 0; v < 4; v++)
	{
		if (domain_D[idx].type == WALL) {
			vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION] = 0;
			vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 1] = 0;
			vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 2] = MAX_INTENSITY;
			continue;
		}

		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 2] = visualisation_D->amplifier * domain_D[idx].ro;
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 1] = visualisation_D->amplifier * domain_D[idx].ro;
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION] = visualisation_D->amplifier * domain_D[idx].ro;
	}
}

__global__ void LGA_K_Draw_Velocity_Norm(Field* domain_D, float* vbo, Visualisation* visualisation_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);
	
	for (int v = 0; v < 4; v++)
	{
		if (domain_D[idx].type == WALL) {
			vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION] = 0;
			vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 1] = 0;
			vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 2] = MAX_INTENSITY;
			continue;
		}
		float velocity = sqrt(domain_D[idx].u[0] * domain_D[idx].u[0] + (domain_D[idx].u[1] * domain_D[idx].u[1]));

		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 2] = visualisation_D->amplifier * velocity;
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 1] = visualisation_D->amplifier * velocity;
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION] = visualisation_D->amplifier * velocity;
	}
}

__global__ void LGA_K_Draw_Velocity_Horizontal(Field* domain_D, float* vbo, Visualisation* visualisation_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);

	for (int v = 0; v < 4; v++)
	{

	if (domain_D[idx].type == WALL) {
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION] = 0;
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 1] = 0;
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 2] = MAX_INTENSITY;
		continue;
	}
	
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 2] = 0;
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 1] = visualisation_D->amplifier * abs(domain_D[idx].u[0]) * (domain_D[idx].u[0] > 0);
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION] = visualisation_D->amplifier * abs(domain_D[idx].u[0]) * (domain_D[idx].u[0] < 0);
	}
}

__global__ void LGA_K_Draw_Velocity_Vertical(Field* domain_D, float* vbo, Visualisation* visualisation_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);

	for (int v = 0; v < 4; v++)
	{
		if (domain_D[idx].type == WALL) {
			vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION] = 0;
			vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 1] = 0;
			vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 2] = MAX_INTENSITY;
			continue;
		}

		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 2] = 0;
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 1] = visualisation_D->amplifier * abs(domain_D[idx].u[1]) * (domain_D[idx].u[1] > 0);
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION] = visualisation_D->amplifier * abs(domain_D[idx].u[1]) * (domain_D[idx].u[1] < 0);
	}
}

void LGA_draw(LGA_Config* configuration, float* devPtr)
{
	cudaMemcpy(configuration->visualisation_Device, &configuration->visualisation, sizeof(Visualisation), cudaMemcpyHostToDevice);
	switch (configuration->visualisation.field)
	{
	case 0:
		LGA_K_Draw_Density << < configuration->gridSize, configuration->blockSize >> > (configuration->domain_Device, devPtr, 
			configuration->visualisation_Device);
		break;
	case 1:
		LGA_K_Draw_Velocity_Norm << < configuration->gridSize, configuration->blockSize >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	case 2:
		LGA_K_Draw_Velocity_Horizontal << < configuration->gridSize, configuration->blockSize >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	case 3:
		LGA_K_Draw_Velocity_Vertical << < configuration->gridSize, configuration->blockSize >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	default:
		LGA_K_Draw << < configuration->gridSize, configuration->blockSize >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	}
	cudaDeviceSynchronize();
}

void setConstantMemory(LGA_Config* config)
{
	cudaMemcpyToSymbol("NX_global", &config->nx, sizeof(unsigned int));
	cudaMemcpyToSymbol("NY_global", &config->ny, sizeof(unsigned int));
	float time_coef = 1 / config->relaxationTime;
	cudaMemcpyToSymbol("Time_coef", &time_coef, sizeof(float), 0, cudaMemcpyHostToDevice);
	float c1 = 1 / (config->cs * config->cs);
	float c3 = c1 * 0.5;
	float c2 = c3 * c1;
	cudaMemcpyToSymbol("COEF_1", &c1, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("COEF_2", &c1, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("COEF_3", &c1, sizeof(float), 0, cudaMemcpyHostToDevice);
}

__global__ void LGA_K_Equalibrium(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);
	if (domain_D[idx].type == WALL) {
		return;
	}

	float u_square =  COEF_3*(domain_D[idx].u[0] * domain_D[idx].u[0] + domain_D[idx].u[1] * domain_D[idx].u[1]);
	domain_D[idx].eqStreams[0] = (4.0f/9.0f)*domain_D[idx].ro*(1.0f - u_square);

	domain_D[idx].eqStreams[1] = (1.0f / 9.0f) * domain_D[idx].ro*(1.0f + COEF_1 * domain_D[idx].u[0] + COEF_2* domain_D[idx].u[0]* domain_D[idx].u[0] - u_square);
	domain_D[idx].eqStreams[2] = (1.0f/9.0f) * domain_D[idx].ro*(1.0f - COEF_1 * domain_D[idx].u[0] + COEF_2 * domain_D[idx].u[0] * domain_D[idx].u[0] - u_square);
	domain_D[idx].eqStreams[3] = (1.0f/9.0f) * domain_D[idx].ro * (1.0f + COEF_1 * domain_D[idx].u[1] + COEF_2 * domain_D[idx].u[1] * domain_D[idx].u[1] - u_square);
	domain_D[idx].eqStreams[4] = (1.0f/9.0f) * domain_D[idx].ro * (1.0f - COEF_1 * domain_D[idx].u[1] + COEF_2 * domain_D[idx].u[1] * domain_D[idx].u[1] - u_square);
	
	domain_D[idx].eqStreams[5] = (1.0f / 36.0f) * domain_D[idx].ro * (1.0f + COEF_1 * (domain_D[idx].u[0] + domain_D[idx].u[1]) + 
		COEF_2 * (domain_D[idx].u[0] + domain_D[idx].u[1]) * (domain_D[idx].u[0] + domain_D[idx].u[1]) - u_square);

	domain_D[idx].eqStreams[6] = (1.0f / 36.0f) * domain_D[idx].ro * (1.0f + COEF_1 * (-domain_D[idx].u[0] + domain_D[idx].u[1]) +
		COEF_2 * (-domain_D[idx].u[0] + domain_D[idx].u[1]) * (-domain_D[idx].u[0] + domain_D[idx].u[1]) - u_square);

	domain_D[idx].eqStreams[7] = (1.0f / 36.0f) * domain_D[idx].ro * (1.0f + COEF_1 * (-domain_D[idx].u[0] - domain_D[idx].u[1]) +
		COEF_2 * (-domain_D[idx].u[0] - domain_D[idx].u[1]) * (-domain_D[idx].u[0] - domain_D[idx].u[1]) - u_square);

	domain_D[idx].eqStreams[8] = (1.0f / 36.0f) * domain_D[idx].ro * (1.0f + COEF_1 * (domain_D[idx].u[0] - domain_D[idx].u[1]) +
		COEF_2 * (domain_D[idx].u[0] - domain_D[idx].u[1]) * (domain_D[idx].u[0] - domain_D[idx].u[1]) - u_square);
}

