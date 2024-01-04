#pragma once
#include"..\inc\LGA_kernels.cuh"
#include<stdint.h>
#include<glfw3.h>

#define C_1 2.7632
#define C_2 3.8177
#define C_3 1.3816

#define FIELD 1

//__constant__ int64_t states = 0b1111110111101100011110100110010010111001010110000011000100100000;

__constant__ unsigned int NX_global;
__constant__ unsigned int NY_global;
__constant__ double Time_coef= 1;
__constant__ double COEF_1;
__constant__ double COEF_2;
__constant__ double COEF_3;

__constant__ double friction;

__constant__ int BOUNDRY_N;
__constant__ int BOUNDRY_S;
__constant__ int BOUNDRY_W;
__constant__ int BOUNDRY_E;

__global__ void LBM_K_Streaming(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x + 1 +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y + 1);
	if (threadIdx.x + blockDim.x * blockIdx.x + 1 >= NX_global) return;
	if (idx >= NX_global*(NY_global-1)) return;
	// ZA£O¯ENIE - MA£A LICZBA KOMÓREK GRANICZY ZE ŒCIANAMI //

	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];

	domain_D[idx].inStreams[1] = domain_D[idx - 1].outStreams[1];

	domain_D[idx].inStreams[2] = domain_D[idx + 1].outStreams[2];

	domain_D[idx].inStreams[3] = domain_D[idx + NX_global].outStreams[3];

	domain_D[idx].inStreams[4] = domain_D[idx - NX_global].outStreams[4];

	domain_D[idx].inStreams[5] = domain_D[idx + NX_global - 1].outStreams[5];

	domain_D[idx].inStreams[6] = domain_D[idx + NX_global + 1].outStreams[6];

	domain_D[idx].inStreams[7] = domain_D[idx - NX_global + 1].outStreams[7];

	domain_D[idx].inStreams[8] = domain_D[idx - NX_global - 1].outStreams[8];

	domain_D[idx].ro = 0;

	for (int i = 0; i < 9; i += 3)
	{
		domain_D[idx].ro += domain_D[idx].inStreams[i];
		domain_D[idx].ro += domain_D[idx].inStreams[i + 1];
		domain_D[idx].ro += domain_D[idx].inStreams[i + 2];
	}
	//domain_D[idx].ro = domain_D[idx].ro * (domain_D[idx].ro > 0);

	double ro_rev =  (domain_D[idx].ro != 0) ? (1.0f / domain_D[idx].ro) : 0.0f;
	
	domain_D[idx].u[0] = (domain_D[idx].inStreams[1] + domain_D[idx].inStreams[5] + domain_D[idx].inStreams[8] -
		domain_D[idx].inStreams[2] - domain_D[idx].inStreams[6] - domain_D[idx].inStreams[7])* ro_rev;

	domain_D[idx].u[1] = (domain_D[idx].inStreams[3] + domain_D[idx].inStreams[5] + domain_D[idx].inStreams[6] -
		domain_D[idx].inStreams[4] - domain_D[idx].inStreams[7] - domain_D[idx].inStreams[8]) * ro_rev;
}

__global__ void LBM_K_Collsion(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);

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

__global__ void LBM_K_Boundry_S(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x + 1;
	domain_D[idx].inStreams[4] = domain_D[idx + NX_global].outStreams[4];
	domain_D[idx].inStreams[1] = domain_D[idx-1].outStreams[1];
	domain_D[idx].inStreams[2] = domain_D[idx+1].outStreams[2];
	domain_D[idx].inStreams[8] = domain_D[idx + NX_global - 1].outStreams[8];
	domain_D[idx].inStreams[7] = domain_D[idx + NX_global + 1].outStreams[7];
	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];

	switch (BOUNDRY_S)
	{
	case 1: //bouncy
		domain_D[idx].inStreams[3] = domain_D[idx].outStreams[4];
		domain_D[idx].inStreams[5] = domain_D[idx].outStreams[7];
		domain_D[idx].inStreams[6] = domain_D[idx].outStreams[8];
		break;
	case 2:
		domain_D[idx].inStreams[3] = domain_D[idx].outStreams[4];
		domain_D[idx].inStreams[5] = domain_D[idx].outStreams[8];
		domain_D[idx].inStreams[6] = domain_D[idx].outStreams[7];
		break;
	case 3:
		domain_D[idx].inStreams[3] = domain_D[idx].inStreams[4];
		domain_D[idx].inStreams[5] = friction * domain_D[idx].inStreams[7] + (1.0 - friction) * domain_D[idx].inStreams[8];
		domain_D[idx].inStreams[6] = friction * domain_D[idx].inStreams[8] + (1.0 - friction) * domain_D[idx].inStreams[7];
		break;
	}

	for (int i = 0; i < 9; i += 3)
	{
		domain_D[idx].ro += domain_D[idx].inStreams[i];
		domain_D[idx].ro += domain_D[idx].inStreams[i + 1];
		domain_D[idx].ro += domain_D[idx].inStreams[i + 2];
	}

	double ro_rev = (domain_D[idx].ro != 0) ? (1.0f / domain_D[idx].ro) : 0.0f;

	domain_D[idx].u[0] = (domain_D[idx].inStreams[1] + domain_D[idx].inStreams[5] + domain_D[idx].inStreams[8] -
		domain_D[idx].inStreams[2] - domain_D[idx].inStreams[6] - domain_D[idx].inStreams[7]) * ro_rev;

	domain_D[idx].u[1] = (domain_D[idx].inStreams[3] + domain_D[idx].inStreams[5] + domain_D[idx].inStreams[6] -
		domain_D[idx].inStreams[4] - domain_D[idx].inStreams[7] - domain_D[idx].inStreams[8]) * ro_rev;
}

__global__ void LBM_K_Boundry_N(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x + 1 + NX_global * (NY_global - 1);
	if (idx >= NX_global * NY_global - 1) return;

	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];
	domain_D[idx].inStreams[1] = domain_D[idx - 1].outStreams[1];
	domain_D[idx].inStreams[2] = domain_D[idx + 1].outStreams[2];
	domain_D[idx].inStreams[3] = domain_D[idx - NX_global].outStreams[3];
	domain_D[idx].inStreams[6] = domain_D[idx - NX_global + 1].outStreams[6];
	domain_D[idx].inStreams[5] = domain_D[idx - NX_global - 1].outStreams[5];

	switch (BOUNDRY_N)
	{
	case 1:
		domain_D[idx].inStreams[4] = domain_D[idx].outStreams[3];
		domain_D[idx].inStreams[7] = domain_D[idx].outStreams[5];
		domain_D[idx].inStreams[8] = domain_D[idx].outStreams[6];
		break;
	case 2: //symmetry
		domain_D[idx].inStreams[4] = domain_D[idx].outStreams[3];
		domain_D[idx].inStreams[7] = domain_D[idx].outStreams[6];
		domain_D[idx].inStreams[8] = domain_D[idx].outStreams[5];
		break;
	case 3: //wall
		domain_D[idx].inStreams[4] = domain_D[idx].outStreams[3];
		domain_D[idx].inStreams[7] = friction * domain_D[idx].inStreams[5] + (1.0 - friction) * domain_D[idx].outStreams[6];
		domain_D[idx].inStreams[8] = friction * domain_D[idx].inStreams[6] + (1.0 - friction) * domain_D[idx].outStreams[5];
		break;
	case 4: //constant normal velocity
		//domain_D[idx].inStreams[4] = domain_D[idx].inStreams[3] + 2.0 / 3.0 * domain_D[idx].ro * domain_D[idx].u[1];
		break;
	case 5:
		break;
	default:
		break;
	}

	for (int i = 0; i < 9; i += 3)
	{
		domain_D[idx].ro += domain_D[idx].inStreams[i];
		domain_D[idx].ro += domain_D[idx].inStreams[i + 1];
		domain_D[idx].ro += domain_D[idx].inStreams[i + 2];
	}
	//domain_D[idx].ro = domain_D[idx].ro * (domain_D[idx].ro > 0);

	double ro_rev = (domain_D[idx].ro != 0) ? (1.0f / domain_D[idx].ro) : 0.0f;

	domain_D[idx].u[0] = (domain_D[idx].inStreams[1] + domain_D[idx].inStreams[5] + domain_D[idx].inStreams[8] -
		domain_D[idx].inStreams[2] - domain_D[idx].inStreams[6] - domain_D[idx].inStreams[7]) * ro_rev;

	domain_D[idx].u[1] = (domain_D[idx].inStreams[3] + domain_D[idx].inStreams[5] + domain_D[idx].inStreams[6] -
		domain_D[idx].inStreams[4] - domain_D[idx].inStreams[7] - domain_D[idx].inStreams[8]) * ro_rev;
}

__global__ void LBM_K_Boundry_E(Field* domain_D)
{
	int idx = (threadIdx.x + blockDim.x * blockIdx.x + 1) * (NX_global) + NX_global - 1;
	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];
	domain_D[idx].inStreams[1] = domain_D[idx - 1].outStreams[1];
	domain_D[idx].inStreams[3] = domain_D[idx + NX_global].outStreams[3];
	domain_D[idx].inStreams[4] = domain_D[idx - NX_global].outStreams[4];
	domain_D[idx].inStreams[5] = domain_D[idx + NX_global - 1].outStreams[5];
	domain_D[idx].inStreams[8] = domain_D[idx - NX_global - 1].outStreams[8];

	switch (BOUNDRY_E)
	{
	case 1:
		domain_D[idx].inStreams[2] = domain_D[idx].outStreams[1];
		domain_D[idx].inStreams[6] = domain_D[idx].outStreams[8];
		domain_D[idx].inStreams[7] = domain_D[idx].outStreams[5];
		break;
	case 2:
		domain_D[idx].inStreams[2] = domain_D[idx].outStreams[1];
		domain_D[idx].inStreams[7] = domain_D[idx].outStreams[8];
		domain_D[idx].inStreams[6] = domain_D[idx].outStreams[5];
		break;
	case 3:
		domain_D[idx].inStreams[2] = domain_D[idx].outStreams[1];
		domain_D[idx].inStreams[6] = friction * domain_D[idx].outStreams[8] + (1.0 - friction) * domain_D[idx].outStreams[5];
		domain_D[idx].inStreams[7] = friction * domain_D[idx].outStreams[5] + (1.0 - friction) * domain_D[idx].outStreams[8];
		break;
	}
	for (int i = 0; i < 9; i += 3)
	{
		domain_D[idx].ro += domain_D[idx].inStreams[i];
		domain_D[idx].ro += domain_D[idx].inStreams[i + 1];
		domain_D[idx].ro += domain_D[idx].inStreams[i + 2];
	}
	//domain_D[idx].ro = domain_D[idx].ro * (domain_D[idx].ro > 0);

	double ro_rev = (domain_D[idx].ro != 0) ? (1.0f / domain_D[idx].ro) : 0.0f;

	domain_D[idx].u[0] = (domain_D[idx].inStreams[1] + domain_D[idx].inStreams[5] + domain_D[idx].inStreams[8] -
		domain_D[idx].inStreams[2] - domain_D[idx].inStreams[6] - domain_D[idx].inStreams[7]) * ro_rev;

	domain_D[idx].u[1] = (domain_D[idx].inStreams[3] + domain_D[idx].inStreams[5] + domain_D[idx].inStreams[6] -
		domain_D[idx].inStreams[4] - domain_D[idx].inStreams[7] - domain_D[idx].inStreams[8]) * ro_rev;
}

__global__ void LBM_K_Boundry_W(Field* domain_D)
{
	int idx = (threadIdx.x + blockDim.x * blockIdx.x + 1) * (NX_global);
	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];
	domain_D[idx].inStreams[2] = domain_D[idx + 1].outStreams[2];
	domain_D[idx].inStreams[3] = domain_D[idx + NX_global].outStreams[3];
	domain_D[idx].inStreams[4] = domain_D[idx - NX_global].outStreams[4];
	domain_D[idx].inStreams[6] = domain_D[idx + NX_global + 1].outStreams[6];
	domain_D[idx].inStreams[7] = domain_D[idx - NX_global + 1].outStreams[7];

	switch (BOUNDRY_W)
	{
	case 1:
		domain_D[idx].inStreams[1] = domain_D[idx].outStreams[2];
		domain_D[idx].inStreams[5] = domain_D[idx].outStreams[7];
		domain_D[idx].inStreams[8] = domain_D[idx].outStreams[6];
		break;
	case 2:
		domain_D[idx].inStreams[1] = domain_D[idx].outStreams[2];
		domain_D[idx].inStreams[8] = domain_D[idx].outStreams[7];
		domain_D[idx].inStreams[5] = domain_D[idx].outStreams[6];
		break;
	case 3:
		domain_D[idx].inStreams[1] = domain_D[idx].outStreams[2];
		domain_D[idx].inStreams[5] = friction * domain_D[idx].outStreams[7] + (1.0 - friction) * domain_D[idx].outStreams[8];
		domain_D[idx].inStreams[8] = friction * domain_D[idx].outStreams[6] + (1.0 - friction) * domain_D[idx].outStreams[7];
		break;
	}

	for (int i = 0; i < 9; i += 3)
	{
		domain_D[idx].ro += domain_D[idx].inStreams[i];
		domain_D[idx].ro += domain_D[idx].inStreams[i + 1];
		domain_D[idx].ro += domain_D[idx].inStreams[i + 2];
	}

	double ro_rev = (domain_D[idx].ro != 0) ? (1.0f / domain_D[idx].ro) : 0.0f;

	domain_D[idx].u[0] = (domain_D[idx].inStreams[1] + domain_D[idx].inStreams[5] + domain_D[idx].inStreams[8] -
		domain_D[idx].inStreams[2] - domain_D[idx].inStreams[6] - domain_D[idx].inStreams[7]) * ro_rev;

	domain_D[idx].u[1] = (domain_D[idx].inStreams[3] + domain_D[idx].inStreams[5] + domain_D[idx].inStreams[6] -
		domain_D[idx].inStreams[4] - domain_D[idx].inStreams[7] - domain_D[idx].inStreams[8]) * ro_rev;
}

__global__ void LBM_K_Boundry_NE(Field* domain_D)
{
	int idx = NX_global * NY_global - 1;
	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];
	domain_D[idx].inStreams[1] = domain_D[idx - 1].outStreams[1];
	domain_D[idx].inStreams[3] = domain_D[idx - NX_global].outStreams[3];
	domain_D[idx].inStreams[5] = domain_D[idx - NX_global - 1].outStreams[5];

	switch (BOUNDRY_N)
	{
	case 1:
		switch (BOUNDRY_E)
		{
		case 1:
		}
	case 2:
		switch (BOUNDRY_E)
		{
		case 1:
		}
	}
}
__global__ void LBM_K_Boundry_NW(Field* domain_D)
{

}
__global__ void LBM_K_Boundry_SE(Field* domain_D)
{

}
__global__ void LBM_K_Boundry_SW(Field* domain_D)
{

}


void LBM_run(LBM_Config* configuration)
{
	LBM_K_Equalibrium << < configuration->gridSize_Collsion, configuration->blockSize_Collsion >> > (configuration->domain_Device);
	cudaDeviceSynchronize();
	LBM_K_Collsion <<< configuration->gridSize_Collsion, configuration->blockSize_Collsion >> > (configuration->domain_Device);
	cudaDeviceSynchronize();
	LBM_K_Streaming << < configuration->gridSize_Streaming, configuration->blockSize_Streaming >> > (configuration->domain_Device);

	LBM_K_Boundry_N << < configuration->gridSize_BoundryN, configuration->blockSize_BoundryN >> > (configuration->domain_Device);
	//LBM_K_Boundry_S << < configuration->gridSize_BoundryS, configuration->blockSize_BoundryS >> > (configuration->domain_Device);
	//LBM_K_Boundry_E << < configuration->gridSize_BoundryE, configuration->blockSize_BoundryE >> > (configuration->domain_Device);
	//LBM_K_Boundry_W << < configuration->gridSize_BoundryW, configuration->blockSize_BoundryW >> > (configuration->domain_Device);
	cudaDeviceSynchronize();
}

__global__ void LBM_K_Draw(Field* domain_D, float* vbo, Visualisation* visualisation_D)
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

__global__ void LBM_K_Draw_Density(Field* domain_D, float* vbo, Visualisation* visualisation_D)
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

		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 2] = visualisation_D->amplifier* domain_D[idx].ro;
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION + 1] = visualisation_D->amplifier * domain_D[idx].ro;
		vbo[idx * VERTEX_SIZE * 4 + v * VERTEX_SIZE + DIMENSION] = visualisation_D->amplifier * domain_D[idx].ro ;
	}
}

__global__ void LBM_K_Draw_Velocity_Norm(Field* domain_D, float* vbo, Visualisation* visualisation_D)
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

__global__ void LBM_K_Draw_Velocity_Horizontal(Field* domain_D, float* vbo, Visualisation* visualisation_D)
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

__global__ void LBM_K_Draw_Velocity_Vertical(Field* domain_D, float* vbo, Visualisation* visualisation_D)
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


void LBM_draw(LBM_Config* configuration, float* devPtr)
{
	cudaMemcpy(configuration->visualisation_Device, &configuration->visualisation, sizeof(Visualisation), cudaMemcpyHostToDevice);
	switch (configuration->visualisation.field)
	{
	case 0:
		LBM_K_Draw_Density << < configuration->gridSize_Collsion, configuration->blockSize_Collsion >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	case 1:
		LBM_K_Draw_Velocity_Norm << < configuration->gridSize_Collsion, configuration->blockSize_Collsion >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	case 2:
		LBM_K_Draw_Velocity_Horizontal << < configuration->gridSize_Collsion, configuration->blockSize_Collsion >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	case 3:
		LBM_K_Draw_Velocity_Vertical << < configuration->gridSize_Collsion, configuration->blockSize_Collsion >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	default:
		LBM_K_Draw << < configuration->gridSize_Collsion, configuration->blockSize_Collsion >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	}
	cudaDeviceSynchronize();
}

void setConstantMemory(LBM_Config* config)
{
	cudaMemcpyToSymbol(NX_global, &config->nx, sizeof(unsigned int));
	cudaMemcpyToSymbol(NY_global, &config->ny, sizeof(unsigned int));
	double time_coef = 1.0 / config->relaxationTime;
	//cudaMemcpyToSymbol(Time_coef, &time_coef, sizeof(double), 0, cudaMemcpyHostToDevice);
	double c1 =  1.0 / (config->cs * config->cs);
	double c3 = c1 * 0.5;
	double c2 = c3* c1;
	cudaMemcpyToSymbol(COEF_1, &c1, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(COEF_2, &c1, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(COEF_3, &c1, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(BOUNDRY_N, &(config->boundryN.type), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(BOUNDRY_S, &(config->boundryS.type), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(BOUNDRY_E, &(config->boundryE.type), sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(BOUNDRY_W, &(config->boundryW.type), sizeof(int), 0, cudaMemcpyHostToDevice);
}

__global__ void LBM_K_Equalibrium(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX_global * (threadIdx.y + blockDim.y * blockIdx.y);
	if (domain_D[idx].type == WALL) {
		return;
	}

	float u_square =  COEF_3*(domain_D[idx].u[0] * domain_D[idx].u[0] + domain_D[idx].u[1] * domain_D[idx].u[1]);
	domain_D[idx].eqStreams[0] = (4.0/9.0)*domain_D[idx].ro*(1.0f - u_square);

	domain_D[idx].eqStreams[1] = (1.0 / 9.0) * domain_D[idx].ro*(1.0 + COEF_1 * domain_D[idx].u[0] + COEF_2* domain_D[idx].u[0]* domain_D[idx].u[0] - u_square);
	domain_D[idx].eqStreams[2] = (1.0/9.0) * domain_D[idx].ro*(1.0 - COEF_1 * domain_D[idx].u[0] + COEF_2 * domain_D[idx].u[0] * domain_D[idx].u[0] - u_square);
	domain_D[idx].eqStreams[3] = (1.0/9.0) * domain_D[idx].ro * (1.0 + COEF_1 * domain_D[idx].u[1] + COEF_2 * domain_D[idx].u[1] * domain_D[idx].u[1] - u_square);
	domain_D[idx].eqStreams[4] = (1.0/9.0) * domain_D[idx].ro * (1.0 - COEF_1 * domain_D[idx].u[1] + COEF_2 * domain_D[idx].u[1] * domain_D[idx].u[1] - u_square);
	
	domain_D[idx].eqStreams[5] = (1.0 / 36.0) * domain_D[idx].ro * (1.0 + COEF_1 * (domain_D[idx].u[0] + domain_D[idx].u[1]) + 
		COEF_2 * (domain_D[idx].u[0] + domain_D[idx].u[1]) * (domain_D[idx].u[0] + domain_D[idx].u[1]) - u_square);

	domain_D[idx].eqStreams[6] = (1.0 / 36.0) * domain_D[idx].ro * (1.0 + COEF_1 * (-domain_D[idx].u[0] + domain_D[idx].u[1]) +
		COEF_2 * (-domain_D[idx].u[0] + domain_D[idx].u[1]) * (-domain_D[idx].u[0] + domain_D[idx].u[1]) - u_square);

	domain_D[idx].eqStreams[7] = (1.0 / 36.0) * domain_D[idx].ro * (1.0 + COEF_1 * (-domain_D[idx].u[0] - domain_D[idx].u[1]) +
		COEF_2 * (-domain_D[idx].u[0] - domain_D[idx].u[1]) * (-domain_D[idx].u[0] - domain_D[idx].u[1]) - u_square);

	domain_D[idx].eqStreams[8] = (1.0 / 36.f) * domain_D[idx].ro * (1.0 + COEF_1 * (domain_D[idx].u[0] - domain_D[idx].u[1]) +
		COEF_2 * (domain_D[idx].u[0] - domain_D[idx].u[1]) * (domain_D[idx].u[0] - domain_D[idx].u[1]) - u_square);
}

