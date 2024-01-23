#pragma once
#include"..\inc\LGA_kernels.cuh"
#include<stdint.h>
#include<glfw3.h>

#define C_1 2.7632
#define C_2 3.8177
#define C_3 1.3816

#define FIELD 1


__constant__ unsigned int NX;
__constant__ unsigned int NY;

__constant__ double Time_coef = 1;
__constant__ double COEF_1;
__constant__ double COEF_2;
__constant__ double COEF_3;

__constant__ double friction;

__constant__ BoundryCondition BOUNDRY_N;
__constant__ BoundryCondition BOUNDRY_S;
__constant__ BoundryCondition BOUNDRY_W;
__constant__ BoundryCondition BOUNDRY_E;


__global__ void LBM_K_Streaming(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x + 1 +
		NX * (threadIdx.y + blockDim.y * blockIdx.y + 1);
	if (threadIdx.x + blockDim.x * blockIdx.x + 1 >= NX - 1) return;
	if (idx >= NX*(NY-1)) return;

	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];

	domain_D[idx].inStreams[1] = domain_D[idx - 1].outStreams[1];

	domain_D[idx].inStreams[2] = domain_D[idx + 1].outStreams[2];

	domain_D[idx].inStreams[3] = domain_D[idx - NX].outStreams[3];

	domain_D[idx].inStreams[4] = domain_D[idx + NX].outStreams[4];

	domain_D[idx].inStreams[5] = domain_D[idx - NX - 1].outStreams[5];

	domain_D[idx].inStreams[6] = domain_D[idx - NX + 1].outStreams[6];

	domain_D[idx].inStreams[7] = domain_D[idx + NX + 1].outStreams[7];

	domain_D[idx].inStreams[8] = domain_D[idx + NX - 1].outStreams[8];

}

__global__ void LBM_K_Collsion(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX * (threadIdx.y + blockDim.y * blockIdx.y);
	if (idx > NX * NY - 1) return;
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

__global__ void LBM_K_Variables(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x + NX * (threadIdx.y + blockDim.y * blockIdx.y);
	domain_D[idx].ro = 0;

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

__global__ void LBM_K_Boundry_S(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x + 1;
	double pos = ((double)idx - 1);

	if (idx >= NX- 1) return;
	domain_D[idx].inStreams[4] = domain_D[idx + NX].outStreams[4];
	domain_D[idx].inStreams[1] = domain_D[idx-1].outStreams[1];
	domain_D[idx].inStreams[2] = domain_D[idx+1].outStreams[2];
	domain_D[idx].inStreams[8] = domain_D[idx + NX - 1].outStreams[8];
	domain_D[idx].inStreams[7] = domain_D[idx + NX + 1].outStreams[7];
	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];

	switch (BOUNDRY_S.type)
	{
	case BOUNCY_BACK: //bouncy
		domain_D[idx].inStreams[3] = domain_D[idx].inStreams[4];
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[7];
		domain_D[idx].inStreams[6] = domain_D[idx].inStreams[8];
		break;
	case SYMMETRY:
		domain_D[idx].inStreams[3] = domain_D[idx].inStreams[4];
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[8];
		domain_D[idx].inStreams[6] = domain_D[idx].inStreams[7];
		break;
	case WALL_FRICTION:
		domain_D[idx].inStreams[3] = domain_D[idx].inStreams[4];
		domain_D[idx].inStreams[5] = friction * domain_D[idx].inStreams[7] + (1.0 - friction) * domain_D[idx].inStreams[8];
		domain_D[idx].inStreams[6] = friction * domain_D[idx].inStreams[8] + (1.0 - friction) * domain_D[idx].inStreams[7];
		break;
	case CONSTANT_NORMAL_SPEED:
	{
		double vn0 = BOUNDRY_S.vn0 + BOUNDRY_S.d_vn * pos;

		domain_D[idx].ro = (domain_D[idx].inStreams[0] + domain_D[idx].inStreams[1] + domain_D[idx].inStreams[2] +
			2 * domain_D[idx].inStreams[4] + 2 * domain_D[idx].inStreams[7] + 2 * domain_D[idx].inStreams[8]) / (1.0 - vn0);
		domain_D[idx].u[1] = vn0;
		domain_D[idx].u[0] = 6.0 / domain_D[idx].ro * (domain_D[idx].inStreams[1] + domain_D[idx].inStreams[2] + domain_D[idx].inStreams[7] +
			domain_D[idx].inStreams[8]) / (5.0 - 3.0 * vn0);
		domain_D[idx].inStreams[3] = domain_D[idx].inStreams[4] + 2.0 / 3.0 * domain_D[idx].ro * vn0;
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[7] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 + domain_D[idx].u[0]);
		domain_D[idx].inStreams[6] = domain_D[idx].inStreams[8] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 - domain_D[idx].u[0]);
	}
		break;
	case CONSTANT_VELOCITY:
	{
		double vn0 = BOUNDRY_S.vn0 + BOUNDRY_S.d_vn * pos;

		domain_D[idx].ro = (domain_D[idx].inStreams[0] + domain_D[idx].inStreams[1] + domain_D[idx].inStreams[2] +
			2 * domain_D[idx].inStreams[4] + 2 * domain_D[idx].inStreams[7] + 2 * domain_D[idx].inStreams[8]) / (1.0 - vn0);
		domain_D[idx].u[1] = vn0;
		domain_D[idx].u[0] = BOUNDRY_S.vt0 + BOUNDRY_S.d_vt * pos;
		domain_D[idx].inStreams[3] = domain_D[idx].inStreams[4] + 2.0 / 3.0 * domain_D[idx].ro * vn0;
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[7] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 + domain_D[idx].u[0]);
		domain_D[idx].inStreams[6] = domain_D[idx].inStreams[8] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 - domain_D[idx].u[0]);
	}
		break;
	}
	
}

__global__ void LBM_K_Boundry_N(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x + 1 + NX * (NY - 1);
	double pos = ((double)(threadIdx.x + blockDim.x * blockIdx.x));

	if (idx >= NX * NY - 1) return;

	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];
	domain_D[idx].inStreams[1] = domain_D[idx - 1].outStreams[1];
	domain_D[idx].inStreams[2] = domain_D[idx + 1].outStreams[2];
	domain_D[idx].inStreams[3] = domain_D[idx - NX].outStreams[3];
	domain_D[idx].inStreams[6] = domain_D[idx - NX + 1].outStreams[6];
	domain_D[idx].inStreams[5] = domain_D[idx - NX - 1].outStreams[5];

	switch (BOUNDRY_N.type)
	{
	case BOUNCY_BACK:
		domain_D[idx].inStreams[4] = domain_D[idx].inStreams[3];
		domain_D[idx].inStreams[7] = domain_D[idx].inStreams[5];
		domain_D[idx].inStreams[8] = domain_D[idx].inStreams[6];
		break;
	case SYMMETRY:
		domain_D[idx].inStreams[4] = domain_D[idx].inStreams[3];
		domain_D[idx].inStreams[7] = domain_D[idx].inStreams[6];
		domain_D[idx].inStreams[8] = domain_D[idx].inStreams[5];
		break;
	case WALL_FRICTION:
		domain_D[idx].inStreams[4] = domain_D[idx].inStreams[3];
		domain_D[idx].inStreams[7] = friction * domain_D[idx].inStreams[5] + (1.0 - friction) * domain_D[idx].inStreams[6];
		domain_D[idx].inStreams[8] = friction * domain_D[idx].inStreams[6] + (1.0 - friction) * domain_D[idx].inStreams[5];
		break;
	case CONSTANT_NORMAL_SPEED:
	{
		double vn0 = BOUNDRY_N.vn0 + BOUNDRY_N.d_vn * pos;
		domain_D[idx].ro = (domain_D[idx].inStreams[0] + domain_D[idx].inStreams[1] + domain_D[idx].inStreams[2] +
			+2 * domain_D[idx].inStreams[3] + 2 * domain_D[idx].inStreams[5] + 2 * domain_D[idx].inStreams[6]) / (1.0 - vn0);
		domain_D[idx].u[1] = vn0;
		domain_D[idx].u[0] = 6.0 / domain_D[idx].ro * (domain_D[idx].inStreams[1] + domain_D[idx].inStreams[2] + domain_D[idx].inStreams[5] +
			domain_D[idx].inStreams[6]) / (5.0 - 3.0 * vn0);

		domain_D[idx].inStreams[4] = domain_D[idx].inStreams[3] + 2.0 / 3.0 * domain_D[idx].ro * vn0;
		domain_D[idx].inStreams[7] = domain_D[idx].inStreams[5] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 + domain_D[idx].u[0]);
		domain_D[idx].inStreams[8] = domain_D[idx].inStreams[6] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 - domain_D[idx].u[0]);
	}
		break;
	case CONSTANT_VELOCITY:
	{
		double vn0 = BOUNDRY_N.vn0 + BOUNDRY_N.d_vn * pos;
		domain_D[idx].ro = (domain_D[idx].inStreams[0] + domain_D[idx].inStreams[1] + domain_D[idx].inStreams[2] +
			+2 * domain_D[idx].inStreams[3] + 2 * domain_D[idx].inStreams[5] + 2 * domain_D[idx].inStreams[6]) / (1.0 - vn0);
		domain_D[idx].u[1] = vn0;
		domain_D[idx].u[0] = BOUNDRY_N.vt0 + BOUNDRY_N.d_vt * pos;

		domain_D[idx].inStreams[4] = domain_D[idx].inStreams[3] + 2.0 / 3.0 * domain_D[idx].ro * vn0;
		domain_D[idx].inStreams[7] = domain_D[idx].inStreams[5] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 + domain_D[idx].u[0]);
		domain_D[idx].inStreams[8] = domain_D[idx].inStreams[6] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 - domain_D[idx].u[0]);
	}
		break;
	default:
		return;
		break;
	}

}

__global__ void LBM_K_Boundry_E(Field* domain_D)
{
	if (threadIdx.x + blockDim.x * blockIdx.x + 1 >= NY) return;
	int idx = (threadIdx.x + blockDim.x * blockIdx.x + 1) * (NX) + NX - 1;
	double pos = (double)(threadIdx.x + blockDim.x * blockIdx.x);
	

	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];
	domain_D[idx].inStreams[1] = domain_D[idx - 1].outStreams[1];
	domain_D[idx].inStreams[3] = domain_D[idx - NX].outStreams[3];
	domain_D[idx].inStreams[4] = domain_D[idx + NX].outStreams[4];
	domain_D[idx].inStreams[5] = domain_D[idx - NX - 1].outStreams[5];
	domain_D[idx].inStreams[8] = domain_D[idx + NX - 1].outStreams[8];

	switch (BOUNDRY_E.type)
	{
	case BOUNCY_BACK:
		domain_D[idx].inStreams[2] = domain_D[idx].inStreams[1];
		domain_D[idx].inStreams[6] = domain_D[idx].inStreams[8];
		domain_D[idx].inStreams[7] = domain_D[idx].inStreams[5];
		break;
	case SYMMETRY:
		domain_D[idx].inStreams[2] = domain_D[idx].inStreams[1];
		domain_D[idx].inStreams[7] = domain_D[idx].inStreams[8];
		domain_D[idx].inStreams[6] = domain_D[idx].inStreams[5];
		break;
	case WALL_FRICTION:
		domain_D[idx].inStreams[2] = domain_D[idx].inStreams[1];
		domain_D[idx].inStreams[6] = friction * domain_D[idx].inStreams[8] + (1.0 - friction) * domain_D[idx].inStreams[5];
		domain_D[idx].inStreams[7] = friction * domain_D[idx].inStreams[5] + (1.0 - friction) * domain_D[idx].inStreams[8];
		break;
	case CONSTANT_NORMAL_SPEED:
	{
		double vn0 = BOUNDRY_E.vn0 + BOUNDRY_E.d_vn * pos;
		domain_D[idx].ro = (domain_D[idx].inStreams[0] + domain_D[idx].inStreams[3] + domain_D[idx].inStreams[4] +
			2 * domain_D[idx].inStreams[1] + 2 * domain_D[idx].inStreams[5] + 2 * domain_D[idx].inStreams[8]) / (1.0 - vn0);
		domain_D[idx].u[0] = vn0;
		domain_D[idx].u[1] = 6.0 / domain_D[idx].ro * (domain_D[idx].inStreams[3] + domain_D[idx].inStreams[4] + domain_D[idx].inStreams[5] +
			domain_D[idx].inStreams[8]) / (5.0 - 3.0 * vn0);
		domain_D[idx].inStreams[2] = domain_D[idx].inStreams[1] + 2.0 / 3.0 * domain_D[idx].ro * vn0;
		domain_D[idx].inStreams[6] = domain_D[idx].inStreams[8] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 + domain_D[idx].u[1]);
		domain_D[idx].inStreams[7] = domain_D[idx].inStreams[5] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 - domain_D[idx].u[1]);
	}
		break;
	case CONSTANT_VELOCITY:
	{
		double vn0 = BOUNDRY_E.vn0 + BOUNDRY_E.d_vn * pos;
		domain_D[idx].ro = (domain_D[idx].inStreams[0] + domain_D[idx].inStreams[3] + domain_D[idx].inStreams[4] +
			2 * domain_D[idx].inStreams[1] + 2 * domain_D[idx].inStreams[5] + 2 * domain_D[idx].inStreams[8]) / (1.0 - vn0);
		domain_D[idx].u[0] = vn0;
		domain_D[idx].u[1] = BOUNDRY_E.vt0 + BOUNDRY_E.d_vt * pos;
		domain_D[idx].inStreams[2] = domain_D[idx].inStreams[1] + 2.0 / 3.0 * domain_D[idx].ro * vn0;
		domain_D[idx].inStreams[6] = domain_D[idx].inStreams[8] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 + domain_D[idx].u[1]);
		domain_D[idx].inStreams[7] = domain_D[idx].inStreams[5] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 - domain_D[idx].u[1]);
	}
		break;
	}

}

__global__ void LBM_K_Boundry_W(Field* domain_D)
{
	if (threadIdx.x + blockDim.x * blockIdx.x + 1 >= NY) return;
	int idx = (threadIdx.x + blockDim.x * blockIdx.x + 1) * (NX);
	double pos = (double)(threadIdx.x + blockDim.x * blockIdx.x);

	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];
	domain_D[idx].inStreams[2] = domain_D[idx + 1].outStreams[2];
	domain_D[idx].inStreams[3] = domain_D[idx - NX].outStreams[3];
	domain_D[idx].inStreams[4] = domain_D[idx + NX].outStreams[4];
	domain_D[idx].inStreams[6] = domain_D[idx - NX + 1].outStreams[6];
	domain_D[idx].inStreams[7] = domain_D[idx + NX + 1].outStreams[7];

	switch (BOUNDRY_W.type)
	{
	case BOUNCY_BACK:
		domain_D[idx].inStreams[1] = domain_D[idx].inStreams[2];
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[7];
		domain_D[idx].inStreams[8] = domain_D[idx].inStreams[6];
		break;
	case SYMMETRY:
		domain_D[idx].inStreams[1] = domain_D[idx].inStreams[2];
		domain_D[idx].inStreams[8] = domain_D[idx].inStreams[7];
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[6];
		break;
	case WALL_FRICTION:
		domain_D[idx].inStreams[1] = domain_D[idx].inStreams[2];
		domain_D[idx].inStreams[5] = friction * domain_D[idx].inStreams[7] + (1.0 - friction) * domain_D[idx].inStreams[8];
		domain_D[idx].inStreams[8] = friction * domain_D[idx].inStreams[6] + (1.0 - friction) * domain_D[idx].inStreams[7];
	case CONSTANT_NORMAL_SPEED:
	{
		double vn0 = BOUNDRY_W.vn0 + BOUNDRY_W.d_vn * pos;
		domain_D[idx].ro = (domain_D[idx].inStreams[0] + domain_D[idx].inStreams[3] + domain_D[idx].inStreams[4] +
			+2 * domain_D[idx].inStreams[2] + 2 * domain_D[idx].inStreams[6] + 2 * domain_D[idx].inStreams[7]) / (1.0 - vn0);
		domain_D[idx].u[0] = BOUNDRY_W.vn0;
		domain_D[idx].u[1] = 6.0 / domain_D[idx].ro * (domain_D[idx].inStreams[3] + domain_D[idx].inStreams[4] + domain_D[idx].inStreams[6] +
			domain_D[idx].inStreams[7]) / (5.0 - 3.0 * vn0);
		domain_D[idx].inStreams[1] = domain_D[idx].inStreams[2] + 2.0 / 3.0 * domain_D[idx].ro * vn0;
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[7] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 - domain_D[idx].u[1]);
		domain_D[idx].inStreams[8] = domain_D[idx].inStreams[6] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 + domain_D[idx].u[1]);
	}
		break;
	case CONSTANT_VELOCITY:
	{
		double vn0 = BOUNDRY_W.vn0 + BOUNDRY_W.d_vn * pos;
		domain_D[idx].ro = (domain_D[idx].inStreams[0] + domain_D[idx].inStreams[3] + domain_D[idx].inStreams[4] +
			+2 * domain_D[idx].inStreams[2] + 2 * domain_D[idx].inStreams[6] + 2 * domain_D[idx].inStreams[7]) / (1.0 - vn0);
		domain_D[idx].u[0] = vn0;
		domain_D[idx].u[1] = BOUNDRY_W.vt0 + BOUNDRY_W.d_vt * pos;
		domain_D[idx].inStreams[1] = domain_D[idx].inStreams[2] + 2.0 / 3.0 * domain_D[idx].ro * vn0;
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[7] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 - domain_D[idx].u[1]);
		domain_D[idx].inStreams[8] = domain_D[idx].inStreams[6] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 + domain_D[idx].u[1]); 
	}
		break;
	}

}

__global__ void LBM_K_Boundry_NE(Field* domain_D)
{
	int idx = NX * NY - 1;

	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];
	domain_D[idx].inStreams[1] = domain_D[idx - 1].outStreams[1];
	domain_D[idx].inStreams[3] = domain_D[idx - NX].outStreams[3];
	domain_D[idx].inStreams[5] = domain_D[idx - NX - 1].outStreams[5];

	if ( (BOUNDRY_N.type == 1 && BOUNDRY_E.type == 1) || (BOUNDRY_N.type == 4 && BOUNDRY_E.type == 1) || (BOUNDRY_N.type == 1 && BOUNDRY_E.type == 4) )
	{
		domain_D[idx].inStreams[2] = domain_D[idx].inStreams[1];
		domain_D[idx].inStreams[4] = domain_D[idx].inStreams[3];
		domain_D[idx].inStreams[6] = domain_D[idx].inStreams[5];  
		domain_D[idx].inStreams[7] = domain_D[idx].inStreams[5];
		domain_D[idx].inStreams[8] = domain_D[idx].inStreams[5];
	}
	else if ( BOUNDRY_N .type != BOUNDRY_E.type) // inflow/outflow E
	{
		double vn0 = BOUNDRY_E.vn0 + BOUNDRY_E.d_vn * (NY-1);
		domain_D[idx].ro = (domain_D[idx].inStreams[0] + 2.0 * domain_D[idx].inStreams[3] + 2.0 * domain_D[idx].inStreams[1] + 4.0 * domain_D[idx].inStreams[5]) / (1.0 - vn0);
		if (BOUNDRY_N.type == CONSTANT_VELOCITY)
		{
			domain_D[idx].u[1] = BOUNDRY_N.vn0;
		}
		else
		{
			domain_D[idx].u[1] = 0;
			domain_D[idx].u[0] = vn0;
		}
		domain_D[idx].inStreams[4] = domain_D[idx].inStreams[3];
		domain_D[idx].inStreams[2] = domain_D[idx].inStreams[1] + 2.0 / 3.0 * domain_D[idx].ro * (vn0 + domain_D[idx].u[1]);
		domain_D[idx].inStreams[6] = domain_D[idx].inStreams[5];
		domain_D[idx].inStreams[7] = domain_D[idx].inStreams[5] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 + domain_D[idx].u[1]);
	}
}

__global__ void LBM_K_Boundry_NW(Field* domain_D)
{
	int idx = NX * (NY - 1);
	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];
	domain_D[idx].inStreams[2] = domain_D[idx + 1].outStreams[2];
	domain_D[idx].inStreams[3] = domain_D[idx - NX].outStreams[3];
	domain_D[idx].inStreams[6] = domain_D[idx - NX + 1].outStreams[6];

	if ( (BOUNDRY_N.type == 1 && BOUNDRY_W.type == 1) || (BOUNDRY_N.type == 4 && BOUNDRY_W.type == 1) || (BOUNDRY_N.type == 1 && BOUNDRY_W.type == 4) )
	{
		domain_D[idx].inStreams[1] = domain_D[idx].inStreams[2];
		domain_D[idx].inStreams[4] = domain_D[idx].inStreams[3];
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[6];
		domain_D[idx].inStreams[7] = domain_D[idx].inStreams[6];
		domain_D[idx].inStreams[8] = domain_D[idx].inStreams[6];
	}
	else if (BOUNDRY_N.type != BOUNDRY_W.type && BOUNDRY_W.type == CONSTANT_NORMAL_SPEED) //inflow/outflow W
	{
		double vn0 = BOUNDRY_W.vn0 + BOUNDRY_W.d_vn * (NY - 1);
		domain_D[idx].ro = (domain_D[idx].inStreams[0] + 2.0 * domain_D[idx].inStreams[3] + 2.0 * domain_D[idx].inStreams[2] + 4.0 * domain_D[idx].inStreams[6]) / (1.0 - vn0);
		domain_D[idx].u[1] = 0;
		domain_D[idx].u[0] = vn0;
		domain_D[idx].inStreams[4] = domain_D[idx].inStreams[3];
		domain_D[idx].inStreams[1] = domain_D[idx].inStreams[2] + 2.0 / 3.0 * domain_D[idx].ro * (vn0 + domain_D[idx].u[1]);
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[6];
		domain_D[idx].inStreams[8] = domain_D[idx].inStreams[6] + 1.0 / 6.0 * domain_D[idx].ro * (vn0 + domain_D[idx].u[1]);

	}
}

__global__ void LBM_K_Boundry_SE(Field* domain_D)
{
	int idx = NX - 1;
	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];
	domain_D[idx].inStreams[1] = domain_D[idx - 1].outStreams[1];
	domain_D[idx].inStreams[4] = domain_D[idx + NX].outStreams[4];
	domain_D[idx].inStreams[8] = domain_D[idx + NX - 1].outStreams[8];

	if ( (BOUNDRY_S.type == 1 && BOUNDRY_E.type == 1) || (BOUNDRY_S.type == 4 && BOUNDRY_E.type == 1) || (BOUNDRY_S.type == 1 && BOUNDRY_E.type == 4) )
	{
		domain_D[idx].inStreams[2] = domain_D[idx].inStreams[1];
		domain_D[idx].inStreams[3] = domain_D[idx].inStreams[4];
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[8];
		domain_D[idx].inStreams[6] = domain_D[idx].inStreams[8];
		domain_D[idx].inStreams[7] = domain_D[idx].inStreams[8];
	}
	else if (BOUNDRY_S.type != BOUNDRY_E.type) // inflow/outflow E
	{
		double vn0 = BOUNDRY_E.vn0;
		domain_D[idx].ro = (domain_D[idx].inStreams[0] + 2 * domain_D[idx].inStreams[1] + 2 * domain_D[idx].inStreams[4] + 4 * domain_D[idx].inStreams[8]) / (1.0 - vn0);
		domain_D[idx].u[1] = 0;
		domain_D[idx].u[0] = vn0;
		domain_D[idx].inStreams[3] = domain_D[idx].inStreams[4];
		domain_D[idx].inStreams[2] = domain_D[idx].inStreams[1] + 2.0 / 3.0 * domain_D[idx].ro * vn0;
		domain_D[idx].inStreams[7] = domain_D[idx].inStreams[8];
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[8] + 1.0 / 6.0 * domain_D[idx].ro * vn0;

	}
	
}

__global__ void LBM_K_Boundry_SW(Field* domain_D)
{
	int idx = 0;

	domain_D[idx].inStreams[0] = domain_D[idx].outStreams[0];
	domain_D[idx].inStreams[2] = domain_D[idx + 1].outStreams[2];
	domain_D[idx].inStreams[4] = domain_D[idx + NX].outStreams[4];
	domain_D[idx].inStreams[7] = domain_D[idx + NX + 1].outStreams[7];

	if ( ( BOUNDRY_S.type == 1 && BOUNDRY_W.type == 1 ) || (BOUNDRY_S.type == 4 && BOUNDRY_W.type == 1) || (BOUNDRY_S.type == 1 && BOUNDRY_W.type == 4) )
	{
		domain_D[idx].inStreams[1] = domain_D[idx].inStreams[2];
		domain_D[idx].inStreams[3] = domain_D[idx].inStreams[4];
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[7];
		domain_D[idx].inStreams[6] = domain_D[idx].inStreams[7];
		domain_D[idx].inStreams[8] = domain_D[idx].inStreams[7];
	}
	else if (BOUNDRY_S.type != BOUNDRY_W.type) //inflow/outflow W
	{
		double vn0 = BOUNDRY_W.vn0;
		domain_D[idx].ro = (domain_D[idx].inStreams[0] + 2.0 * domain_D[idx].inStreams[4] + 2.0 * domain_D[idx].inStreams[2] + 4.0 * domain_D[idx].inStreams[7]) / (1.0 - vn0);
		domain_D[idx].u[1] = 0;
		domain_D[idx].u[0] = vn0;
		domain_D[idx].inStreams[4] = domain_D[idx].inStreams[3];
		domain_D[idx].inStreams[1] = domain_D[idx].inStreams[2] + 2.0 / 3.0 * domain_D[idx].ro * vn0;
		domain_D[idx].inStreams[5] = domain_D[idx].inStreams[7];
		domain_D[idx].inStreams[8] = domain_D[idx].inStreams[7] + 1.0 / 6.0 * domain_D[idx].ro * vn0;

	}
	
}

void LBM_run(LBM_Config* configuration)
{
	LBM_K_Equalibrium << < configuration->gridSize_Entire, configuration->blockSize_Entire >> > (configuration->domain_Device);
	cudaDeviceSynchronize();
	LBM_K_Collsion <<< configuration->gridSize_Entire, configuration->blockSize_Entire >> > (configuration->domain_Device);
	cudaDeviceSynchronize();

	LBM_K_Streaming << < configuration->gridSize_Interior, configuration->blockSize_Interior >> > (configuration->domain_Device);
	LBM_K_Boundry_N << < configuration->gridSize_BoundryN, configuration->blockSize_BoundryN >> > (configuration->domain_Device);
	LBM_K_Boundry_S << < configuration->gridSize_BoundryS, configuration->blockSize_BoundryS >> > (configuration->domain_Device);
	LBM_K_Boundry_E << < configuration->gridSize_BoundryE, configuration->blockSize_BoundryE >> > (configuration->domain_Device);
	LBM_K_Boundry_W << < configuration->gridSize_BoundryW, configuration->blockSize_BoundryW >> > (configuration->domain_Device);
	LBM_K_Boundry_NE << <1, 1 >> > (configuration->domain_Device);
	LBM_K_Boundry_NW << <1, 1 >> > (configuration->domain_Device);
	LBM_K_Boundry_SW << <1, 1 >> > (configuration->domain_Device);
	LBM_K_Boundry_SE << <1, 1 >> > (configuration->domain_Device);
	cudaDeviceSynchronize();

	LBM_K_Variables << < configuration->gridSize_Entire, configuration->blockSize_Entire >> > (configuration->domain_Device);
	cudaDeviceSynchronize();
}

__global__ void LBM_K_Draw(Field* domain_D, float* vbo, Visualisation* visualisation_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX * (threadIdx.y + blockDim.y * blockIdx.y);
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
		NX * (threadIdx.y + blockDim.y * blockIdx.y);

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
		NX * (threadIdx.y + blockDim.y * blockIdx.y);
	
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
		NX * (threadIdx.y + blockDim.y * blockIdx.y);

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
		NX * (threadIdx.y + blockDim.y * blockIdx.y);

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
		LBM_K_Draw_Density << < configuration->gridSize_Entire, configuration->blockSize_Entire >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	case 1:
		LBM_K_Draw_Velocity_Norm << < configuration->gridSize_Entire, configuration->blockSize_Entire >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	case 2:
		LBM_K_Draw_Velocity_Horizontal << < configuration->gridSize_Entire, configuration->blockSize_Entire >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	case 3:
		LBM_K_Draw_Velocity_Vertical << < configuration->gridSize_Entire, configuration->blockSize_Entire >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	default:
		LBM_K_Draw << < configuration->gridSize_Entire, configuration->blockSize_Entire >> > (configuration->domain_Device, devPtr,
			configuration->visualisation_Device);
		break;
	}
	cudaDeviceSynchronize();
}

void setConstantMemory(LBM_Config* config)
{
	cudaMemcpyToSymbol(NX, &config->nx, sizeof(unsigned int));
	cudaMemcpyToSymbol(NY, &config->ny, sizeof(unsigned int));
	double time_coef = 1.0 / config->relaxationTime;
	cudaMemcpyToSymbol(Time_coef, &time_coef, sizeof(double), 0, cudaMemcpyHostToDevice);
	double c1 =  (config->flow == 1) ? 1.0 / (config->cs * config->cs): 0;
	double c3 = (config->flow == 1) ? 1.0 / c1 * 0.5: 0;
	double c2 = (config->flow == 1) ? 1.0 / c3 * c1 : 0;
	cudaMemcpyToSymbol(COEF_1, &c1, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(COEF_2, &c1, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(COEF_3, &c1, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(BOUNDRY_N, &(config->northBC), sizeof(BoundryCondition), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(BOUNDRY_S, &(config->southBC), sizeof(BoundryCondition), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(BOUNDRY_E, &(config->eastBC), sizeof(BoundryCondition), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(BOUNDRY_W, &(config->westBC), sizeof(BoundryCondition), 0, cudaMemcpyHostToDevice);
}

__global__ void LBM_K_Equalibrium(Field* domain_D)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x +
		NX * (threadIdx.y + blockDim.y * blockIdx.y);
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

