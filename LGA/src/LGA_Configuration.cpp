#include "..\inc\LGA_Configuration.h"
#include "device_launch_parameters.h"
#include"..\lib\glew-2.2.0\include\GL\glew.h"


void calculateBlockSize(LBM_Config* config)
{
	config->blockSize_Entire.x = 32;
	config->blockSize_Entire.y = 32;
	config->blockSize_Entire.z = 1;
	dim3 grid = { (config->nx + config->blockSize_Entire.x - 1) / config->blockSize_Entire.x,
			(config->ny + config->blockSize_Entire.y - 1) / config->blockSize_Entire.y, 1 };
	config->gridSize_Entire = grid;

	config->blockSize_Interior.x = 32;
	config->blockSize_Interior.y = 32;
	config->blockSize_Interior.z = 1;
	grid = { (config->nx - 2 + config->blockSize_Interior.x - 1) / config->blockSize_Interior.x,
			(config->ny - 2 + config->blockSize_Interior.y - 1) / config->blockSize_Interior.y, 1 };
	config->gridSize_Interior = grid;

	config->blockSize_BoundryE = 32;
	config->gridSize_BoundryE = (config->ny - 2 + config->blockSize_BoundryE - 1)/ config->blockSize_BoundryE;

	config->blockSize_BoundryW = 32;
	config->gridSize_BoundryW = (config->ny - 2 + config->blockSize_BoundryW - 1)/ config->blockSize_BoundryW;

	config->blockSize_BoundryN = 32;
	config->gridSize_BoundryN = (config->nx - 2 + config->blockSize_BoundryN - 1)/ config->blockSize_BoundryN;

	config->blockSize_BoundryS = 32;
	config->gridSize_BoundryS = (config->nx - 2 + config->blockSize_BoundryS - 1)/ config->blockSize_BoundryS;
}

void pushDomainToDevice(LBM_Config* config)
{
	cudaMalloc(&config->domain_Device, sizeof(Field) * config->nx * config->ny);
	cudaMemcpy(config->domain_Device, config->domain_Host, sizeof(Field) * config->nx * config->ny, cudaMemcpyHostToDevice);
}

void pullDomainFromDevice(LBM_Config* config)
{
	cudaMemcpy(config->domain_Host, config->domain_Device, sizeof(Field) * config->nx * config->ny, cudaMemcpyDeviceToHost);
	cudaFree(config->domain_Device);
	config->domain_Device = NULL;
	cudaDeviceReset();
}

void copyDomainFromDevice(LBM_Config* config)
{
	cudaMemcpy(config->domain_Host, config->domain_Device, sizeof(Field) * config->nx * config->ny, cudaMemcpyDeviceToHost);
}

void printLGAConfig(const LBM_Config* config) {
    printf("LGA_Config Data:\n");
    printf("nx: %u\n", config->nx);
    printf("ny: %u\n", config->ny);
    printf("dx: %lf\n", config->dx);
    printf("dt: %lf\n", config->dt);
    printf("defaultRo: %lf\n", config->defaultRo);
    //printf("fluidRo: %lf\n", config->fluidRo);
    //printf("dynamicViscosity: %lf\n", config->dynamicViscousity);
    printf("relaxationTime: %lf\n", config->relaxationTime);
    printf("cs: %lf\n", config->cs);
}

void calcRelaxationTime(LBM_Config* config)
{
	config->relaxationTime = config->dynamicViscousity / (config->fluidRo * config->cs* config->cs* config->dt) + 0.5;
}

void calcLatticeSoundSpeed(LBM_Config* config)
{
	config->cs = config->dx / (config->dt * sqrt(3));
}

void setBC_NormalVelocity(LBM_Config* config, int boundry, double normalVelocity)
{
	switch (boundry)
	{
	case NORTH:
		config->northBC.type = CONSTANT_NORMAL_SPEED;
		config->northBC.vn0 = normalVelocity;
		config->northBC.d_vn = 0;
		break;
	case SOUTH:
		config->southBC.type = CONSTANT_NORMAL_SPEED;
		config->southBC.vn0 = normalVelocity;
		config->southBC.d_vn = 0;
		break;
	case EAST:
		config->eastBC.type = CONSTANT_NORMAL_SPEED;
		config->eastBC.vn0 = normalVelocity;
		config->eastBC.d_vn = 0;
		break;
	case WEST:
		config->westBC.type = CONSTANT_NORMAL_SPEED;
		config->westBC.vn0 = normalVelocity;
		config->westBC.d_vn = 0;
		break;
	default:
		break;
	}
}

void setBC_NormalVelocity(LBM_Config* config, int boundry, double v0, double v1)
{
	switch (boundry)
	{
	case NORTH:
		config->northBC.type = CONSTANT_NORMAL_SPEED;
		config->northBC.vn0 = v0;
		config->northBC.d_vn = (v1 - v0) / (double)config->nx;
		break;
	case SOUTH:
		config->southBC.type = CONSTANT_NORMAL_SPEED;
		config->southBC.vn0 = v0;
		config->southBC.d_vn = (v1 - v0) / (double)config->nx;
		break;
	case WEST:
		config->westBC.type = CONSTANT_NORMAL_SPEED;
		config->westBC.vn0 = v0;
		config->westBC.d_vn = (v1 - v0) / (double)config->ny;
		break;
	case EAST:
		config->eastBC.type = CONSTANT_NORMAL_SPEED;
		config->eastBC.vn0 = v0;
		config->eastBC.d_vn = (v1 - v0) / (double)config->ny;
		break;
	}
}

void setBC_BouncyBack(LBM_Config* config, int boundry)
{
	switch (boundry)
	{
	case NORTH:
		config->northBC.type = BOUNCY_BACK;
		break;
	case SOUTH:
		config->southBC.type = BOUNCY_BACK;
		break;
	case EAST:
		config->eastBC.type = BOUNCY_BACK;
		break;
	case WEST:
		config->westBC.type = BOUNCY_BACK;
		break;
	default:
		break;
	}
}
void setBC_Symmetry(LBM_Config* config, int boundry)
{
	switch (boundry)
	{
	case NORTH:
		config->northBC.type = SYMMETRY;
		break;
	case SOUTH:
		config->southBC.type = SYMMETRY;
		break;
	case EAST:
		config->eastBC.type = SYMMETRY;
		break;
	case WEST:
		config->westBC.type = SYMMETRY;
		break;
	default:
		break;
	}
}

void setBC_ConstantVelocity(LBM_Config* config, int boundry, double vn0, double vn1, double vt0, double vt1)
{
	switch (boundry)
	{
	case NORTH:
		config->northBC.type = CONSTANT_VELOCITY;
		config->northBC.vn0 = vn0;
		config->northBC.d_vn = (vn1 - vn0) / (double)config->nx;
		config->northBC.vt0 = vt0;
		config->northBC.d_vt = (vt1 - vt0) / (double)config->nx;
		break;

	case SOUTH:
		config->southBC.type = CONSTANT_VELOCITY;
		config->southBC.vn0 = vn0;
		config->southBC.d_vn = (vn1 - vn0) / (double)config->nx;
		config->southBC.vt0 = vt0;
		config->southBC.d_vt = (vt1 - vt0) / (double)config->nx;
		break;
	case WEST:
		config->westBC.type = CONSTANT_VELOCITY;
		config->westBC.vn0 = vn0;
		config->westBC.d_vn = (vn1 - vn0) / (double)config->ny;
		config->westBC.vt0 = vt0;
		config->westBC.d_vt = (vt1 - vt0) / (double)config->ny;
		break;
	case EAST:
		config->eastBC.type = CONSTANT_VELOCITY;
		config->eastBC.vn0 = vn0;
		config->eastBC.d_vn = (vn1 - vn0) / (double)config->ny;
		config->eastBC.vt0 = vt0;
		config->eastBC.d_vt = (vt1 - vt0) / (double)config->ny;
		break;
	}
}

void initLBM(LBM_Config** config, double length, int nx, int ny, int flow, double mach)
{
	(*config) = (LBM_Config*)malloc(sizeof(LBM_Config));
	(* config)->domain_Device = NULL;
	(* config)->domain_Host = NULL;
	(* config)->isWorking = 0;
	(*config)->shutDown = 0;
	(*config)->nx = nx;
	(*config)->ny = ny;
	(*config)->flow = flow;
	(*config)->dx = length / (double)nx;
	(*config)->dt = (*config)->dx * mach;
}