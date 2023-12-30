#pragma once

#include"LGA_kernels.cuh"
#include"..\inc\LGA_Graphics.h"


LGA_Config* LGA_setup(const char* setupFile);
void LGA_simulation(LGA_Config* configuration);

int createEmptySpace(LGA_Config* config, unsigned int nx, unsigned int ny);
void drawWall(LGA_Config* config, unsigned int x0, unsigned int width, unsigned int y0, unsigned int height);

void randomInitialState(LGA_Config* config, unsigned int x0, unsigned int width, unsigned int y0, unsigned int height);


