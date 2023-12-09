#include"..\inc\LGA_Simulation.h"
#include<GL\glew.h>
#include<glfw3.h>
#include<stdio.h>
#include<Windows.h>
#include<ctime>

// CPU THREADS //

CRITICAL_SECTION accessSimulationState;
CRITICAL_SECTION accessShutDown;

int isStillWorking(LGA_Config* configuration)
{
	int result = 0;
	EnterCriticalSection(&accessSimulationState);
	result = configuration->isWorking;
	LeaveCriticalSection(&accessSimulationState);
	return result;
}

int shutDown(LGA_Config* configuration)
{
	int result = 0;
	EnterCriticalSection(&accessShutDown);
	result = configuration->shutDown;
	LeaveCriticalSection(&accessShutDown);
	return result;
}

DWORD WINAPI RunSimulationParallel(LPVOID configuration)
{
	printf("\nSimulation start\n");
	LGA_Config* ptr = (LGA_Config*)configuration;
	while (shutDown(ptr) == 0)
	{
		while (isStillWorking(ptr) == 0) Sleep(1);
		LGA_run(ptr);

		EnterCriticalSection(&accessSimulationState);
		ptr->isWorking = 0;
		LeaveCriticalSection(&accessSimulationState);
	}
	ptr->shutDown = -1;
	printf("\nSimulation shut down\n");
	return 0;
}

//

LGA_Config* LGA_setup(const char* setupFile)
{
	return NULL;
}

int createEmptySpace(LGA_Config* config, unsigned int nx, unsigned int ny)
{
	if (config == NULL)
	{
		return -1;
	}

	if (config->domain_Device != nullptr)
	{
		return -1;
	}

	if (config->domain_Host != nullptr)
	{
		free(config->domain_Host);
		config->domain_Host = NULL;
	}
	
	config->domain_Host = (Field*)malloc(sizeof(Field) * nx * ny);
	config->nx = nx;
	config->ny = ny;
	for (unsigned int y = 0; y < ny; y++)
	{
		for (unsigned int x = 0; x < nx; x++)
		{
			config->domain_Host[x + y * nx].type = EMPTY_SPACE;
			config->domain_Host[x + y * nx].ro = 0;
			config->domain_Host[x + y * nx].u[0] = 0;
			config->domain_Host[x + y * nx].u[1] = 0;
			for (int i = 0; i < 9; i++)
			{
				config->domain_Host[x + y * nx].inStreams[i] = 0;
				config->domain_Host[x + y * nx].outStreams[i] = 0;
				config->domain_Host[x + y * nx].eqStreams[i] = 0;
			}
			
		}
	}
	return 0;
}

void drawWall(LGA_Config* config, unsigned int x0, unsigned int width, unsigned int y0, unsigned int height)
{
	unsigned int yend = y0 + height;
	unsigned int xend = x0 + width;
	for (unsigned int y = y0; y < yend; y++)
	{
		for (unsigned int x = x0; x < xend; x++)
		{
			config->domain_Host[x + y * config->nx].type = WALL;
		}
	}
}


void LGA_simulation(LGA_Config* configuration)
{
	// Set GLFW error callback
	// Configure GLFW
	if (!glfwInit()) {
		printf("Failed to initialize GLFW");
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create a window
	int width = 800;
	int height = 600;
	const char* title = "LGA";

	GLenum err = glewInit();
	if (err != GLEW_OK) {
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		// Handle GLEW initialization error
	}

	GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
	if (!window) {
		printf("\nFailed to create GLFW window\n");
		glfwTerminate();
		return;
	}
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// Make the window's context current
	glfwMakeContextCurrent(window);

	// Set up //
	calculateBlockSize(configuration);
	pushDomainToDevice(configuration); 
	setConstantMemory(configuration);

	// Graphics resources set up //
	Graphics_Objects* graphicsRes = createGraphicsObjects(configuration);

	// Threads WINAPI //
	HANDLE simulation = NULL;
	DWORD pid_simulation;
	InitializeCriticalSection(&accessSimulationState);
	InitializeCriticalSection(&accessShutDown);

	// Thread creation //
	simulation = CreateThread(NULL, 0, RunSimulationParallel, configuration, 0, &pid_simulation);
	// Detach //
	CloseHandle(simulation);
	// Main loop
	while (!glfwWindowShouldClose(window)) {
		// Process input
		glfwPollEvents();
		//Sleep(10);
		if(isStillWorking(configuration) == 0)
		{
			// Update VBO //
			mapCudaResources(graphicsRes);
			LGA_draw(configuration, graphicsRes->devPtr);
			unmapCudaGraphicResources(graphicsRes);
			// Start new work //
			EnterCriticalSection(&accessSimulationState);
			configuration->isWorking = 1;
			LeaveCriticalSection(&accessSimulationState);
		}

		// Swap front and back buffers
		glClearColor(0.0f, 0.4f, 0.5f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glDrawArrays(GL_POINTS, 0, configuration->nx*configuration->ny);
		glfwSwapBuffers(window);
	}
	EnterCriticalSection(&accessShutDown);
	configuration->shutDown = 1;
	LeaveCriticalSection(&accessShutDown);

	// Terminate GLFW
	glfwTerminate();

	while (shutDown(configuration) != -1)Sleep(10);
	// Close critical section //
	DeleteCriticalSection(&accessSimulationState);
	DeleteCriticalSection(&accessShutDown);

	// Release graphic resources //
	//releaseCudaGraphicResources(graphicsRes);
	releaseOpenGLResources(graphicsRes);
	pullDomainFromDevice(configuration);
	free(graphicsRes);
}

void randomInitialState(LGA_Config* config, float C_max, unsigned int prob, unsigned int x0, unsigned int width, unsigned int y0, unsigned int height)
{
	unsigned int counter = 0;
	unsigned int size = config->nx * config->ny;
	for(unsigned int y = y0; y < y0 + height; y++)
	for(unsigned int x = x0; x < x0 + width; x++)
	{ 
		int idx = x + y*config->nx; 
		if (config->domain_Host[idx].type == WALL) continue;

		if ((rand() % 100) > prob)
		{
			config->domain_Host[idx].ro = 1; (float)rand() / (float)RAND_MAX * C_max;
			counter++;
		}
	}
}