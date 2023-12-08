#pragma once
#include <stdlib.h>
#include<stdio.h>
#include<string.h>

#define NX 1024
#define NY 1024

#define EMPTY_SPACE 0
#define WALL 1

#define POS_LOCATION 0
#define COLOR_LOCATION 1
#define DIMENSION 2
#define COLOR_ATTRIB 3
#define VERTEX_SIZE 5

#define MAX_INTENSITY 1.0f

#define DEFAULT_VERTEX_SHADER "C:/Users/piero/source/repos/LGA/LGA/resources/default.vert"
#define DEFAULT_FRAGMENT_SHADER "C:/Users/piero/source/repos/LGA/LGA/resources/default.fragment"

typedef struct 
{
	int type;
	float inStreams[4];
	//float eqStreams[Q];
	float outStreams[4];
	float C;
} Field;

// Failures - information 

/*
#define DEVICE_DOMAIN_NOT_NULL "Initializing - device domain is not null!"
#define CONFIGURATION_IS_NULL "Configuration structure is null!"
#define SUCCES "SUCCES";
#define MAX_SIZE 64

typedef struct 
{
	char* file;
	char* cause;
} Failure;

typedef Failure* Failure_t;

Failure_t generateFailureMessage(const char* file, const char* cause)
{
	Failure_t message = (Failure_t)malloc(sizeof(Failure));
	message->file = (char*)malloc(sizeof(char) * strlen(file));
	message->cause = (char*)malloc(sizeof(char) * strlen(cause));
	strcpy(message->file, file);
	strcpy(message->file, file);
	return message;
}

void readFailureMessage(Failure_t message)
{
	printf("\nFILE: %s\n", message->file);
	printf("\nCAUSE: %s\n", message->cause);
	free(message);
}
*/