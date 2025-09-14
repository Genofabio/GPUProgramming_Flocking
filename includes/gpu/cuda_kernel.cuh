#pragma once
#include <cuda_runtime.h>


__global__ void computeForcesKernelGridOptimized(
    int N,
    const float* posX, const float* posY,
    const float* velX, const float* velY,
    const float* influence,
    const int* type,
    const int* particleArrayIndices,
    const int* particleGridIndices,
    const int* gridCellStartIndices,
    const int* gridCellEndIndices,
    int gridResolutionX,
    int gridResolutionY,
    float cellWidth,
    float cohesionDistance, float cohesionScale,
    float separationDistance, float separationScale,
    float alignmentDistance, float alignmentScale,
    float width, float height, float borderAlertDistance,
    float* outVelChangeX, float* outVelChangeY // <<< aggiunto >>>
);


__global__ void kernComputeIndices(
    int N,
    float* posX, float* posY,
    int* particleGridIndices,
    int* particleArrayIndices,
    int gridResolutionX, int gridResolutionY,
    float gridMinX, float gridMinY,
    float cellWidth
);

__global__ void kernIdentifyCellStartEnd(
    int N,
    int* particleGridIndices,
    int* gridCellStartIndices,
    int* gridCellEndIndices
);

__global__ void kernApplyVelocityChange(
    int N,
    float* posX, float* posY,
    float* velX, float* velY,
    const float* velChangeX, const float* velChangeY,
    float dt, float slowDownFactor, float maxSpeed);
__global__ void kernComputeRotations(int N, const float* velX, const float* velY, float* rotations);
__global__ void kernIntegratePositions(int N, float dt,
    float* posX, float* posY,
	const float* velX, const float* velY);