#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <core/BoidParams.h>

// Copia i parametri di simulazione sulla GPU
void setSimulationParamsOnGPU(int width, int height, const BoidParams& params);

// 1. Utility kernels per la griglia (hashing, ordinamento, reorder)
__global__ void kernComputeGridIndices(
    int N,
    float* posX, float* posY,
    int* particleGridIndices,
    int* particleArrayIndices,
    int gridResolutionX, int gridResolutionY,
    float gridMinX, float gridMinY,
    float cellWidth);

__global__ void kernIdentifyCellStartEnd(
    int N,
    int* particleGridIndices,
    int* gridCellStartIndices,
    int* gridCellEndIndices);

__global__ void kernReorderBoidData(
    int N,
    const float* posX, const float* posY,
    const float* velX, const float* velY,
    const float* scale, const float* influence,
    const int* type,
    const float* colorR, const float* colorG, const float* colorB,
    const float* velChangeX, const float* velChangeY,
    const int* particleArrayIndices, // permutazione ordinata
    float* posX_sorted, float* posY_sorted,
    float* velX_sorted, float* velY_sorted,
    float* scale_sorted, float* influence_sorted,
    int* type_sorted,
    float* colorR_sorted, float* colorG_sorted, float* colorB_sorted,
    float* velChangeX_sorted, float* velChangeY_sorted);

// 2. Forze tra boids (cohesion, separation, alignment, muri)
__global__ void kernComputeBoidNeighborForces(
    int N,
    const float* posX_sorted, const float* posY_sorted,
    const float* velX_sorted, const float* velY_sorted,
    const float* influence_sorted,
    const int* gridCellStartIndices,
    const int* gridCellEndIndices,
    int gridResolutionX, int gridResolutionY,
    float cellWidth,
    const int* type_sorted,
    float* outVelChangeX, float* outVelChangeY);

__global__ void kernComputeBorderWallForces(
    int N,
    const float* posX_sorted, const float* posY_sorted,
    const float* velX_sorted, const float* velY_sorted,
    int numWalls,
    const float2* wallPositions,
    float* outVelChangeX, float* outVelChangeY);

// 3. Forze speciali (leader, predatori)
__global__ void kernLeaderFollowForces(
    int N,
    const float* posX_sorted,
    const float* posY_sorted,
    const float* velX_sorted,
    const float* velY_sorted,
    const int* type_sorted,
    float* velChangeX_sorted,
    float* velChangeY_sorted);

__global__ void kernPredatorPreyForces(
    int N,
    const float* posX_sorted,
    const float* posY_sorted,
    const int* type_sorted,
    float* velChangeX_sorted,
    float* velChangeY_sorted);

// 4. Integrazione e aggiornamento stato
__global__ void kernApplyVelocityChange(
    int N,
    const float* velChangeX_sorted, const float* velChangeY_sorted,
    float* posX, float* posY,
    float* velX, float* velY,
    const int* particleArrayIndices,
    const int* type_sorted,
    float dt);

__global__ void kernIntegratePositions(
    int N, float dt,
    float* posX, float* posY,
    const float* velX, const float* velY);

__global__ void kernComputeRotations(
    int N,
    const float* velX, const float* velY,
    float* rotations);

__global__ void kernSumBuffers(
    int N,
    float* outX, float* outY,
    const float* bufX1, const float* bufY1,
    const float* bufX2, const float* bufY2,
    const float* bufX3, const float* bufY3,
    const float* bufX4, const float* bufY4
);

// 5. Copia dati per il rendering
__global__ void kernCopyRenderData(
    int N,
    const float* posX, const float* posY,
    const float* rotations,
    const float* colorR, const float* colorG, const float* colorB,
    const float* scale,
    glm::vec2* outPositions,
    float* outRotations,
    glm::vec3* outColors,
    float* outScales);
