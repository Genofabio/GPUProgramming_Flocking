#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

// ============================================================
// 1. Utility kernels per la griglia (hashing e ordinamento)
// ============================================================

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

__global__ void kernReorderData(
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
    float* velChangeX_sorted, float* velChangeY_sorted
);

// ============================================================
// 2. Kernel principale: calcolo forze tra boids
// ============================================================

// Versione aggressiva ottimizzata con Shared Memory + Grid
__global__ void computeForcesKernelAggressive(
    int N,
    const float* posX_sorted, const float* posY_sorted,
    const float* velX_sorted, const float* velY_sorted,
    const float* influence_sorted,
    const int* gridCellStartIndices,
    const int* gridCellEndIndices,
    int gridResolutionX, int gridResolutionY,
    float cellWidth,
    float cohesionDistance, float cohesionScale,
    float separationDistance, float separationScale,
    float alignmentDistance, float alignmentScale,
    float width, float height, float borderAlertDistance,
    float* outVelChangeX, float* outVelChangeY
);

// ============================================================
// 3. Integrazione e aggiornamento stato boids
// ============================================================

__global__ void kernApplyVelocityChangeSorted(
    int N,
    const float* velChangeX_sorted, const float* velChangeY_sorted,
    float* posX, float* posY,
    float* velX, float* velY,
    const int* particleArrayIndices,
    float dt, float slowDownFactor, float maxSpeed
);

__global__ void kernIntegratePositions(
    int N, float dt,
    float* posX, float* posY,
    const float* velX, const float* velY
);

__global__ void kernComputeRotations(
    int N,
    const float* velX, const float* velY,
    float* rotations
);

// ============================================================
// 4. Copia dati per il rendering
// ============================================================

__global__ void copyRenderDataKernel(
    int N,
    const float* posX, const float* posY,
    const float* rotations,
    const float* colorR, const float* colorG, const float* colorB,
    const float* scale,
    glm::vec2* outPositions,
    float* outRotations,
    glm::vec3* outColors,
    float* outScales
);

