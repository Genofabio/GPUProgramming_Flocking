#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <gpu/cuda_kernel.cuh>

__global__ void computeForcesKernel(
    int N,
    const float* posX, const float* posY,
    const float* velX, const float* velY,
    const float* influence,
    const int* type,
    float* outVelChangeX,
    float* outVelChangeY,
    float cohesionDistance, float cohesionScale,
    float separationDistance, float separationScale,
    float alignmentDistance, float alignmentScale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (type[i] != 0) return; // solo PREY

    float px = posX[i];
    float py = posY[i];

    float perceivedCenterX = 0.0f;
    float perceivedCenterY = 0.0f;
    float perceivedVelX = 0.0f;
    float perceivedVelY = 0.0f;
    float sepX = 0.0f;
    float sepY = 0.0f;
    float totalWeight = 0.0f;
    int neighborCount = 0;

    // Loop unico per tutti i vicini
    for (int j = 0; j < N; j++) {
        if (i == j) continue;
        if (type[j] != 0) continue; // solo PREY

        float dx = posX[j] - px;
        float dy = posY[j] - py;
        float dist = sqrtf(dx * dx + dy * dy);

        // Coesione
        if (dist < cohesionDistance) {
            perceivedCenterX += posX[j];
            perceivedCenterY += posY[j];
            neighborCount++;
        }

        // Separazione
        if (dist < separationDistance && dist > 0.0f) {
            sepX += (px - posX[j]) / dist;
            sepY += (py - posY[j]) / dist;
        }

        // Allineamento
        if (dist < alignmentDistance) {
            float w = influence[j];
            perceivedVelX += velX[j] * w;
            perceivedVelY += velY[j] * w;
            totalWeight += w;
        }
    }

    // Coesione finale
    if (neighborCount > 0) {
        perceivedCenterX = (perceivedCenterX / neighborCount - px) * cohesionScale;
        perceivedCenterY = (perceivedCenterY / neighborCount - py) * cohesionScale;
    }

    // Allineamento finale
    if (totalWeight > 0.0f) {
        perceivedVelX = (perceivedVelX / totalWeight) * alignmentScale;
        perceivedVelY = (perceivedVelY / totalWeight) * alignmentScale;
    }

    // Separazione
    sepX *= separationScale;
    sepY *= separationScale;

    // Somma finale
    outVelChangeX[i] = perceivedCenterX + sepX + perceivedVelX;
    outVelChangeY[i] = perceivedCenterY + sepY + perceivedVelY;
}
