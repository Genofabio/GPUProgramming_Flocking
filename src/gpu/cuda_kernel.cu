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

    float px = posX[i];
    float py = posY[i];

    // --- componenti distinte ---
    float cohX = 0.0f, cohY = 0.0f;  // coesione
    float sepX = 0.0f, sepY = 0.0f;  // separazione
    float aliX = 0.0f, aliY = 0.0f;  // allineamento

    int neighborCount = 0;
    float totalWeight = 0.0f;

    // loop sui vicini
    for (int j = 0; j < N; j++) {
        if (i == j) continue;
        //if (type[j] != 0) continue; // solo PREY

        float dx = posX[j] - px;
        float dy = posY[j] - py;
        float dist = sqrtf(dx * dx + dy * dy);

        // --- coesione ---
        if (dist < cohesionDistance) {
            cohX += posX[j];
            cohY += posY[j];
            neighborCount++;
        }

        // --- separazione ---
        if (dist < separationDistance && dist > 0.0f) {
            sepX += (px - posX[j]) / dist;
            sepY += (py - posY[j]) / dist;
        }

        // --- allineamento ---
        if (dist < alignmentDistance) {
            float w = influence[j];
            aliX += velX[j] * w;
            aliY += velY[j] * w;
            totalWeight += w;
        }
    }

    // normalizzazione coesione
    if (neighborCount > 0) {
        cohX = (cohX / neighborCount - px);
        cohY = (cohY / neighborCount - py);
    }

    // normalizzazione allineamento
    if (totalWeight > 0.0f) {
        aliX = (aliX / totalWeight);
        aliY = (aliY / totalWeight);
    }

    if (i < 10) {
        printf("Boid %d: coh=(%.6f, %.6f) sep=(%.6f, %.6f) ali=(%.6f, %.6f) cohesionScale=%.6f\n",
            i, cohX * cohesionScale, cohY * cohesionScale,
            sepX * separationScale, sepY * separationScale,
            aliX * alignmentScale, aliY * alignmentScale,
            cohesionScale);
    }

    // --- somma finale ---
    outVelChangeX[i] = cohX * cohesionScale + sepX * separationScale + aliX * alignmentScale;
    outVelChangeY[i] = cohY * cohesionScale + sepY * separationScale + aliY * alignmentScale;
}
