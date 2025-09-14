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
    float alignmentDistance, float alignmentScale,
    float width, float height, float borderAlertDistance
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px = posX[i];
    float py = posY[i];

    // --- componenti distinte ---
    float cohX = 0.0f, cohY = 0.0f;
    float sepX = 0.0f, sepY = 0.0f;
    float aliX = 0.0f, aliY = 0.0f;

    int neighborCount = 0;
    float totalWeight = 0.0f;

    // --- Loop sui vicini ---
    for (int j = 0; j < N; j++) {
        if (i == j) continue;

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

    // Normalizzazione coesione
    if (neighborCount > 0) {
        cohX = (cohX / neighborCount - px) * cohesionScale;
        cohY = (cohY / neighborCount - py) * cohesionScale;
    }

    // Normalizzazione allineamento
    if (totalWeight > 0.0f) {
        aliX = (aliX / totalWeight) * alignmentScale;
        aliY = (aliY / totalWeight) * alignmentScale;
    }

    // Scaling separazione
    sepX *= separationScale;
    sepY *= separationScale;

    // --- Border Repulsion ---
    float borderX = 0.0f, borderY = 0.0f;
    if (px < borderAlertDistance) borderX += (borderAlertDistance - px);
    if ((width - px) < borderAlertDistance) borderX -= (borderAlertDistance - (width - px));
    if (py < borderAlertDistance) borderY += (borderAlertDistance - py);
    if ((height - py) < borderAlertDistance) borderY -= (borderAlertDistance - (height - py));
    borderX *= 0.2f;
    borderY *= 0.2f;

    // --- Somma finale ---
    outVelChangeX[i] = cohX + sepX + aliX + borderX;
    outVelChangeY[i] = cohY + sepY + aliY + borderY;
}
