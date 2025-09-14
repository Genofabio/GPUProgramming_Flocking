#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <gpu/cuda_kernel.cuh>

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
    float* outVelChangeX, float* outVelChangeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px = posX[i];
    float py = posY[i];

    float cohX = 0.0f, cohY = 0.0f;
    float sepX = 0.0f, sepY = 0.0f;
    float aliX = 0.0f, aliY = 0.0f;

    int neighborCount = 0;
    float totalWeight = 0.0f;

    int col = int(px / cellWidth);
    int row = int(py / cellWidth);

    // clamp esplicito
    if (col < 0) col = 0;
    if (col >= gridResolutionX) col = gridResolutionX - 1;
    if (row < 0) row = 0;
    if (row >= gridResolutionY) row = gridResolutionY - 1;

    // --- logica delle 4 celle ---
    float localX = (px - col * cellWidth) / cellWidth;
    float localY = (py - row * cellWidth) / cellWidth;

    int worldDx = (localX > 0.5f) ? 1 : -1;
    int worldDy = (localY > 0.5f) ? 1 : -1;

    int dr[4] = { 0, worldDy, 0, worldDy };
    int dc[4] = { 0, 0, worldDx, worldDx };

    for (int q = 0; q < 4; ++q) {
        int neighRow = row + dr[q];
        int neighCol = col + dc[q];

        if (neighRow < 0 || neighRow >= gridResolutionY) continue;
        if (neighCol < 0 || neighCol >= gridResolutionX) continue;

        int neighCell = neighCol + neighRow * gridResolutionX;
        int startIdx = gridCellStartIndices[neighCell];
        int endIdx = gridCellEndIndices[neighCell];
        if (startIdx == -1) continue;

        for (int jIdx = startIdx; jIdx <= endIdx; ++jIdx) {
            int j = particleArrayIndices[jIdx];
            if (i == j) continue;

            float dx = posX[j] - px;
            float dy = posY[j] - py;
            float dist = sqrtf(dx * dx + dy * dy);

            if (dist < cohesionDistance) {
                cohX += posX[j];
                cohY += posY[j];
                neighborCount++;
            }

            if (dist < separationDistance && dist > 0.0f) {
                sepX += (px - posX[j]) / dist;
                sepY += (py - posY[j]) / dist;
            }

            if (dist < alignmentDistance) {
                float w = influence[j];
                aliX += velX[j] * w;
                aliY += velY[j] * w;
                totalWeight += w;
            }
        }
    }

    if (neighborCount > 0) {
        cohX = (cohX / neighborCount - px) * cohesionScale;
        cohY = (cohY / neighborCount - py) * cohesionScale;
    }

    if (totalWeight > 0.0f) {
        aliX = (aliX / totalWeight) * alignmentScale;
        aliY = (aliY / totalWeight) * alignmentScale;
    }

    sepX *= separationScale;
    sepY *= separationScale;

    // Border
    float borderX = 0.0f, borderY = 0.0f;
    if (px < borderAlertDistance) borderX += (borderAlertDistance - px);
    if ((width - px) < borderAlertDistance) borderX -= (borderAlertDistance - (width - px));
    if (py < borderAlertDistance) borderY += (borderAlertDistance - py);
    if ((height - py) < borderAlertDistance) borderY -= (borderAlertDistance - (height - py));
    borderX *= 0.2f;
    borderY *= 0.2f;

    outVelChangeX[i] = cohX + sepX + aliX + borderX;
    outVelChangeY[i] = cohY + sepY + aliY + borderY;
}


__global__ void kernComputeIndices(
    int N,
    float* posX, float* posY,
    int* particleGridIndices,
    int* particleArrayIndices,
    int gridResolutionX, int gridResolutionY,
    float gridMinX, float gridMinY,
    float cellWidth)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int cellX = (int)floorf((posX[i] - gridMinX) / cellWidth);
    int cellY = (int)floorf((posY[i] - gridMinY) / cellWidth);

    // clamp esplicito
    if (cellX < 0) cellX = 0;
    if (cellX >= gridResolutionX) cellX = gridResolutionX - 1;
    if (cellY < 0) cellY = 0;
    if (cellY >= gridResolutionY) cellY = gridResolutionY - 1;

    int cellIndex = cellX + cellY * gridResolutionX;

    particleGridIndices[i] = cellIndex;
    particleArrayIndices[i] = i;
}


__global__ void kernIdentifyCellStartEnd(
    int N,
    int* particleGridIndices,
    int* gridCellStartIndices,
    int* gridCellEndIndices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int currentCell = particleGridIndices[i];
    int prevCell = (i > 0) ? particleGridIndices[i - 1] : -1;
    int nextCell = (i < N - 1) ? particleGridIndices[i + 1] : -1;

    if (currentCell != prevCell) {
        gridCellStartIndices[currentCell] = i;
    }
    if (currentCell != nextCell) {
        gridCellEndIndices[currentCell] = i;
    }
}
