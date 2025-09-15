#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <gpu/cuda_kernel.cuh>

__global__ void computeForcesKernelGridOptimized(
    int N,
    const float* posX_sorted, const float* posY_sorted,
    const float* velX_sorted, const float* velY_sorted,
    const float* influence_sorted,
    const int* type_sorted,
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

    int sortedIdx = particleArrayIndices[i];  // usa l'indice ordinato
    float px = posX_sorted[sortedIdx];
    float py = posY_sorted[sortedIdx];
    float vx = velX_sorted[sortedIdx];
    float vy = velY_sorted[sortedIdx];
    float wInfluence = influence_sorted[sortedIdx];
    int t = type_sorted[sortedIdx];

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
            int j = particleArrayIndices[jIdx];  // indice ordinato del vicino
            if (i == j) continue;

            float neighX = posX_sorted[j];
            float neighY = posY_sorted[j];
            float neighVX = velX_sorted[j];
            float neighVY = velY_sorted[j];
            float neighInfluence = influence_sorted[j];

            float dx = neighX - px;
            float dy = neighY - py;
            float dist = sqrtf(dx * dx + dy * dy);

            if (dist < cohesionDistance) {
                cohX += neighX;
                cohY += neighY;
                neighborCount++;
            }

            if (dist < separationDistance && dist > 0.0f) {
                sepX += (px - neighX) / dist;
                sepY += (py - neighY) / dist;
            }

            if (dist < alignmentDistance) {
                aliX += neighVX * neighInfluence;
                aliY += neighVY * neighInfluence;
                totalWeight += neighInfluence;
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

    outVelChangeX[sortedIdx] = cohX + sepX + aliX + borderX;
    outVelChangeY[sortedIdx] = cohY + sepY + aliY + borderY;
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

__global__ void kernApplyVelocityChangeSorted(
    int N,
    const float* velChangeX_sorted, const float* velChangeY_sorted,
    float* posX, float* posY,
    float* velX, float* velY,
    const int* particleArrayIndices,
    float dt, float slowDownFactor, float maxSpeed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int origIdx = particleArrayIndices[i];

    // Applica la variazione di velocità con il fattore di rallentamento
    velX[origIdx] += velChangeX_sorted[i] * slowDownFactor;
    velY[origIdx] += velChangeY_sorted[i] * slowDownFactor;

    // Limita la velocità
    float speed = sqrtf(velX[origIdx] * velX[origIdx] + velY[origIdx] * velY[origIdx]);
    if (speed > maxSpeed) {
        velX[origIdx] = (velX[origIdx] / speed) * maxSpeed;
        velY[origIdx] = (velY[origIdx] / speed) * maxSpeed;
    }

    // Aggiorna la posizione
    posX[origIdx] += velX[origIdx] * dt;
    posY[origIdx] += velY[origIdx] * dt;
}

__global__ void kernComputeRotations(int N, const float* velX, const float* velY, float* rotations) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float angle = atan2f(velY[i], velX[i]);   // radianti
    angle = angle * (180.0f / 3.14159265f);   // conversione in gradi
    rotations[i] = angle + 270.0f;            // offset per far “puntare” il modello in avanti
}

__global__ void kernIntegratePositions(int N, float dt,
    float* posX, float* posY,
    const float* velX, const float* velY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    posX[i] += velX[i] * dt;
    posY[i] += velY[i] * dt;
}

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
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int srcIdx = particleArrayIndices[i]; // indice originale

    posX_sorted[i] = posX[srcIdx];
    posY_sorted[i] = posY[srcIdx];

    velX_sorted[i] = velX[srcIdx];
    velY_sorted[i] = velY[srcIdx];

    scale_sorted[i] = scale[srcIdx];
    influence_sorted[i] = influence[srcIdx];

    type_sorted[i] = type[srcIdx];

    colorR_sorted[i] = colorR[srcIdx];
    colorG_sorted[i] = colorG[srcIdx];
    colorB_sorted[i] = colorB[srcIdx];

    velChangeX_sorted[i] = velChangeX[srcIdx];
    velChangeY_sorted[i] = velChangeY[srcIdx];
}






