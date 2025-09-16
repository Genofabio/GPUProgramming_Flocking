#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <gpu/cuda_kernel.cuh>
#include <algorithm>

//__global__ void computeForcesKernelGridOptimized(
//    int N,
//    const float* posX_sorted, const float* posY_sorted,
//    const float* velX_sorted, const float* velY_sorted,
//    const float* influence_sorted,
//    const int* type_sorted,
//    const int* particleArrayIndices,
//    const int* particleGridIndices,
//    const int* gridCellStartIndices,
//    const int* gridCellEndIndices,
//    int gridResolutionX,
//    int gridResolutionY,
//    float cellWidth,
//    float cohesionDistance, float cohesionScale,
//    float separationDistance, float separationScale,
//    float alignmentDistance, float alignmentScale,
//    float width, float height, float borderAlertDistance,
//    float* outVelChangeX, float* outVelChangeY
//)
//{
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i >= N) return;
//
//    // --- Shared memory dinamica ---
//    extern __shared__ float shMemory[];
//    float* shPosX = shMemory;                     // blockDim.x elementi
//    float* shPosY = shPosX + blockDim.x;
//    float* shVelX = shPosY + blockDim.x;
//    float* shVelY = shVelX + blockDim.x;
//    float* shInfluence = shVelY + blockDim.x;
//    int* shType = (int*)(shInfluence + blockDim.x);
//
//    int tid = threadIdx.x;
//    int blockStart = blockIdx.x * blockDim.x;
//
//    // --- Caricamento collaborativo in shared memory ---
//    for (int idx = tid; idx < blockDim.x && (blockStart + idx) < N; idx += blockDim.x) {
//        int sortedIdx = particleArrayIndices[blockStart + idx];
//        shPosX[idx] = posX_sorted[sortedIdx];
//        shPosY[idx] = posY_sorted[sortedIdx];
//        shVelX[idx] = velX_sorted[sortedIdx];
//        shVelY[idx] = velY_sorted[sortedIdx];
//        shInfluence[idx] = influence_sorted[sortedIdx];
//        shType[idx] = type_sorted[sortedIdx];
//    }
//    __syncthreads();
//
//    // --- Variabili locali ---
//    int sortedIdx = particleArrayIndices[i];
//    float px = shPosX[tid];
//    float py = shPosY[tid];
//
//    float cohX = 0.0f, cohY = 0.0f;
//    float sepX = 0.0f, sepY = 0.0f;
//    float aliX = 0.0f, aliY = 0.0f;
//
//    int neighborCount = 0;
//    float totalWeight = 0.0f;
//
//    int col = int(px / cellWidth);
//    int row = int(py / cellWidth);
//    col = (col < 0) ? 0 : (col >= gridResolutionX ? gridResolutionX - 1 : col);
//    row = (row < 0) ? 0 : (row >= gridResolutionY ? gridResolutionY - 1 : row);
//
//    float localX = (px - col * cellWidth) / cellWidth;
//    float localY = (py - row * cellWidth) / cellWidth;
//    int worldDx = (localX > 0.5f) ? 1 : -1;
//    int worldDy = (localY > 0.5f) ? 1 : -1;
//
//    int dr[4] = { 0, worldDy, 0, worldDy };
//    int dc[4] = { 0, 0, worldDx, worldDx };
//
//    for (int q = 0; q < 4; ++q) {
//        int neighRow = row + dr[q];
//        int neighCol = col + dc[q];
//        if (neighRow < 0 || neighRow >= gridResolutionY) continue;
//        if (neighCol < 0 || neighCol >= gridResolutionX) continue;
//
//        int neighCell = neighCol + neighRow * gridResolutionX;
//        int startIdx = gridCellStartIndices[neighCell];
//        int endIdx = gridCellEndIndices[neighCell];
//        if (startIdx == -1) continue;
//
//        for (int jIdx = startIdx; jIdx <= endIdx; ++jIdx) {
//            int j = particleArrayIndices[jIdx]; // indice globale del vicino
//            if (i == j) continue;
//
//            float neighX, neighY, neighVX, neighVY, neighInfluence;
//
//            int jThread = j - blockStart;
//            if (jThread >= 0 && jThread < blockDim.x) {
//                // vicino nello stesso blocco -> shared memory
//                neighX = shPosX[jThread];
//                neighY = shPosY[jThread];
//                neighVX = shVelX[jThread];
//                neighVY = shVelY[jThread];
//                neighInfluence = shInfluence[jThread];
//            }
//            else {
//                // vicino in altro blocco -> global memory
//                neighX = posX_sorted[j];
//                neighY = posY_sorted[j];
//                neighVX = velX_sorted[j];
//                neighVY = velY_sorted[j];
//                neighInfluence = influence_sorted[j];
//            }
//
//            float dx = neighX - px;
//            float dy = neighY - py;
//            float dist = sqrtf(dx * dx + dy * dy);
//
//            if (dist < cohesionDistance) { cohX += neighX; cohY += neighY; neighborCount++; }
//            if (dist < separationDistance && dist > 0.0f) { sepX += (px - neighX) / dist; sepY += (py - neighY) / dist; }
//            if (dist < alignmentDistance) { aliX += neighVX * neighInfluence; aliY += neighVY * neighInfluence; totalWeight += neighInfluence; }
//        }
//    }
//
//    if (neighborCount > 0) { cohX = (cohX / neighborCount - px) * cohesionScale; cohY = (cohY / neighborCount - py) * cohesionScale; }
//    if (totalWeight > 0.0f) { aliX = (aliX / totalWeight) * alignmentScale; aliY = (aliY / totalWeight) * alignmentScale; }
//
//    sepX *= separationScale; sepY *= separationScale;
//
//    float borderX = 0.0f, borderY = 0.0f;
//    if (px < borderAlertDistance) borderX += (borderAlertDistance - px);
//    if ((width - px) < borderAlertDistance) borderX -= (borderAlertDistance - (width - px));
//    if (py < borderAlertDistance) borderY += (borderAlertDistance - py);
//    if ((height - py) < borderAlertDistance) borderY -= (borderAlertDistance - (height - py));
//    borderX *= 0.2f; borderY *= 0.2f;
//
//    outVelChangeX[sortedIdx] = cohX + sepX + aliX + borderX;
//    outVelChangeY[sortedIdx] = cohY + sepY + aliY + borderY;
//}  15.1ms

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
    float* outVelChangeX, float* outVelChangeY)
{
    extern __shared__ float shMem[];
    float* shPosX = shMem;
    float* shPosY = shPosX + blockDim.x;
    float* shVelX = shPosY + blockDim.x;
    float* shVelY = shVelX + blockDim.x;
    float* shInfluence = shVelY + blockDim.x;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px = posX_sorted[i];
    float py = posY_sorted[i];

    float cohX = 0.f, cohY = 0.f, sepX = 0.f, sepY = 0.f, aliX = 0.f, aliY = 0.f;
    int neighborCount = 0;
    float totalWeight = 0.f;

    int col = min(max(int(px / cellWidth), 0), gridResolutionX - 1);
    int row = min(max(int(py / cellWidth), 0), gridResolutionY - 1);

    int dr[4] = { 0, (px - col * cellWidth > 0.5f * cellWidth) ? 1 : -1, 0, (px - col * cellWidth > 0.5f * cellWidth) ? 1 : -1 };
    int dc[4] = { 0, 0, (py - row * cellWidth > 0.5f * cellWidth) ? 1 : -1, (py - row * cellWidth > 0.5f * cellWidth) ? 1 : -1 };

    for (int q = 0; q < 4; ++q) {
        int neighRow = row + dr[q];
        int neighCol = col + dc[q];
        if (neighRow < 0 || neighRow >= gridResolutionY) continue;
        if (neighCol < 0 || neighCol >= gridResolutionX) continue;

        int cellIdx = neighCol + neighRow * gridResolutionX;
        int startIdx = gridCellStartIndices[cellIdx];
        int endIdx = gridCellEndIndices[cellIdx];
        if (startIdx == -1) continue;

        int tileSize = endIdx - startIdx + 1;

        // Caricamento tiling in shared memory
        for (int offset = 0; offset < tileSize; offset += blockDim.x) {
            int tid = threadIdx.x + offset;
            if (tid < tileSize) {
                int idxTile = startIdx + tid;
                shPosX[threadIdx.x] = posX_sorted[idxTile];
                shPosY[threadIdx.x] = posY_sorted[idxTile];
                shVelX[threadIdx.x] = velX_sorted[idxTile];
                shVelY[threadIdx.x] = velY_sorted[idxTile];
                shInfluence[threadIdx.x] = influence_sorted[idxTile];
            }
            __syncthreads();

            int limit = min(tileSize - offset, blockDim.x);
            for (int j = 0; j < limit; ++j) {
                int globalIdx = startIdx + offset + j;
                if (globalIdx == i) continue;

                float dx = shPosX[j] - px;
                float dy = shPosY[j] - py;
                float dist = sqrtf(dx * dx + dy * dy);

                if (dist < cohesionDistance) { cohX += shPosX[j]; cohY += shPosY[j]; neighborCount++; }
                if (dist < separationDistance && dist>0.f) { sepX += (px - shPosX[j]) / dist; sepY += (py - shPosY[j]) / dist; }
                if (dist < alignmentDistance) { aliX += shVelX[j] * shInfluence[j]; aliY += shVelY[j] * shInfluence[j]; totalWeight += shInfluence[j]; }
            }
            __syncthreads();
        }
    }

    if (neighborCount > 0) { cohX = (cohX / neighborCount - px) * cohesionScale; cohY = (cohY / neighborCount - py) * cohesionScale; }
    if (totalWeight > 0) { aliX = (aliX / totalWeight) * alignmentScale; aliY = (aliY / totalWeight) * alignmentScale; }

    sepX *= separationScale; sepY *= separationScale;

    // --- Border forces ---
    float borderX = 0.f, borderY = 0.f;
    if (px < borderAlertDistance) borderX += (borderAlertDistance - px);
    if ((width - px) < borderAlertDistance) borderX -= (borderAlertDistance - (width - px));
    if (py < borderAlertDistance) borderY += (borderAlertDistance - py);
    if ((height - py) < borderAlertDistance) borderY -= (borderAlertDistance - (height - py));
    borderX *= 0.2f; borderY *= 0.2f;

    outVelChangeX[i] = cohX + sepX + aliX + borderX;
    outVelChangeY[i] = cohY + sepY + aliY + borderY;
}

//9.5ms/10.5ms



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

__global__ void copyRenderDataKernel(
    int N,
    const float* posX, const float* posY,
    const float* rotations,
    const float* colorR, const float* colorG, const float* colorB,
    const float* scale,
    glm::vec2* outPositions,
    float* outRotations,
    glm::vec3* outColors,
    float* outScales)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    outPositions[i] = { posX[i], posY[i] };
    outRotations[i] = rotations[i];
    outColors[i] = { colorR[i], colorG[i], colorB[i] };
    outScales[i] = scale[i];
}



