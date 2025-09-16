#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <gpu/CudaKernels.cuh>
#include <algorithm>


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
    float* outVelChangeX, float* outVelChangeY,
    int numWalls,
    const float2* wallPositions,   // x,y start/end concatenati
    float wallRepulsionDistance,
    float wallRepulsionScale)
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

    float cohX = 0.f, cohY = 0.f;
    float sepX = 0.f, sepY = 0.f;
    float aliX = 0.f, aliY = 0.f;
    int neighborCount = 0;
    float totalWeight = 0.f;

    int col = (int)(px / cellWidth);
    col = (col < 0) ? 0 : ((col >= gridResolutionX) ? gridResolutionX - 1 : col);

    int row = (int)(py / cellWidth);
    row = (row < 0) ? 0 : ((row >= gridResolutionY) ? gridResolutionY - 1 : row);

    int dr[4] = {
        0,
        ((px - col * cellWidth) > 0.5f * cellWidth) ? 1 : -1,
        0,
        ((px - col * cellWidth) > 0.5f * cellWidth) ? 1 : -1
    };
    int dc[4] = {
        0,
        0,
        ((py - row * cellWidth) > 0.5f * cellWidth) ? 1 : -1,
        ((py - row * cellWidth) > 0.5f * cellWidth) ? 1 : -1
    };

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

        // --- tiling in shared memory ---
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

            int limit = (tileSize - offset < blockDim.x) ? (tileSize - offset) : blockDim.x;
            for (int j = 0; j < limit; ++j) {
                int globalIdx = startIdx + offset + j;
                if (globalIdx == i) continue;

                float dx = shPosX[j] - px;
                float dy = shPosY[j] - py;
                float dist = sqrtf(dx * dx + dy * dy);

                if (dist < cohesionDistance) {
                    cohX += shPosX[j];
                    cohY += shPosY[j];
                    neighborCount++;
                }
                if (dist < separationDistance && dist > 0.f) {
                    sepX += (px - shPosX[j]) / dist;
                    sepY += (py - shPosY[j]) / dist;
                }
                if (dist < alignmentDistance) {
                    aliX += shVelX[j] * shInfluence[j];
                    aliY += shVelY[j] * shInfluence[j];
                    totalWeight += shInfluence[j];
                }
            }
            __syncthreads();
        }
    }

    if (neighborCount > 0) {
        cohX = (cohX / neighborCount - px) * cohesionScale;
        cohY = (cohY / neighborCount - py) * cohesionScale;
    }
    if (totalWeight > 0.f) {
        aliX = (aliX / totalWeight) * alignmentScale;
        aliY = (aliY / totalWeight) * alignmentScale;
    }

    sepX *= separationScale;
    sepY *= separationScale;

    // --- Border forces ---
    float borderX = 0.f, borderY = 0.f;
    if (px < borderAlertDistance) borderX += (borderAlertDistance - px);
    if ((width - px) < borderAlertDistance) borderX -= (borderAlertDistance - (width - px));
    if (py < borderAlertDistance) borderY += (borderAlertDistance - py);
    if ((height - py) < borderAlertDistance) borderY -= (borderAlertDistance - (height - py));
    borderX *= 0.2f;
    borderY *= 0.2f;

    // --- Repulsione dai muri ---
    float wallRepX = 0.f, wallRepY = 0.f;

    float lookAhead = 30.0f;

    // Calcola direzione normalizzata della velocità
    float velLen = sqrtf(velX_sorted[i] * velX_sorted[i] + velY_sorted[i] * velY_sorted[i]);
    float dirX = (velLen > 0.0001f) ? velX_sorted[i] / velLen : 0.f;
    float dirY = (velLen > 0.0001f) ? velY_sorted[i] / velLen : 0.f;

    for (int w = 0; w < numWalls; ++w) {
        float2 start = wallPositions[2 * w];     // punto inizio muro
        float2 end = wallPositions[2 * w + 1]; // punto fine muro

        // distanza punto-muro: proiezione del boid sul segmento
        float dx = px - start.x;
        float dy = py - start.y;
        float wallLenX = end.x - start.x;
        float wallLenY = end.y - start.y;
        float wallLenSq = wallLenX * wallLenX + wallLenY * wallLenY;

        float t = fmaxf(0.f, fminf(1.f, (dx * wallLenX + dy * wallLenY) / wallLenSq));

        float closestX = start.x + t * wallLenX;
        float closestY = start.y + t * wallLenY;

        float distX = px - closestX;
        float distY = py - closestY;
        float dist = sqrtf(distX * distX + distY * distY);

        if (dist < wallRepulsionDistance && dist > 0.001f) {
            // safe lookahead (per evitare instabilità troppo vicino al muro)
            float safeLookAhead = fmaxf(0.001f, fminf(lookAhead, dist - 0.2f));
            float probeX = px + dirX * safeLookAhead;
            float probeY = py + dirY * safeLookAhead;

            // direzione "via dal muro"
            float awayX = probeX - closestX;
            float awayY = probeY - closestY;
            float awayLen = sqrtf(awayX * awayX + awayY * awayY);
            if (awayLen > 0.0001f) {
                awayX /= awayLen;
                awayY /= awayLen;
            }

            // forza con fattore quadratico e divisione per distanza
            float factor = (wallRepulsionDistance - dist) / dist;
            float force = factor * factor * wallRepulsionScale;

            wallRepX += awayX * force;
            wallRepY += awayY * force;
        }
    }

    outVelChangeX[i] = cohX + sepX + aliX + borderX + wallRepX;
    outVelChangeY[i] = cohY + sepY + aliY + borderY + wallRepY;
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

    cellX = (cellX < 0) ? 0 : ((cellX >= gridResolutionX) ? gridResolutionX - 1 : cellX);
    cellY = (cellY < 0) ? 0 : ((cellY >= gridResolutionY) ? gridResolutionY - 1 : cellY);

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

    velX[origIdx] += velChangeX_sorted[i] * slowDownFactor;
    velY[origIdx] += velChangeY_sorted[i] * slowDownFactor;

    float speed = sqrtf(velX[origIdx] * velX[origIdx] + velY[origIdx] * velY[origIdx]);
    if (speed > maxSpeed) {
        velX[origIdx] = (velX[origIdx] / speed) * maxSpeed;
        velY[origIdx] = (velY[origIdx] / speed) * maxSpeed;
    }

    posX[origIdx] += velX[origIdx] * dt;
    posY[origIdx] += velY[origIdx] * dt;
}


__global__ void kernComputeRotations(
    int N, const float* velX, const float* velY, float* rotations)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float angle = atan2f(velY[i], velX[i]);   // radianti
    angle = angle * (180.0f / 3.14159265f);   // in gradi
    rotations[i] = angle + 270.0f;            // offset per orientamento modello
}


__global__ void kernIntegratePositions(
    int N, float dt,
    float* posX, float* posY,
    const float* velX, const float* velY)
{
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
    const int* particleArrayIndices,
    float* posX_sorted, float* posY_sorted,
    float* velX_sorted, float* velY_sorted,
    float* scale_sorted, float* influence_sorted,
    int* type_sorted,
    float* colorR_sorted, float* colorG_sorted, float* colorB_sorted,
    float* velChangeX_sorted, float* velChangeY_sorted)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int srcIdx = particleArrayIndices[i];

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
