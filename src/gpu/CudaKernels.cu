#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <gpu/CudaKernels.cuh>
#include <algorithm>

// --- Dichiarazioni in memoria costante ---
__constant__ int d_width;
__constant__ int d_height;

// Parametri boid generali
__constant__ float d_maxSpeed;
__constant__ float d_slowDownFactor;

// Distanze per le regole
__constant__ float d_cohesionDistance;
__constant__ float d_separationDistance;
__constant__ float d_alignmentDistance;
__constant__ float d_borderDistance;
__constant__ float d_predatorFearDistance;
__constant__ float d_predatorChaseDistance;
__constant__ float d_predatorSeparationDistance;
__constant__ float d_predatorEatDistance;
__constant__ float d_leaderInfluenceDistance;
__constant__ float d_desiredLeaderDistance;
__constant__ float d_wallRepulsionDistance;

// Pesi per le regole
__constant__ float d_cohesionScale;
__constant__ float d_separationScale;
__constant__ float d_alignmentScale;
__constant__ float d_borderScale;
__constant__ float d_predatorFearScale;
__constant__ float d_predatorChaseScale;
__constant__ float d_predatorSeparationScale;
__constant__ float d_borderAlertDistance;
__constant__ float d_leaderInfluenceScale;
__constant__ float d_wallRepulsionScale;

// Parametri specifici
__constant__ float d_mateDistance;
__constant__ int   d_mateThreshold;
__constant__ int   d_matingAge;
__constant__ float d_predatorBoostRadius;
__constant__ float d_allyRadius;

// Copia i parametri di simulazione sulla GPU
void setSimulationParamsOnGPU(int width, int height, const BoidParams& params) {
    cudaMemcpyToSymbol(d_width, &width, sizeof(int));
    cudaMemcpyToSymbol(d_height, &height, sizeof(int));

    // Generali
    cudaMemcpyToSymbol(d_maxSpeed, &params.maxSpeed, sizeof(float));
    cudaMemcpyToSymbol(d_slowDownFactor, &params.slowDownFactor, sizeof(float));

    // Distanze
    cudaMemcpyToSymbol(d_cohesionDistance, &params.cohesionDistance, sizeof(float));
    cudaMemcpyToSymbol(d_separationDistance, &params.separationDistance, sizeof(float));
    cudaMemcpyToSymbol(d_alignmentDistance, &params.alignmentDistance, sizeof(float));
    cudaMemcpyToSymbol(d_borderDistance, &params.borderDistance, sizeof(float));
    cudaMemcpyToSymbol(d_predatorFearDistance, &params.predatorFearDistance, sizeof(float));
    cudaMemcpyToSymbol(d_predatorChaseDistance, &params.predatorChaseDistance, sizeof(float));
    cudaMemcpyToSymbol(d_predatorSeparationDistance, &params.predatorSeparationDistance, sizeof(float));
    cudaMemcpyToSymbol(d_predatorEatDistance, &params.predatorEatDistance, sizeof(float));
    cudaMemcpyToSymbol(d_leaderInfluenceDistance, &params.leaderInfluenceDistance, sizeof(float));
    cudaMemcpyToSymbol(d_desiredLeaderDistance, &params.desiredLeaderDistance, sizeof(float));
    cudaMemcpyToSymbol(d_wallRepulsionDistance, &params.wallRepulsionDistance, sizeof(float));

    // Pesi
    cudaMemcpyToSymbol(d_cohesionScale, &params.cohesionScale, sizeof(float));
    cudaMemcpyToSymbol(d_separationScale, &params.separationScale, sizeof(float));
    cudaMemcpyToSymbol(d_alignmentScale, &params.alignmentScale, sizeof(float));
    cudaMemcpyToSymbol(d_borderScale, &params.borderScale, sizeof(float));
    cudaMemcpyToSymbol(d_predatorFearScale, &params.predatorFearScale, sizeof(float));
    cudaMemcpyToSymbol(d_predatorChaseScale, &params.predatorChaseScale, sizeof(float));
    cudaMemcpyToSymbol(d_predatorSeparationScale, &params.predatorSeparationScale, sizeof(float));
    cudaMemcpyToSymbol(d_borderAlertDistance, &params.borderAlertDistance, sizeof(float));
    cudaMemcpyToSymbol(d_leaderInfluenceScale, &params.leaderInfluenceScale, sizeof(float));
    cudaMemcpyToSymbol(d_wallRepulsionScale, &params.wallRepulsionScale, sizeof(float));

    // Specifici
    cudaMemcpyToSymbol(d_mateDistance, &params.mateDistance, sizeof(float));
    cudaMemcpyToSymbol(d_mateThreshold, &params.mateThreshold, sizeof(int));
    cudaMemcpyToSymbol(d_matingAge, &params.matingAge, sizeof(int));
    cudaMemcpyToSymbol(d_predatorBoostRadius, &params.predatorBoostRadius, sizeof(float));
    cudaMemcpyToSymbol(d_allyRadius, &params.allyRadius, sizeof(float));
}

// 1. Utility kernels per la griglia (hashing, ordinamento, reorder)

__global__ void kernComputeGridIndices(
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

__global__ void kernReorderBoidData(
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

    float cohX = 0.f, cohY = 0.f;
    float sepX = 0.f, sepY = 0.f;
    float aliX = 0.f, aliY = 0.f;
    int neighborCount = 0;
    float totalWeight = 0.f;

    int col = (int)(px / cellWidth);
    col = (col < 0) ? 0 : ((col >= gridResolutionX) ? gridResolutionX - 1 : col);
    int row = (int)(py / cellWidth);
    row = (row < 0) ? 0 : ((row >= gridResolutionY) ? gridResolutionY - 1 : row);

    int dr[4] = { 0,
        ((px - col * cellWidth) > 0.5f * cellWidth) ? 1 : -1,
        0,
        ((px - col * cellWidth) > 0.5f * cellWidth) ? 1 : -1 };
    int dc[4] = { 0, 0,
        ((py - row * cellWidth) > 0.5f * cellWidth) ? 1 : -1,
        ((py - row * cellWidth) > 0.5f * cellWidth) ? 1 : -1 };

    for (int q = 0; q < 4; ++q) {
        if (type_sorted[i] != 0) break; // solo prede

        int neighRow = row + dr[q];
        int neighCol = col + dc[q];
        if (neighRow < 0 || neighRow >= gridResolutionY) continue;
        if (neighCol < 0 || neighCol >= gridResolutionX) continue;

        int cellIdx = neighCol + neighRow * gridResolutionX;
        int startIdx = gridCellStartIndices[cellIdx];
        int endIdx = gridCellEndIndices[cellIdx];
        if (startIdx == -1) continue;

        int tileSize = endIdx - startIdx + 1;

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
                if (type_sorted[globalIdx] == 1) continue; // ignora predatori

                float dx = shPosX[j] - px;
                float dy = shPosY[j] - py;
                float dist = sqrtf(dx * dx + dy * dy);

                if (dist < d_cohesionDistance) {
                    cohX += shPosX[j];
                    cohY += shPosY[j];
                    neighborCount++;
                }
                if (dist < d_separationDistance && dist > 0.f) {
                    sepX += (px - shPosX[j]) / dist;
                    sepY += (py - shPosY[j]) / dist;
                }
                if (dist < d_alignmentDistance && type_sorted[globalIdx] == 0) {
                    aliX += shVelX[j] * shInfluence[j];
                    aliY += shVelY[j] * shInfluence[j];
                    totalWeight += shInfluence[j];
                }
            }
            __syncthreads();
        }
    }

    if (neighborCount > 0) {
        cohX = (cohX / neighborCount - px) * d_cohesionScale;
        cohY = (cohY / neighborCount - py) * d_cohesionScale;
    }
    if (totalWeight > 0.f) {
        aliX = (aliX / totalWeight) * d_alignmentScale;
        aliY = (aliY / totalWeight) * d_alignmentScale;
    }

    sepX *= d_separationScale;
    sepY *= d_separationScale;

    outVelChangeX[i] += cohX + sepX + aliX;
    outVelChangeY[i] += cohY + sepY + aliY;
}

__global__ void kernComputeBorderWallForces(
    int N,
    const float* posX_sorted, const float* posY_sorted,
    const float* velX_sorted, const float* velY_sorted,
    int numWalls,
    const float2* wallPositions,
    float* outVelChangeX, float* outVelChangeY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px = posX_sorted[i];
    float py = posY_sorted[i];

    // --- Border forces ---
    float borderX = 0.f, borderY = 0.f;
    if (px < d_borderAlertDistance) borderX += (d_borderAlertDistance - px);
    if ((d_width - px) < d_borderAlertDistance) borderX -= (d_borderAlertDistance - (d_width - px));
    if (py < d_borderAlertDistance) borderY += (d_borderAlertDistance - py);
    if ((d_height - py) < d_borderAlertDistance) borderY -= (d_borderAlertDistance - (d_height - py));
    borderX *= 0.2f;
    borderY *= 0.2f;

    // --- Wall repulsion ---
    float wallRepX = 0.f, wallRepY = 0.f;
    float lookAhead = 30.0f;

    float velLen = sqrtf(velX_sorted[i] * velX_sorted[i] + velY_sorted[i] * velY_sorted[i]);
    float dirX = (velLen > 0.0001f) ? velX_sorted[i] / velLen : 0.f;
    float dirY = (velLen > 0.0001f) ? velY_sorted[i] / velLen : 0.f;

    for (int w = 0; w < numWalls; ++w) {
        float2 start = wallPositions[2 * w];
        float2 end = wallPositions[2 * w + 1];

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

        if (dist < d_wallRepulsionDistance && dist > 0.001f) {
            float safeLookAhead = fmaxf(0.001f, fminf(lookAhead, dist - 0.2f));
            float probeX = px + dirX * safeLookAhead;
            float probeY = py + dirY * safeLookAhead;

            float awayX = probeX - closestX;
            float awayY = probeY - closestY;
            float awayLen = sqrtf(awayX * awayX + awayY * awayY);
            if (awayLen > 0.0001f) {
                awayX /= awayLen;
                awayY /= awayLen;
            }

            float factor = (d_wallRepulsionDistance - dist) / dist;
            float force = factor * factor * d_wallRepulsionScale;

            wallRepX += awayX * force;
            wallRepY += awayY * force;
        }
    }

    outVelChangeX[i] += borderX + wallRepX;
    outVelChangeY[i] += borderY + wallRepY;
}

// 3. Forze speciali (leader, predatori)

__global__ void kernLeaderFollowForces(
    int N,
    const float* posX_sorted,
    const float* posY_sorted,
    const float* velX_sorted,
    const float* velY_sorted,
    const int* type_sorted,
    float* velChangeX_sorted,
    float* velChangeY_sorted)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px = posX_sorted[i];
    float py = posY_sorted[i];

    float deltaX = 0.f;
    float deltaY = 0.f;

    if (type_sorted[i] == 2) {
        // Leader: evita altri leader
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            if (type_sorted[j] != 2) continue;

            float dx = px - posX_sorted[j];
            float dy = py - posY_sorted[j];
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < d_desiredLeaderDistance && dist > 0.001f) {
                float factor = (d_desiredLeaderDistance - dist) / dist;
                deltaX += dx * factor * 0.8f;
                deltaY += dy * factor * 0.8f;
            }
        }
    }
    else if (type_sorted[i] == 0) {
        // Follower: allineamento + coesione verso leader più vicino
        float closestDist = 1e20f;
        int closestIdx = -1;

        for (int j = 0; j < N; ++j) {
            if (type_sorted[j] != 2) continue; // solo leader
            float dx = posX_sorted[j] - px;
            float dy = posY_sorted[j] - py;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < d_leaderInfluenceDistance && dist < closestDist) {
                closestDist = dist;
                closestIdx = j;
            }
        }

        if (closestIdx >= 0) {
            // Coesione verso leader
            float dx = posX_sorted[closestIdx] - px;
            float dy = posY_sorted[closestIdx] - py;
            float norm = (d_leaderInfluenceDistance - closestDist) / d_leaderInfluenceDistance;
            float cohesionWeight = norm * norm;
            deltaX += dx * cohesionWeight * d_leaderInfluenceScale * 0.05f;
            deltaY += dy * cohesionWeight * d_leaderInfluenceScale * 0.05f;

            // Allineamento con velocità del leader
            float alignWeight = 0.5f; // puoi regolare
            deltaX += (velX_sorted[closestIdx] - 0.0f) * alignWeight * 0.5f;
            deltaY += (velY_sorted[closestIdx] - 0.0f) * alignWeight * 0.5f;
        }
    }

    velChangeX_sorted[i] += deltaX;
    velChangeY_sorted[i] += deltaY;
}

__global__ void kernPredatorPreyForces(
    int N,
    const float* posX_sorted,
    const float* posY_sorted,
    const int* type_sorted,
    float* velChangeX_sorted,
    float* velChangeY_sorted)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px = posX_sorted[i];
    float py = posY_sorted[i];

    // --- Predatori ---
    if (type_sorted[i] == 1) { // PREDATOR
        float repX = 0.f, repY = 0.f;

        // Separazione tra predatori
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            if (type_sorted[j] != 1) continue;
            float dx = px - posX_sorted[j];
            float dy = py - posY_sorted[j];
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < d_predatorSeparationDistance && dist > 0.001f) {
                repX += dx / dist;
                repY += dy / dist;
            }
        }

        // Inseguimento prede
        float closestDist = 1e20f;
        float targetX = 0.f, targetY = 0.f;
        int nearbyCount = 0;

        for (int j = 0; j < N; ++j) {
            if (type_sorted[j] == 1) continue; // solo prede
            float dx = posX_sorted[j] - px;
            float dy = posY_sorted[j] - py;
            float dist = sqrtf(dx * dx + dy * dy);

            if (dist < closestDist && dist < d_predatorChaseDistance) {
                closestDist = dist;
                targetX = posX_sorted[j];
                targetY = posY_sorted[j];
            }
            if (dist < d_predatorBoostRadius) nearbyCount++;
        }

        float chaseX = 0.f, chaseY = 0.f;
        if (closestDist < 1e20f) {
            float dirX = targetX - px;
            float dirY = targetY - py;
            float len = sqrtf(dirX * dirX + dirY * dirY);
            if (len > 0.001f) {
                float baseForce = ((d_predatorChaseDistance - closestDist) / d_predatorChaseDistance) * d_predatorChaseScale * 100.0f;
                float boost = (nearbyCount <= 2) ? 1.5f : 1.0f;
                chaseX = (dirX / len) * baseForce * boost;
                chaseY = (dirY / len) * baseForce * boost;
            }
        }

        velChangeX_sorted[i] += repX * d_predatorSeparationScale + chaseX;
        velChangeY_sorted[i] += repY * d_predatorSeparationScale + chaseY;
    }
    // --- Prede/Leader ---
    else {
        float evadeX = 0.f, evadeY = 0.f;
        int nearbyAllies = 0;

        for (int j = 0; j < N; ++j) {
            if (i == j) continue;

            // Solo predatori
            if (type_sorted[j] == 1) {
                float dx = px - posX_sorted[j];
                float dy = py - posY_sorted[j];
                float dist = sqrtf(dx * dx + dy * dy);
                if (dist < d_predatorFearDistance && dist > 0.001f) {
                    evadeX += (dx / dist) * (d_predatorFearDistance - dist);
                    evadeY += (dy / dist) * (d_predatorFearDistance - dist);
                }
            }
            // Conteggio alleati vicini
            else {
                float dx = posX_sorted[j] - px;
                float dy = posY_sorted[j] - py;
                float dist = sqrtf(dx * dx + dy * dy);
                if (dist < d_allyRadius) nearbyAllies++;
            }
        }

        // Fattore di gruppo (sigmoide)
        float k = 10.03f;
        float n0 = 2.08f;
        float groupFactor = 1.0f / (1.0f + expf(-k * (nearbyAllies - n0)));

        velChangeX_sorted[i] += evadeX * d_predatorFearScale * groupFactor;
        velChangeY_sorted[i] += evadeY * d_predatorFearScale * groupFactor;
    }
}

// 4. Integrazione e aggiornamento stato
__global__ void kernApplyVelocityChange(
    int N,
    const float* velChangeX_sorted, const float* velChangeY_sorted,
    float* posX, float* posY,
    float* velX, float* velY,
    const int* particleArrayIndices,
    const int* type_sorted,  // aggiungi questo array
    float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int origIdx = particleArrayIndices[i];

    // Applica cambiamento di velocità
    velX[origIdx] += velChangeX_sorted[i] * d_slowDownFactor;
    velY[origIdx] += velChangeY_sorted[i] * d_slowDownFactor;

    // Calcola velocità
    float speed = sqrtf(velX[origIdx] * velX[origIdx] + velY[origIdx] * velY[origIdx]);

    // Limita la velocità massima
    if (speed > d_maxSpeed) {
        velX[origIdx] = (velX[origIdx] / speed) * d_maxSpeed;
        velY[origIdx] = (velY[origIdx] / speed) * d_maxSpeed;
        speed = d_maxSpeed;  // aggiorna la velocità
    }

    // **Mantieni velocità minima per i leader**
    if (type_sorted[i] == 2) {  // 2 = leader
        float minSpeed = d_maxSpeed / 1.1f; // definisci questa costante
        if (speed < minSpeed) {
            velX[origIdx] = (velX[origIdx] / speed) * minSpeed;
            velY[origIdx] = (velY[origIdx] / speed) * minSpeed;
        }
    }

    // Aggiorna posizione
    posX[origIdx] += velX[origIdx] * dt;
    posY[origIdx] += velY[origIdx] * dt;
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

__global__ void kernComputeRotations(
    int N, const float* velX, const float* velY, float* rotations)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float angle = atan2f(velY[i], velX[i]);   // radianti
    angle = angle * (180.0f / 3.14159265f);   // in gradi
    rotations[i] = angle + 270.0f;            // offset per orientamento modello
}

__global__ void kernSumBuffers(
    int N,
    float* outX, float* outY,
    const float* bufX1, const float* bufY1,
    const float* bufX2, const float* bufY2,
    const float* bufX3, const float* bufY3,
    const float* bufX4, const float* bufY4
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        outX[i] += bufX1[i] + bufX2[i] + bufX3[i] + bufX4[i];
        outY[i] += bufY1[i] + bufY2[i] + bufY3[i] + bufY4[i];
    }
}

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
    float* outScales)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    outPositions[i] = { posX[i], posY[i] };
    outRotations[i] = rotations[i];
    outColors[i] = { colorR[i], colorG[i], colorB[i] };
    outScales[i] = scale[i];
}

