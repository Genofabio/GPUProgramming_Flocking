#include "../gpu/BoidData.h"
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

// --- Helper per controllare errori CUDA ---
#define CUDA_CHECK(err) \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

// --- Allocazione GPU ---
void allocateBoidDataGPU(BoidData& bd, size_t N) {
    bd.N = N;
    CUDA_CHECK(cudaMalloc(&bd.posX, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bd.posY, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bd.velX, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bd.velY, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bd.driftX, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bd.driftY, N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&bd.scale, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bd.influence, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bd.type, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bd.age, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bd.birthTime, N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&bd.colorR, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bd.colorG, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bd.colorB, N * sizeof(float)));

    for (int i = 0; i < 5; ++i)
        CUDA_CHECK(cudaMalloc(&bd.debugX[i], N * sizeof(float)));
    for (int i = 0; i < 5; ++i)
        CUDA_CHECK(cudaMalloc(&bd.debugY[i], N * sizeof(float)));
}

// --- Copia CPU -> GPU ---
void copyBoidsToGPU(const std::vector<Boid>& cpuBoids, BoidData& bd) {
    size_t N = cpuBoids.size();

    std::vector<float> posX(N), posY(N), velX(N), velY(N), driftX(N), driftY(N);
    std::vector<float> scale(N), influence(N), birthTime(N);
    std::vector<int> type(N), age(N);
    std::vector<float> colorR(N), colorG(N), colorB(N);
    std::vector<float> debugX[5], debugY[5];
    for (int i = 0; i < 5; i++) { debugX[i].resize(N); debugY[i].resize(N); }

    for (size_t i = 0; i < N; i++) {
        const Boid& b = cpuBoids[i];
        posX[i] = b.position.x;
        posY[i] = b.position.y;
        velX[i] = b.velocity.x;
        velY[i] = b.velocity.y;
        driftX[i] = b.drift.x;
        driftY[i] = b.drift.y;

        scale[i] = b.scale;
        influence[i] = b.influence;
        type[i] = static_cast<int>(b.type);
        age[i] = b.age;
        birthTime[i] = b.birthTime;

        colorR[i] = b.color.r;
        colorG[i] = b.color.g;
        colorB[i] = b.color.b;

        for (int j = 0; j < 5; j++) {
            debugX[j][i] = b.debugVectors[j].x;
            debugY[j][i] = b.debugVectors[j].y;
        }
    }

    CUDA_CHECK(cudaMemcpy(bd.posX, posX.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd.posY, posY.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd.velX, velX.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd.velY, velY.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd.driftX, driftX.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd.driftY, driftY.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(bd.scale, scale.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd.influence, influence.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd.type, type.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd.age, age.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd.birthTime, birthTime.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(bd.colorR, colorR.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd.colorG, colorG.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd.colorB, colorB.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    for (int j = 0; j < 5; j++) {
        CUDA_CHECK(cudaMemcpy(bd.debugX[j], debugX[j].data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bd.debugY[j], debugY[j].data(), N * sizeof(float), cudaMemcpyHostToDevice));
    }
}

// --- Copia GPU -> CPU ---
void copyBoidsToCPU(BoidData& bd, std::vector<Boid>& cpuBoids) {
    size_t N = bd.N;
    cpuBoids.resize(N);

    std::vector<float> posX(N), posY(N), velX(N), velY(N), driftX(N), driftY(N);
    std::vector<float> scale(N), influence(N), birthTime(N);
    std::vector<int> type(N), age(N);
    std::vector<float> colorR(N), colorG(N), colorB(N);
    std::vector<float> debugX[5], debugY[5];
    for (int i = 0; i < 5; i++) { debugX[i].resize(N); debugY[i].resize(N); }

    CUDA_CHECK(cudaMemcpy(posX.data(), bd.posX, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(posY.data(), bd.posY, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velX.data(), bd.velX, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velY.data(), bd.velY, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(driftX.data(), bd.driftX, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(driftY.data(), bd.driftY, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(scale.data(), bd.scale, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(influence.data(), bd.influence, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(type.data(), bd.type, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(age.data(), bd.age, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(birthTime.data(), bd.birthTime, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(colorR.data(), bd.colorR, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(colorG.data(), bd.colorG, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(colorB.data(), bd.colorB, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int j = 0; j < 5; j++) {
        CUDA_CHECK(cudaMemcpy(debugX[j].data(), bd.debugX[j], N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(debugY[j].data(), bd.debugY[j], N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    for (size_t i = 0; i < N; i++) {
        cpuBoids[i].position = { posX[i], posY[i] };
        cpuBoids[i].velocity = { velX[i], velY[i] };
        cpuBoids[i].drift = { driftX[i], driftY[i] };
        cpuBoids[i].scale = scale[i];
        cpuBoids[i].influence = influence[i];
        cpuBoids[i].type = static_cast<BoidType>(type[i]);
        cpuBoids[i].age = age[i];
        cpuBoids[i].birthTime = birthTime[i];
        cpuBoids[i].color = { colorR[i], colorG[i], colorB[i] };
        for (int j = 0; j < 5; j++)
            cpuBoids[i].debugVectors[j] = { debugX[j][i], debugY[j][i] };
    }
}

// --- Libera GPU ---
void freeBoidDataGPU(BoidData& bd) {
    cudaFree(bd.posX); cudaFree(bd.posY);
    cudaFree(bd.velX); cudaFree(bd.velY);
    cudaFree(bd.driftX); cudaFree(bd.driftY);

    cudaFree(bd.scale); cudaFree(bd.influence);
    cudaFree(bd.type); cudaFree(bd.age); cudaFree(bd.birthTime);

    cudaFree(bd.colorR); cudaFree(bd.colorG); cudaFree(bd.colorB);

    for (int i = 0; i < 5; i++) { cudaFree(bd.debugX[i]); cudaFree(bd.debugY[i]); }
}