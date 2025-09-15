#include "gpu/GridData.h"
#include <stdexcept>
#include <cuda_runtime.h>

void allocateGridDataGPU(GridData& gd, size_t N, size_t numCells)
{
    // Prima libera eventuali buffer già allocati
    if (gd.particleGridIndices) cudaFree(gd.particleGridIndices);
    if (gd.particleArrayIndices) cudaFree(gd.particleArrayIndices);
    if (gd.cellStartIndices) cudaFree(gd.cellStartIndices);
    if (gd.cellEndIndices) cudaFree(gd.cellEndIndices);

    // Aggiorna dimensioni
    gd.numCells = numCells;

    // Allocazione dei buffer per i boid
    if (cudaMalloc(&gd.particleGridIndices, N * sizeof(int)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed: particleGridIndices");
    if (cudaMalloc(&gd.particleArrayIndices, N * sizeof(int)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed: particleArrayIndices");

    // Allocazione dei buffer per le celle
    if (cudaMalloc(&gd.cellStartIndices, numCells * sizeof(int)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed: cellStartIndices");
    if (cudaMalloc(&gd.cellEndIndices, numCells * sizeof(int)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed: cellEndIndices");

    // Inizializza le celle a -1 (vuote)
    cudaMemset(gd.cellStartIndices, -1, numCells * sizeof(int));
    cudaMemset(gd.cellEndIndices, -1, numCells * sizeof(int));
}

void freeGridDataGPU(GridData& gd)
{
    if (gd.particleGridIndices) {
        cudaFree(gd.particleGridIndices);
        gd.particleGridIndices = nullptr;
    }
    if (gd.particleArrayIndices) {
        cudaFree(gd.particleArrayIndices);
        gd.particleArrayIndices = nullptr;
    }
    if (gd.cellStartIndices) {
        cudaFree(gd.cellStartIndices);
        gd.cellStartIndices = nullptr;
    }
    if (gd.cellEndIndices) {
        cudaFree(gd.cellEndIndices);
        gd.cellEndIndices = nullptr;
    }

    gd.numCells = 0;
}
