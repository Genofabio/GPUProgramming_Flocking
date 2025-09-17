#pragma once
#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <numeric>
#include <fstream>
#include <iostream>
#include <stack>

#include <cuda_runtime.h>

class Profiler {
public:
    Profiler() {
        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
            useCUDA = true;
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "[Profiler] Mod GPU attiva (device: "
                << prop.name << ", " << prop.multiProcessorCount << " SM)\n";
        }
        else {
            useCUDA = false;
            std::cout << "[Profiler] Mod CPU attiva (nessuna GPU CUDA disponibile)\n";
        }
    }

    void start() {
        if (useCUDA) {
            cudaEvent_t startEvent, stopEvent;
            cudaEventCreate(&startEvent);
            cudaEventCreate(&stopEvent);
            cudaEventRecord(startEvent, 0);
            event_stack.push({ startEvent, stopEvent });
        }
        else {
            start_stack.push(clock_type::now());
        }
    }

    double stop() {
        if (useCUDA) {
            if (event_stack.empty()) return 0.0;
            auto ev = event_stack.top();
            event_stack.pop();

            cudaEventRecord(ev.second, 0);
            cudaEventSynchronize(ev.second);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev.first, ev.second);

            cudaEventDestroy(ev.first);
            cudaEventDestroy(ev.second);

            return static_cast<double>(ms);
        }
        else {
            if (start_stack.empty()) return 0.0;
            auto end_time = clock_type::now();
            auto start_time = start_stack.top();
            start_stack.pop();
            ms duration = end_time - start_time;
            return duration.count();
        }
    }

    void log(const std::string& label, double value) {
        measurements[label].push_back(value);
    }

    void printAverage(const std::string& label) const {
        auto it = measurements.find(label);
        if (it == measurements.end() || it->second.empty()) {
            std::cout << label << ": nessun dato registrato.\n";
            return;
        }
        const std::vector<double>& vec = it->second;
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        double avg = sum / vec.size();
        std::cout << label << " average: " << avg << " ms\n";
    }

    void saveCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Errore nell'apertura del file: " << filename << "\n";
            return;
        }
        file << "label,value_ms\n";
        for (const auto& pair : measurements) {
            for (double v : pair.second) {
                file << pair.first << "," << v << "\n";
            }
        }
    }

    void printAllAverages() const {
        for (const auto& pair : measurements)
            printAverage(pair.first);
    }

    // --- FPS ---
    void updateFrameStats(double dt) {
        frameCounter++;
        consoleTimer += dt;
        if (consoleTimer >= 1.0) {
            currentFPS = static_cast<double>(frameCounter) / consoleTimer;
            std::cout << "\n";
            printAllAverages();
            std::cout << "FPS: " << static_cast<int>(currentFPS) << "\n";
            frameCounter = 0;
            consoleTimer = 0.0;
        }
    }

    double getCurrentFPS() const { return currentFPS; }

private:
    using clock_type = std::chrono::high_resolution_clock;
    using ms = std::chrono::duration<double, std::milli>;

    bool useCUDA = false;

    std::stack<clock_type::time_point> start_stack;
    std::stack<std::pair<cudaEvent_t, cudaEvent_t>> event_stack;

    std::map<std::string, std::vector<double>> measurements;

    // FPS
    double consoleTimer = 0.0;
    int frameCounter = 0;
    double currentFPS = 0.0;
};
