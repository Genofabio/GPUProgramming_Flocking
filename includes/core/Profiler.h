#pragma once
#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <numeric>
#include <fstream>
#include <iostream>
#include <stack>

class Profiler {
public:
    // Avvia un timer (annidabile)
    void start() {
        start_stack.push(clock_type::now());
    }

    // Ferma l'ultimo timer e ritorna durata in ms
    double stop() {
        if (start_stack.empty()) {
            std::cerr << "Profiler error: stop() called without matching start()\n";
            return 0.0;
        }
        auto end_time = clock_type::now();
        auto start_time = start_stack.top();
        start_stack.pop();
        ms duration = end_time - start_time;
        return duration.count();
    }

    // Registra una misura associata ad un'etichetta
    void log(const std::string& label, double value) {
        measurements[label].push_back(value);
    }

    // Calcola e stampa la media di una specifica etichetta
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

    // Salva tutti i dati su file CSV
    void saveCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Errore nell'apertura del file: " << filename << "\n";
            return;
        }
        file << "label,value_ms\n";
        for (const auto& pair : measurements) {
            const std::string& label = pair.first;
            const std::vector<double>& vec = pair.second;
            for (double v : vec) {
                file << label << "," << v << "\n";
            }
        }
        file.close();
    }

    // Stampa la media di tutte le etichette
    void printAllAverages() const {
        for (const auto& pair : measurements) {
            printAverage(pair.first);
        }
    }

    // --- Gestione FPS ---
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

    std::stack<clock_type::time_point> start_stack;
    std::map<std::string, std::vector<double>> measurements;

    // FPS
    double consoleTimer = 0.0;
    int frameCounter = 0;
    double currentFPS = 0.0;
};
