#pragma once
#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <numeric>
#include <fstream>
#include <iostream>

class Profiler {
public:
    // Avvia cronometro
    void start() {
        start_time = clock_type::now();
    }

    // Ferma cronometro e ritorna durata in millisecondi
    double stop() {
        auto end_time = clock_type::now();
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

    // --- Nuova funzione: gestisce FPS e stampa periodica ---
    void updateFrameStats(double dt) {
        frameCounter++;
        consoleTimer += dt;
        if (consoleTimer >= 1.0) {
            currentFPS = static_cast<double>(frameCounter) / consoleTimer;  // <-- salvo l'FPS
            std::cout << "\n";
            printAverage("update");
            printAverage("render");
            std::cout << "FPS: " << static_cast<int>(currentFPS) << "\n";
            // reset
            frameCounter = 0;
            consoleTimer = 0.0;
        }
    }

    double getCurrentFPS() const { return currentFPS; }   // <-- nuovo getter

private:
    using clock_type = std::chrono::high_resolution_clock;
    using ms = std::chrono::duration<double, std::milli>;

    clock_type::time_point start_time;
    std::map<std::string, std::vector<double>> measurements;

    // --- Nuovi membri per FPS ---
    double consoleTimer = 0.0;  // accumula secondi
    int frameCounter = 0;       // conta frame nell'intervallo
    double currentFPS = 0.0;
};
