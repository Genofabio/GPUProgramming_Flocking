#pragma once
#include <vector>
#include <glm/glm.hpp>

class Wall {
public:
    std::vector<glm::vec2> points;  // polilinea del muro
    float repulsionDistance;        // raggio entro il quale i boid reagiscono
    float repulsionStrength;        // forza di repulsione

    Wall(const std::vector<glm::vec2>& pts, float dist = 80.0f, float strength = 0.7f)
        : points(pts), repulsionDistance(dist), repulsionStrength(strength) {
    }
};
