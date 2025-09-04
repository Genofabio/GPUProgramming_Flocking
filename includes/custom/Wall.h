#pragma once
#include <vector>
#include <glm/glm.hpp>

class Wall {
public:
    std::vector<glm::vec2> points;  // polilinea del muro
    float repulsionDistance;        // raggio entro il quale i boid reagiscono
    float repulsionStrength;        // forza di repulsione

    Wall(const std::vector<glm::vec2>& pts, float dist = 90.0f, float strength = 5.0f)
        : points(pts), repulsionDistance(dist), repulsionStrength(strength) {
    }
};
