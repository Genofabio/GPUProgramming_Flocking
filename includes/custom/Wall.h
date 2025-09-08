#pragma once
#include <vector>
#include <glm/glm.hpp>

class Wall {
public:
    std::vector<glm::vec2> points;  // polilinea del muro
    float repulsionDistance;        // raggio entro il quale i boid reagiscono
    float repulsionStrength;        // forza di repulsione

    Wall(const std::vector<glm::vec2>& pts,
        float dist = 90.0f,
        float strength = 5.0f)
        : points(pts),
        repulsionDistance(dist),
        repulsionStrength(strength) {
    }

    // Restituisce la distanza minima tra un punto p e questo muro
    // closestPoint sarà aggiornato con il punto sulla polilinea più vicino a p
    float distanceToPoint(const glm::vec2& p, glm::vec2& closestPoint) const {
        if (points.size() < 2) {
            closestPoint = glm::vec2(0.0f);
            return std::numeric_limits<float>::infinity();
        }

        float minDist = std::numeric_limits<float>::infinity();

        for (size_t i = 0; i + 1 < points.size(); ++i) {
            const glm::vec2& a = points[i];
            const glm::vec2& b = points[i + 1];

            glm::vec2 ab = b - a;
            float abLenSquared = glm::dot(ab, ab);
            float t = 0.0f;

            if (abLenSquared > 0.0f) {
                t = glm::dot(p - a, ab) / abLenSquared;
                t = glm::clamp(t, 0.0f, 1.0f);
            }

            glm::vec2 candidate = a + t * ab;
            float dist = glm::length(p - candidate);

            if (dist < minDist) {
                minDist = dist;
                closestPoint = candidate;  // aggiorno il punto più vicino
            }
        }

        return minDist;
    }
};