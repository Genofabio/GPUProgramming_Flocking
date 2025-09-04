#pragma once
#include <glm/glm.hpp>

enum BoidType { PREY, PREDATOR };

struct Boid {
    BoidType type;

    glm::vec2 position;
    glm::vec2 velocity;
};
