#pragma once
#include <glm/glm.hpp>

enum BoidType { PREY, PREDATOR, LEADER };

struct Boid {
    BoidType type;
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec2 drift;
};
