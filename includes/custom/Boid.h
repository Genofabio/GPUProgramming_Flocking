#pragma once
#include <glm/glm.hpp>

enum BoidType { PREY, PREDATOR, LEADER };

struct Boid {
    BoidType type;
    unsigned int age; // max 10
    float scale;
    glm::vec3 color;
    float influence;
    float birthTime;

    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec2 drift;
};
