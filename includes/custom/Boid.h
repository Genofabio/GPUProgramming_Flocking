#pragma once
#include <glm/glm.hpp>

struct Boid {
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec2 drift;
}; 
