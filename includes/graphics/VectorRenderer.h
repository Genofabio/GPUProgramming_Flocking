#pragma once
#ifndef VECTOR_RENDERER_H
#define VECTOR_RENDERER_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <graphics/Shader.h>

class VectorRenderer {
public:
    VectorRenderer(const Shader& shader);
    ~VectorRenderer();

    // Disegna un vettore dalla posizione start alla posizione end, con un colore
    void DrawVector(const glm::vec2& start, const glm::vec2& end, const glm::vec3& color);

private:
    Shader shader;
    unsigned int lineVAO, lineVBO;
};

#endif
