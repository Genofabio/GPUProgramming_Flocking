#include "custom/VectorRenderer.h"
#include <glm/gtc/matrix_transform.hpp>

VectorRenderer::VectorRenderer(const Shader& shader) : shader(shader) {
    // setup VAO/VBO per linee
    glGenVertexArrays(1, &lineVAO);
    glGenBuffers(1, &lineVBO);

    glBindVertexArray(lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    glBindVertexArray(0);
}

VectorRenderer::~VectorRenderer() {
    glDeleteVertexArrays(1, &lineVAO);
    glDeleteBuffers(1, &lineVBO);
}

void VectorRenderer::DrawVector(const glm::vec2& start, const glm::vec2& end, const glm::vec3& color) {
    this->shader.Use();
    this->shader.SetMatrix4("model", glm::mat4(1.0f));
    this->shader.SetVector3f("spriteColor", color);

    float vertices[4] = {
        start.x, start.y,
        end.x, end.y
    };

    glBindVertexArray(lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

    glDrawArrays(GL_LINES, 0, 2);
    glBindVertexArray(0);
}
