#include "custom/GridRenderer.h"
#include "custom/Grid.h"
#include <glm/gtc/matrix_transform.hpp>

GridRenderer::GridRenderer(const Shader& shader) : shader(shader) {
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

GridRenderer::~GridRenderer() {
    glDeleteVertexArrays(1, &lineVAO);
    glDeleteBuffers(1, &lineVBO);
}

void GridRenderer::DrawGrid(const Grid& grid, glm::vec3 color) {
    this->shader.Use();
    this->shader.SetMatrix4("model", glm::mat4(1.0f));
    this->shader.SetVector3f("spriteColor", color);

    glBindVertexArray(this->lineVAO);

    for (const auto& line : grid.lines) {
        float vertices[4] = {
            line.first.x, line.first.y,
            line.second.x, line.second.y
        };

        glBindBuffer(GL_ARRAY_BUFFER, this->lineVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

        glDrawArrays(GL_LINES, 0, 2);
    }

    glBindVertexArray(0);
}
