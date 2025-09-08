#include "custom/WallRenderer.h"

WallRenderer::WallRenderer(const Shader& shader) {
    this->shader = shader;

    glGenVertexArrays(1, &this->lineVAO);
    glGenBuffers(1, &this->lineVBO);

    glBindVertexArray(this->lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, this->lineVBO);

    // Due vertici (inizio e fine del muro)
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

WallRenderer::~WallRenderer() {
    glDeleteVertexArrays(1, &this->lineVAO);
    glDeleteBuffers(1, &this->lineVBO);
}

void WallRenderer::DrawWall(const Wall& wall, glm::vec3 color) {
    this->shader.Use();
    this->shader.SetMatrix4("model", glm::mat4(1.0f));
    this->shader.SetVector3f("spriteColor", color);

    glBindVertexArray(this->lineVAO);

    for (size_t i = 0; i + 1 < wall.points.size(); i++) {
        float vertices[4] = {
            wall.points[i].x, wall.points[i].y,
            wall.points[i+1].x, wall.points[i+1].y
        };

        glBindBuffer(GL_ARRAY_BUFFER, this->lineVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

        glDrawArrays(GL_LINES, 0, 2);
    }

    glBindVertexArray(0);
}

