#include <graphics/GridRenderer.h>
#include <external/glm/gtc/matrix_transform.hpp>

GridRenderer::GridRenderer(const Shader& shader)
    : shader(shader) {
    initBuffers();
}

GridRenderer::~GridRenderer() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

void GridRenderer::initBuffers() {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(GL_ARRAY_BUFFER, 2 * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void GridRenderer::draw(const Grid& grid, const glm::vec3& color) {
    shader.Use();
    shader.SetMatrix4("model", glm::mat4(1.0f));
    shader.SetVector3f("spriteColor", color);

    glBindVertexArray(vao);

    for (const auto& line : grid.lines) {
        float vertices[4] = {
            line.first.x,  line.first.y,
            line.second.x, line.second.y
        };

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

        glDrawArrays(GL_LINES, 0, 2);
    }

    glBindVertexArray(0);
}
