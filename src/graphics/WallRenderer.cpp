#include <graphics/WallRenderer.h>

WallRenderer::WallRenderer(const Shader& shader)
    : shader(shader) {
    initBuffers();
}

WallRenderer::~WallRenderer() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

void WallRenderer::initBuffers() {
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

void WallRenderer::draw(const Wall& wall, const glm::vec3& color) {
    shader.Use();
    shader.SetMatrix4("model", glm::mat4(1.0f));
    shader.SetVector3f("spriteColor", color);

    glBindVertexArray(vao);

    for (size_t i = 0; i + 1 < wall.points.size(); ++i) {
        float vertices[4] = {
            wall.points[i].x,     wall.points[i].y,
            wall.points[i + 1].x, wall.points[i + 1].y
        };
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

        glDrawArrays(GL_LINES, 0, 2);
    }

    glBindVertexArray(0);
}