#version 330 core
layout (location = 0) in vec2 aPos;

// Attributi per instanza
layout (location = 1) in mat4 instanceModel;
layout (location = 5) in vec3 instanceColor;

out vec3 fragColor;

uniform mat4 projection;

void main()
{
    gl_Position = projection * instanceModel * vec4(aPos, 0.0, 1.0);
    fragColor = instanceColor;
}
