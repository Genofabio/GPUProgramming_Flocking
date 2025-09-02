#include "custom/Simulation.h"  
#include "custom/ResourceManager.h"
#include "custom/SpriteRenderer.h"

SpriteRenderer* Renderer;

Simulation::Simulation(unsigned int width, unsigned int height)
    : State(SIMULATION_RUNNING)   
    , Keys()
    , Width(width)
    , Height(height)
{
}

Simulation::~Simulation()
{
    delete Renderer;
}

void Simulation::Init()
{
    // load shaders
    ResourceManager::LoadShader("shaders/sprite_shader.vert", "shaders/sprite_shader.frag", nullptr, "sprite");
    // configure shaders
    glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(this->Width),
        static_cast<float>(this->Height), 0.0f, -1.0f, 1.0f);
    ResourceManager::GetShader("sprite").Use().SetInteger("image", 0);
    ResourceManager::GetShader("sprite").SetMatrix4("projection", projection);
    // set render-specific controls
    Renderer = new SpriteRenderer(ResourceManager::GetShader("sprite"));
    // load textures
    ResourceManager::LoadTexture("resources/textures/awesomeface.png", true, "face");
}

void Simulation::Update(float dt)
{
}

void Simulation::ProcessInput(float dt)
{
}

void Simulation::Render()
{
    Renderer->DrawSprite(ResourceManager::GetTexture("face"), glm::vec2(200.0f, 200.0f), glm::vec2(300.0f, 400.0f), 45.0f, glm::vec3(0.0f, 1.0f, 0.0f));
}
