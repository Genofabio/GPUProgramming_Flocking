#include "custom/Simulation.h"  
#include "custom/ResourceManager.h"
#include "custom/SpriteRenderer.h"

Simulation::Simulation(unsigned int width, unsigned int height)
    : state(SIMULATION_RUNNING)
    , keys()
    , width(width)
    , height(height)
    , grid(10, 15, width, height)
    , boidRender(nullptr)
    , wallRender(nullptr)
    , gridRender(nullptr)
    , cohesionDistance(100.0f)
    , separationDistance(25.0f)
    , alignmentDistance(50.0f)
    , cohesionScale(0.2f)
    , separationScale(8.0f)
    , alignmentScale(0.125f)
    , borderAlertDistance(width/11.0f)
    , rng(std::random_device{}()) // Mersenne Twister con seed casuale
    , dist(-1.0f, 1.0f)
{
}

Simulation::~Simulation()
{
    delete boidRender;
}

void Simulation::init()
{
    // caricamento shaders
    ResourceManager::LoadShader("shaders/boid_shader.vert", "shaders/boid_shader.frag", nullptr, "boid");
    ResourceManager::LoadShader("shaders/wall_shader.vert", "shaders/wall_shader.frag", nullptr, "wall");
    ResourceManager::LoadShader("shaders/grid_shader.vert", "shaders/grid_shader.frag", nullptr, "grid");    

    glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(this->width),
        static_cast<float>(this->height), 0.0f, -1.0f, 1.0f);

    // impostazione shaders
    ResourceManager::GetShader("boid").Use().SetMatrix4("projection", projection);
    ResourceManager::GetShader("wall").Use().SetMatrix4("projection", projection);
    ResourceManager::GetShader("grid").Use().SetMatrix4("projection", projection);

    // inizializzazione renderers
    boidRender = new BoidRenderer(ResourceManager::GetShader("boid"));
    wallRender = new WallRenderer(ResourceManager::GetShader("wall"));
    gridRender = new GridRenderer(ResourceManager::GetShader("grid"));

    // inizializzazione dei boids con posizioni e velocità random
    const int NUM_BOIDS = 200;
    for (int i = 0; i < NUM_BOIDS; i++) {
        Boid b;
        b.position = glm::vec2(rand() % width, rand() % height);
        b.velocity = glm::vec2((((rand() % 200) - 100) / 100.0f) * 100,
            (((rand() % 200) - 100) / 100.0f) * 100);
        b.drift = glm::vec2(0);
        boids.push_back(b);
    }

    // === inizializzazione muri di test ===
    std::vector<Wall> newWalls = generateRandomWalls(70);
    for (const Wall& w : newWalls) {
        walls.emplace_back(w);
    }

}

void Simulation::update(float dt) {
    std::vector<glm::vec2> velocityChanges(boids.size(), glm::vec2(0.0f));
    float slowDown = 0.3f; 

    // 1. Calcola i cambiamenti di velocità da cohesion, separation, alignment e la repulsione di bordi e muri
    for (size_t i = 0; i < boids.size(); i++) {
        glm::vec2 v1 = moveTowardCenter(i);
        glm::vec2 v2 = avoidNeighbors(i);
        glm::vec2 v3 = matchVelocity(i);
        glm::vec2 v4 = avoidBorders(i);
        glm::vec2 v5 = avoidWalls(i);

        velocityChanges[i] = v1 + v2 + v3 + v4 + v5;

        Boid& b = boids[i];

        // Update movimento randomico
        float driftChange = 10.0f;   // quanto cambia il drift a ogni frame
        float driftMax = 100.0f;     // massimo contributo random

        b.drift += glm::vec2(dist(rng), dist(rng)) * driftChange * dt;;

        // clamp per non farlo crescere troppo
        if (glm::length(b.drift) > driftMax) {
            b.drift = glm::normalize(b.drift) * driftMax;
        }

        velocityChanges[i] += b.drift;
    }

    // 2. Applica i cambiamenti e stabilizza la velocità
    for (size_t i = 0; i < boids.size(); i++) {
        // applica le regole
        boids[i].velocity += velocityChanges[i] * slowDown;

        // clamp della velocità
        float maxSpeed = 100.0f;
        float speed = glm::length(boids[i].velocity);
        if (speed > maxSpeed) {
            boids[i].velocity = (boids[i].velocity / speed) * maxSpeed;
        }

        // 3. aggiorna la posizione
        boids[i].position += boids[i].velocity * dt;
    }
}

void Simulation::processInput(float dt)
{
}

void Simulation::render() {
    glLineWidth(1.0f);
    gridRender->DrawGrid(grid, glm::vec3(0.2f, 0.2f, 0.2f));

    for (Boid& b : boids) {
        float angle = glm::degrees(atan2(b.velocity.y, b.velocity.x)) + 270;
        boidRender->DrawBoid(b.position, angle, glm::vec3(0.4f, 0.5f, 0.9f), 10.0f);
    }

    glLineWidth(3.0f);
    for (const Wall& w : walls) {
        wallRender->DrawWall(w, glm::vec3(0.25f, 0.88f, 0.82f));
    }
}

// Cohesion rule
glm::vec2 Simulation::moveTowardCenter(size_t i) {
    glm::vec2 perceived_center(0.0f);
    int count = 0;

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        float dist = glm::length(boids[j].position - boids[i].position);
        if (dist < cohesionDistance) {
            perceived_center += boids[j].position;
            count++;
        }
    }

    if (count > 0)
        perceived_center = (perceived_center / (float)count - boids[i].position) * cohesionScale;
    else
        perceived_center = glm::vec2(0.0f);

    return perceived_center;
}

// Separation rule
glm::vec2 Simulation::avoidNeighbors(size_t i) {
    glm::vec2 c(0.0f);

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        glm::vec2 diff = boids[i].position - boids[j].position;
        float dist = glm::length(diff);
        if (dist < separationDistance && dist > 0.0f) {
            c += diff / dist; // respinta proporzionale
        }
    }

    return c * separationScale;
}

// Alignment rule
glm::vec2 Simulation::matchVelocity(size_t i) {
    glm::vec2 perceived_velocity(0.0f);
    int count = 0;

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        float dist = glm::length(boids[j].position - boids[i].position);
        if (dist < alignmentDistance) {
            perceived_velocity += boids[j].velocity;
            count++;
        }
    }

    if (count > 0)
        perceived_velocity = (perceived_velocity / (float)count) * alignmentScale;
    else
        perceived_velocity = glm::vec2(0.0f);

    return perceived_velocity;
}

// Border avoidance 
glm::vec2 Simulation::avoidBorders(size_t i) {
    glm::vec2 borderRepulsion(0.0f);

    // distanza dai bordi
    float distLeft = boids[i].position.x;
    float distRight = width - boids[i].position.x;
    float distTop = boids[i].position.y;
    float distBottom = height - boids[i].position.y;

    // somma dei contributi dei bordi che superano la soglia
    if (distLeft < borderAlertDistance)   borderRepulsion += glm::vec2(1, 0) * (borderAlertDistance - distLeft);
    if (distRight < borderAlertDistance)  borderRepulsion += glm::vec2(-1, 0) * (borderAlertDistance - distRight);
    if (distTop < borderAlertDistance)    borderRepulsion += glm::vec2(0, 1) * (borderAlertDistance - distTop);
    if (distBottom < borderAlertDistance) borderRepulsion += glm::vec2(0, -1) * (borderAlertDistance - distBottom);

    // scala finale per evitare scatti troppo forti
    borderRepulsion *= 0.2f;

    return borderRepulsion;
}

// Walls avoidance
glm::vec2 Simulation::avoidWalls(size_t i) {

}

// distanza tra punto p e segmento ab, ritorna anche il punto più vicino
inline float Simulation::pointSegmentDistance(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b, glm::vec2& closest) {
    glm::vec2 ab = b - a;
    float t = glm::dot(p - a, ab) / glm::dot(ab, ab);
    t = glm::clamp(t, 0.0f, 1.0f); // proiezione limitata al segmento
    closest = a + t * ab;
    return glm::length(p - closest);
} //!!!!!!!!! occhio a divisioni per 0

std::vector<Wall> Simulation::generateRandomWalls(int n) {
    std::vector<Wall> result;

    auto candidates = this->grid.cellEdges;
    std::shuffle(candidates.begin(), candidates.end(), rng);

    int count = std::min(n, static_cast<int>(candidates.size()));
    for (int i = 0; i < count; i++) {
        const auto& edge = candidates[i];
        std::vector<glm::vec2> pts = { edge.first, edge.second };
        result.emplace_back(pts, height / 11.0f, 3.0f);
    }

    return result;
}
