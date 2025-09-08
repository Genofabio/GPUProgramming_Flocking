#include "custom/Simulation.h"  
#include "custom/ResourceManager.h"
#include "custom/SpriteRenderer.h"
#include <set>


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
    , borderAlertDistance(height/5.0f)
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

    // inizializzazione dei muri
    std::vector<Wall> newWalls = generateRandomWalls(50);
    for (const Wall& w : newWalls) {
        walls.emplace_back(w);
    }
}

void Simulation::update(float dt) {
    const float slowDownFactor = 0.3f;
    const float maxSpeed = 100.0f;

    std::vector<glm::vec2> velocityChanges(boids.size(), glm::vec2(0.0f));

    // 1. Calcola tutte le variazioni
    for (size_t i = 0; i < boids.size(); ++i) {
        glm::vec2 totalChange =
            moveTowardCenter(i) +
            avoidNeighbors(i) +
            matchVelocity(i) +
            avoidBorders(i) +
            avoidWalls(i) +
            computeDrift(i, dt);

        velocityChanges[i] = totalChange;
    }

    // 2. Applica le variazioni
    for (size_t i = 0; i < boids.size(); ++i) {
        Boid& b = boids[i];

        b.velocity += velocityChanges[i] * slowDownFactor;

        float speed = glm::length(b.velocity);
        if (speed > maxSpeed) {
            b.velocity = (b.velocity / speed) * maxSpeed;
        }

        b.position += b.velocity * dt;
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

// === REGOLE ===

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
            c += diff / dist;
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
    glm::vec2 v5(0.0f);
    const float lookAhead = 30.0f;
    glm::vec2 dir = glm::normalize(boids[i].velocity);

    for (const Wall& w : walls) {
        glm::vec2 closest;
        float dist = w.distanceToPoint(boids[i].position, closest);

        if (dist < w.repulsionDistance && dist > 0.001f) {
            float safeLookAhead = glm::clamp(dist - 0.2f, 0.001f, lookAhead);

            glm::vec2 probePos = boids[i].position + dir * safeLookAhead;

            glm::vec2 away = glm::normalize(probePos - closest);
            float factor = (w.repulsionDistance - dist) / dist;
            v5 += away * factor * factor * w.repulsionStrength;
        }
    }

    return v5;
}

glm::vec2 Simulation::computeDrift(size_t i, float dt) {
    const float driftStep = 10.0f;   // Intensità del drift random
    const float driftLimit = 100.0f;  // Massima ampiezza del drift

    // Aggiorna il drift esistente con nuovo contributo casuale
    boids[i].drift += glm::vec2(dist(rng), dist(rng)) * driftStep * dt;

    // Clamp della lunghezza del drift per evitare crescite incontrollate
    if (glm::length(boids[i].drift) > driftLimit) {
        boids[i].drift = glm::normalize(boids[i].drift) * driftLimit;
    }

    return boids[i].drift;
}

// === UTILITY ===

// Genera n muri casuali rispettando i vincoli
std::vector<Wall> Simulation::generateRandomWalls(int n) {
    std::vector<Wall> result;

    auto candidates = this->grid.cellEdges;
    std::shuffle(candidates.begin(), candidates.end(), rng);

    // memorizza i lati occupati: ogni lato è definito da due vertici ordinati
    std::set<std::pair<std::pair<int, int>, std::pair<int, int>>> usedEdges;

    // memorizza per ogni vertice se è raggiunto da un muro orizzontale o verticale
    std::map<std::pair<int, int>, std::pair<bool, bool>> vertexOccupancy;
    // .first = orizzontale, .second = verticale

    int added = 0;
    for (const auto& edge : candidates) {
        if (added >= n) break;

        auto p1 = std::make_pair(int(edge.first.x), int(edge.first.y));
        auto p2 = std::make_pair(int(edge.second.x), int(edge.second.y));

        // ordina i vertici per uniformità
        if (p2 < p1) std::swap(p1, p2);

        bool horizontal = (p1.second == p2.second);

        // controlla se il lato è già occupato
        if (usedEdges.count({ p1, p2 })) continue;

        // controlla se ci sono muri di orientamento opposto ai vertici
        if (horizontal) {
            if (vertexOccupancy[p1].second || vertexOccupancy[p2].second) continue;
        }
        else {
            if (vertexOccupancy[p1].first || vertexOccupancy[p2].first) continue;
        }

        // aggiungi il muro
        std::vector<glm::vec2> pts = { edge.first, edge.second };
        result.emplace_back(pts, height / 11.0f, 3.0f);

        // segna il lato come occupato
        usedEdges.insert({ p1, p2 });

        // aggiorna i vertici
        if (horizontal) {
            vertexOccupancy[p1].first = true;
            vertexOccupancy[p2].first = true;
        }
        else {
            vertexOccupancy[p1].second = true;
            vertexOccupancy[p2].second = true;
        }

        added++;
    }

    return result;
}
