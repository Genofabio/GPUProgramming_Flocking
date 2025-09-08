#include "custom/Simulation.h"  
#include "custom/ResourceManager.h"
#include "custom/SpriteRenderer.h"
#include <set>
#include <iostream>
#include <cmath>


Simulation::Simulation(unsigned int width, unsigned int height)
    : state(SIMULATION_RUNNING)
    , keys()
    , width(width)
    , height(height)
    , grid(10, 15, width, height)
    , boidRender(nullptr)
    , wallRender(nullptr)
    , gridRender(nullptr)
    , textRender(nullptr)
    , cohesionDistance(100.0f)
    , separationDistance(25.0f)
    , alignmentDistance(50.0f)
    , borderDistance(300.0f)
    , predatorFearDistance(150.0f)
    , predatorChaseDistance(800.0f)
    , predatorSeparationDistance(60.0f)
    , predatorEatDistance(5.0f)
    // scales / weights defaults
    , cohesionScale(0.05f)
    , separationScale(2.0f)
    , alignmentScale(0.15f)
    , borderScale(0.3f)
    , predatorFearScale(0.8f)
    , predatorChaseScale(0.12f)
    , predatorSeparationScale(2.0f)

    , borderAlertDistance(height/5.0f)
    , rng(std::random_device{}()) // Mersenne Twister con seed casuale
    , dist(-1.0f, 1.0f)
{
}

Simulation::~Simulation()
{
    delete boidRender;
    delete textRender;
}

void Simulation::init()
{
    // caricamento shaders
    ResourceManager::LoadShader("shaders/boid_shader.vert", "shaders/boid_shader.frag", nullptr, "boid");
    ResourceManager::LoadShader("shaders/wall_shader.vert", "shaders/wall_shader.frag", nullptr, "wall");
    ResourceManager::LoadShader("shaders/grid_shader.vert", "shaders/grid_shader.frag", nullptr, "grid");    
    ResourceManager::LoadShader("shaders/text_shader.vert", "shaders/text_shader.frag", nullptr, "text");

    glm::mat4 projection = glm::ortho(
        0.0f, static_cast<float>(width),   // left, right
        0.0f, static_cast<float>(height),  // bottom, top
        -1.0f, 1.0f                         // near, far
    );

    // impostazione shaders
    ResourceManager::GetShader("boid").Use().SetMatrix4("projection", projection);
    ResourceManager::GetShader("text").Use().SetInteger("text", 0);
    ResourceManager::GetShader("text").SetMatrix4("projection", projection);
    ResourceManager::GetShader("wall").Use().SetMatrix4("projection", projection);
    ResourceManager::GetShader("grid").Use().SetMatrix4("projection", projection);

    // inizializzazione renderers
    boidRender = new BoidRenderer(ResourceManager::GetShader("boid"));
    wallRender = new WallRenderer(ResourceManager::GetShader("wall"));
    gridRender = new GridRenderer(ResourceManager::GetShader("grid"));
    textRender = new TextRenderer(ResourceManager::GetShader("text"));

    textRender->Use("resources/fonts/Roboto/Roboto-Regular.ttf", 24);  // carica font e dimensione

    // inizializzazione dei boids con posizioni e velocità random
    const int NUM_PREY = 180;
    const int NUM_PREDATORS = 2;
    const int NUM_LEADERS = 2;

    // Leader
    for (int i = 0; i < NUM_LEADERS; i++) {
        Boid leader;
        leader.position = glm::vec2(rand() % width, rand() % height);
        leader.velocity = glm::vec2((((rand() % 200) - 100) / 100.0f) * 50,
            (((rand() % 200) - 100) / 100.0f) * 50);
        leader.type = LEADER;
        leader.age = 10;
        leader.scale = 1.7f;
        leader.color = glm::vec3(0.9f, 0.9f, 0.2f);
        leader.drift = glm::vec2(0);
        boids.push_back(leader);
    }

    // Prede
    std::uniform_int_distribution<int> ageDist(0, 6);
    std::uniform_real_distribution<float> offsetDist(0.0f, 30.0f);

    for (int i = 0; i < NUM_PREY; i++) {
        Boid b;
        b.position = glm::vec2(rand() % width, rand() % height);
        b.velocity = glm::vec2((((rand() % 200) - 100) / 100.0f) * 100,
            (((rand() % 200) - 100) / 100.0f) * 100);
        b.type = PREY;
        b.birthTime = currentTime + offsetDist(rng); // Birth time random continuo [0, 30]
        b.age = ageDist(rng); // età ranodm [0, 6]
        
        float t = b.age / 10.0f; // Normalizzazione t da 0.0 a 0.6
        
        b.scale = 1.0f + 0.04f * b.age; // Aggiorna scale (1.0 -> 1.24) 

        glm::vec3 blue(0.2f, 0.2f, 0.9f);
        glm::vec3 blue_marine(0.05f, 0.8f, 0.7f);
        b.color = glm::mix(blue, blue_marine, t); // Aggiorna colore (da blu a blu_marine) 

        b.influence = 0.8f + 0.04f * b.age; // Setta influence tra 0.8 (b.age = 0) e 1.04 (b.age = 6) 

        boids.push_back(b);
    }

    // Predatori
    for (int i = 0; i < NUM_PREDATORS; i++) {
        Boid b;
        b.position = glm::vec2(rand() % width, rand() % height);
        b.velocity = glm::vec2((((rand() % 200) - 100) / 100.0f) * 100,
            (((rand() % 200) - 100) / 100.0f) * 100);
        b.type = PREDATOR;
        b.age = 10;
        b.scale = 1.9f;
        b.color = glm::vec3(0.9f, 0.2f, 0.2f);
        b.drift = glm::vec2(0);
        boids.push_back(b);
    }

    // inizializzazione dei muri
    std::vector<Wall> newWalls = generateRandomWalls(0);
    for (const Wall& w : newWalls) {
        walls.emplace_back(w);
    }
}

void Simulation::update(float dt) {
    const float slowDownFactor = 0.3f;
    const float maxSpeed = 100.0f;

    currentTime += dt;

    std::vector<glm::vec2> velocityChanges(boids.size(), glm::vec2(0.0f));

    // 1. Calcola i cambiamenti di velocità da cohesion, separation, alignment
    for (size_t i = 0; i < boids.size(); i++) {
        Boid& b = boids[i];
        glm::vec2 v(0.0f);
        glm::vec2 totalChange(0.0f);

        upgradeBoid(b, currentTime);

        if (b.type == PREY) {
            totalChange =
                moveTowardCenter(i) +
                avoidNeighbors(i) +
                matchVelocity(i) +
                avoidBorders(i) +
                avoidWalls(i) +
                evadePredators(i) +
                followLeaders(i) +
                computeDrift(i, dt);
        }
        else if (b.type == PREDATOR) {
            totalChange =
                chasePrey(i) +
                (avoidOtherPredators(i) * 2.0f) +
                avoidWalls(i) +
                avoidBorders(i) +
                computeDrift(i, dt);
        }
        else if (b.type == LEADER) {
            totalChange =
                leaderSeparation(i) +
                evadePredators(i) +
                avoidWalls(i) +
                avoidBorders(i) +
                computeDrift(i, dt);
        }
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

    // Gestisce lo spawn di nuovi pesci 
    updateMating();

    // Alla fine dell'update elimina i boids mangiati messi in pending
    if (!eatenPrey.empty()) {
        // ordiniamo al contrario per non invalidare gli indici
        std::sort(eatenPrey.rbegin(), eatenPrey.rend());
        for (size_t idx : eatenPrey) {
            if (idx < boids.size()) {
                boids.erase(boids.begin() + idx);
            }
        }
        eatenPrey.clear();
    }
}

void Simulation::processInput(float dt)
{
}

void Simulation::render() {
    glLineWidth(1.0f);
    gridRender->DrawGrid(grid, glm::vec3(0.2f, 0.2f, 0.2f));

    for (Boid& b : boids) {
        float angle = glm::degrees(atan2(b.velocity.y, b.velocity.x)) + 270.0f;

        boidRender->DrawBoid(b.position, angle, b.color, 10.0f*b.scale);
    }

    glLineWidth(3.0f);
    for (const Wall& w : walls) {
        wallRender->DrawWall(w, glm::vec3(0.25f, 0.88f, 0.82f));
    }
}

// === PROFILING ===

void Simulation::updateWithProfiling(float dt)
{
    profiler.start();
    this->update(dt);
    double updateTime = profiler.stop();
    profiler.log("update", updateTime);
}

void Simulation::renderWithProfiling()
{
    profiler.start();
    this->render();  // render boids
    double renderTime = profiler.stop();
    profiler.log("render", renderTime);

    float margin = 10.0f;    // margine dai bordi
    float scale = 0.7f;     // dimensione del testo
    glm::vec3 color(0.9f, 0.9f, 0.3f); // colore giallo chiaro

    // FPS
    double fps = profiler.getCurrentFPS();
    if (fps > 0.0) {
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
        textRender->RenderText(fpsText, margin, static_cast<float>(height) - margin - 1 * 20.0f, scale, color);
    }

    // Numero di boids
    std::string boidText = "Boids: " + std::to_string(boids.size());
    textRender->RenderText(boidText, margin, static_cast<float>(height) - margin - 2 * 20.0f, scale, color);
}

void Simulation::updateStats(float dt)
{
    profiler.updateFrameStats(dt);
}

void Simulation::saveProfilerCSV(const std::string& path)
{
    profiler.saveCSV(path);
}

// === REGOLE ===

// Cohesion rule
glm::vec2 Simulation::moveTowardCenter(size_t i) {
    glm::vec2 perceived_center(0.0f);
    int count = 0;
    Boid& self = boids[i];

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != PREY) continue;  // ignora predatori

        float dist = glm::length(boids[j].position - self.position);
        if (dist < cohesionDistance) {
            perceived_center += boids[j].position;
            count++;
        }
    }

    if (count > 0)
        perceived_center = (perceived_center / (float)count - self.position) * cohesionScale;
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

glm::vec2 Simulation::avoidOtherPredators(size_t i) {
    glm::vec2 c(0.0f);

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != PREDATOR) continue;
        glm::vec2 diff = boids[i].position - boids[j].position;
        float dist = glm::length(diff);
        if (dist < predatorSeparationDistance && dist > 0.0f) {
            c += diff / dist; // respinta proporzionale
        }
    }

    return c; // moltiplicato dal chiamante con predatorSeparationScale
}

// Alignment rule
glm::vec2 Simulation::matchVelocity(size_t i) {
    glm::vec2 perceived_velocity(0.0f);
    float totalWeight = 0.0f;
    Boid& self = boids[i];

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != PREY) continue;  // ignora predatori

        float dist = glm::length(boids[j].position - self.position);
        if (dist < alignmentDistance) {
            float w = boids[j].influence; // peso
            perceived_velocity += boids[j].velocity * w;
            totalWeight += w;
        }
    }

    if (totalWeight > 0.0f)
        perceived_velocity = (perceived_velocity / totalWeight) * alignmentScale;
    else
        perceived_velocity = glm::vec2(0.0f);

    return perceived_velocity;
}

// Borders rule (usa borderDistance e borderScale)
//glm::vec2 Simulation::avoidBorders(const Boid& b) {
//    glm::vec2 force(0.0f);
//
//    float distLeft = b.position.x;
//    float distRight = width - b.position.x;
//    float distTop = b.position.y;
//    float distBottom = height - b.position.y;
//
//    // lambda che calcola la forza non lineare (stessa forma, ma usa borderDistance)
//    auto computeForce = [this](float dist) -> float {
//        if (dist < borderDistance)
//            return pow(borderDistance - dist, 2) / borderDistance;
//        return 0.0f;
//        };
//
//    force += glm::vec2(1, 0) * computeForce(distLeft);
//    force += glm::vec2(-1, 0) * computeForce(distRight);
//    force += glm::vec2(0, 1) * computeForce(distTop);
//    force += glm::vec2(0, -1) * computeForce(distBottom);
//
//    // Scala finale: usa borderScale
//    force *= borderScale;
//
//    return force;
//}

// Evade predators (usa predatorFearDistance e predatorFearScale)
glm::vec2 Simulation::evadePredators(size_t i) {
    glm::vec2 c(0.0f);
    int nearbyAllies = 0;
    float allyRadius = 60.0f; // raggio per contare prede vicine

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != PREDATOR) continue;

        glm::vec2 diff = boids[i].position - boids[j].position;
        float dist = glm::length(diff);
        if (dist < predatorFearDistance && dist > 0.0f) {
            // respinta che aumenta quando il predatore è più vicino
            c += (diff / dist) * (predatorFearDistance - dist);
        }
    }

    // conta quante prede sono vicine (per ridurre l'effetto della fuga)
    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != PREY) continue;
        float dist = glm::length(boids[j].position - boids[i].position);
        if (dist < allyRadius) nearbyAllies++;
    }

    // 3) groupFactor continuo (nessun if)
    // parametri della logistic function (regolabili):
    const float k = 10.03f;   // steepness (maggiore -> transizione più ripida)
    const float n0 = 2.08f;  // centro della transizione (vicino a 4) zona sicura
    float na = static_cast<float>(nearbyAllies);
    float groupFactor = 1.0f / (1.0f + std::exp(-k * (na - n0)));

    return c * predatorFearScale * groupFactor;
}

// gestisce l'accoppiamento
void Simulation::updateMating() {
    size_t N = boids.size();
    if (N == 0) return;

    // inizializza se necessario
    if (boidCouples.size() != N * N) {
        boidCouples.assign(N * N, 0);
    }

    const float mateDistance = 10.0f;   // distanza per considerare i boids "in coppia"
    const int mateThreshold = 200;      // numero di update vicini prima di spawnare
    const int matingAge = 6;

    for (size_t i = 0; i < N; i++) {
        if (boids[i].type != PREY) continue;  // solo prede si accoppiano?
        for (size_t j = i + 1; j < N; j++) {
            if (boids[j].type != PREY) continue;

            float dist = glm::length(boids[i].position - boids[j].position);
            size_t idx = i * N + j;

            if (dist < mateDistance && (boids[i].age >= matingAge && boids[j].age >= matingAge)) {
                boidCouples[idx]++;

                if (boidCouples[idx] >= mateThreshold) {
                    spawnBoid(i, j);               // <-- tua funzione già definita
                    boidCouples[idx] = 0;      // reset contatore
                }
            }
            else {
                boidCouples[idx] = 0; // si sono separati, resetta
            }
        }
    }
}

// Spawna un cucciolo di boid :)
void Simulation::spawnBoid(size_t parentA, size_t parentB) {
    Boid b;

    b.position = (boids[parentA].position + boids[parentB].position) / glm::vec2(2.0f);
    b.velocity = (boids[parentA].velocity + boids[parentB].velocity) / glm::vec2(2.0f);

    b.type = PREY;
    b.birthTime = currentTime;
    b.age = 0; 
    b.scale = 1.0f;
    glm::vec3 blue(0.2f, 0.2f, 0.9f);
    b.color = blue;
    b.influence = 0.8f;

    boids.push_back(b);
}

// Chase prey (con possibilità di mangiare direttamente)
glm::vec2 Simulation::chasePrey(size_t predatorIndex) {
    glm::vec2 target(0.0f);
    float closest = std::numeric_limits<float>::infinity();
    size_t preyIndex = boids.size(); // indice preda candidata

    int nearbyPrey = 0;
    float predatorBoostRadius = 80.0f;

    for (size_t j = 0; j < boids.size(); j++) {
        if (boids[j].type != PREY) continue;

        glm::vec2 diff = boids[j].position - boids[predatorIndex].position;
        float dist = glm::length(diff);

        if (dist < closest && dist < predatorChaseDistance) {
            closest = dist;
            target = boids[j].position;
            preyIndex = j;
        }

        if (dist < predatorBoostRadius) {
            nearbyPrey++;
        }
    }

    if (preyIndex < boids.size()) {
        // Se è abbastanza vicina, mangiala direttamente
        if (closest < predatorEatDistance) {
            eatPrey(predatorIndex, preyIndex);
            return glm::vec2(0.0f); // non serve più la forza di inseguimento
        }

        // Altrimenti calcola la forza di inseguimento
        glm::vec2 dir = target - boids[predatorIndex].position;
        float len = glm::length(dir);
        if (len > 0.0f) {
            float baseForce = ((predatorChaseDistance - closest) / predatorChaseDistance)
                * predatorChaseScale * 100.0f;

            float boost = (nearbyPrey <= 2) ? 1.5f : 1.0f;

            return (dir / len) * baseForce * boost;
        }
    }
    return glm::vec2(0.0f);
}

// Eliminazione dei boid mangiati
void Simulation::eatPrey(size_t predatorIndex, size_t preyIndex) {
    if (predatorIndex >= boids.size() || preyIndex >= boids.size()) return;
    if (boids[predatorIndex].type != PREDATOR || boids[preyIndex].type != PREY) return;

    float dist = glm::length(boids[predatorIndex].position - boids[preyIndex].position);
    if (dist < predatorEatDistance) {
        eatenPrey.push_back(preyIndex);
    }
}

glm::vec2 Simulation::followLeaders(size_t i) {
    Boid& self = boids[i];

    glm::vec2 towardLeader(0.0f);
    float closestDist = std::numeric_limits<float>::infinity();
    glm::vec2 closestLeaderPos(0.0f);

    float leaderInfluenceDistance = 200.0f; // regolabile

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != LEADER) continue;  // <-- fix confronto

        float dist = glm::length(boids[j].position - self.position);
        if (dist < leaderInfluenceDistance && dist < closestDist) {
            closestDist = dist;
            closestLeaderPos = boids[j].position;
        }
    }

    if (closestDist < std::numeric_limits<float>::infinity()) {
        float norm = (leaderInfluenceDistance - closestDist) / leaderInfluenceDistance;
        // curva dolce: vicino = peso quasi nullo, lontano = attrazione maggiore
        float weight = norm * norm; // quadratica, abbassa la forza a corto raggio
        towardLeader = (closestLeaderPos - self.position) * weight * 0.4f;
    }

    return towardLeader; // 0 se nessun leader vicino
}

glm::vec2 Simulation::leaderSeparation(size_t i) {
    glm::vec2 force(0.0f);
    Boid& self = boids[i];

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != LEADER) continue;

        glm::vec2 diff = self.position - boids[j].position;
        float dist = glm::length(diff);
        float desiredLeaderDistance = 200.0f; // regolabile
        if (dist < desiredLeaderDistance && dist > 0.0f) {
            force += diff / dist * (desiredLeaderDistance - dist);
        }
    }

    return force * 0.8f; // scala la forza
}

// Aggiorna i parametri di crescita della preda
void Simulation::upgradeBoid(Boid& b, float currentTime) {
    if (b.type != PREY || b.age >= 10) return;  // solo PREY e max età 10

    if (currentTime - b.birthTime >= 15.0f) { // iniziano a crescere dopo 15s
        //std::cout << "upgrade al tempo " << currentTime << ". Age: " << b.age << " -> " << b.age +1 << std::endl;
        b.age++;  // incremento età

        b.scale += 0.04f; // Aggiorna scale (1.0 -> 1.4)

        // Aggiorna colore (da blu a blu_marine)
        float t = b.age / 10.0f; // Normalizzazione t da 0.0 a 1.0

        glm::vec3 blue(0.2f, 0.2f, 0.9f);
        glm::vec3 blue_marine(0.05f, 0.8f, 0.7f);
        b.color = glm::mix(blue, blue_marine, t);

        // Aggiorna influence (0.8 -> 1.2)
        b.influence += 0.04f;

        // Reset birthTime per prossimo step
        b.birthTime = currentTime;
    }
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
