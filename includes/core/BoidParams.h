#pragma once

struct BoidParams {
    // Parametri boid generali
    float maxSpeed;
    float slowDownFactor;

    // Distanze per le regole
    float cohesionDistance;
    float separationDistance;
    float alignmentDistance;
    float borderDistance;
    float predatorFearDistance;
    float predatorChaseDistance;
    float predatorSeparationDistance;
    float predatorEatDistance;
    float leaderInfluenceDistance;

    // Pesi per le regole
    float cohesionScale;
    float separationScale;
    float alignmentScale;
    float borderScale;
    float predatorFearScale;
    float predatorChaseScale;
    float predatorSeparationScale;
    float borderAlertDistance;
    float desiredLeaderDistance;

    // Parametri specifici
    float mateDistance;
    int mateThreshold;
    int matingAge;
    float predatorBoostRadius;
};
