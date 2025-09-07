#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "custom/Simulation.h" 
#include "custom/ResourceManager.h"
#include "custom/Profiler.h"

#include <iostream>

// GLFW function declarations
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);

// The Width of the screen
const unsigned int SCREEN_WIDTH = 1200;
const unsigned int SCREEN_HEIGHT = 800;

// Oggetto globale della simulazione
Simulation simulation(SCREEN_WIDTH, SCREEN_HEIGHT);

int main(int argc, char* argv[])
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_RESIZABLE, false);

    GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Flocking Simulation", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // OpenGL configuration
    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // initialize simulation
    simulation.init();

    Profiler profiler;

    float deltaTime = 0.0f;
    float lastFrame = 0.0f;
    float consoleTimer = 0.0f;
    int frameCounter = 0;

    while (!glfwWindowShouldClose(window))
    {
        // calculate delta time
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        glfwPollEvents();

        // manage user input
        simulation.processInput(deltaTime);

        // update simulation state
        profiler.start();
        simulation.update(deltaTime);
        double updateTime = profiler.stop();
        profiler.log("update", updateTime);

        // render
        profiler.start();
        glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        simulation.render();
        double renderTime = profiler.stop();
        profiler.log("render", renderTime);

        glfwSwapBuffers(window);

        // Aggiornamento console ogni secondo
        consoleTimer += deltaTime;
        frameCounter++;

        if (consoleTimer >= 1.0f)
        {
            double avgUpdate = updateTime; // media aggiornata al momento
            double avgRender = renderTime;

            // Oppure usa le medie calcolate dal Profiler:
            std::cout << "\n";
            profiler.printAverage("update");
            profiler.printAverage("render");

            // Calcolo FPS
            double fps = frameCounter / consoleTimer;
            std::cout << "FPS: " << fps << std::endl;

            // reset contatori
            frameCounter = 0;
            consoleTimer = 0.0f;
        }
    }

    profiler.printAllAverages();
    profiler.saveCSV("./output/benchmark_cpu.csv");

    ResourceManager::Clear();

    glfwTerminate();
    return 0;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (key >= 0 && key < 1024)
    {
        if (action == GLFW_PRESS)
            simulation.keys[key] = true;
        else if (action == GLFW_RELEASE)
            simulation.keys[key] = false;
    }
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}
