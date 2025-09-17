# GPU Boid Simulation – Prey, Predators & Leaders  

## Introduction  
This project focuses on the **optimization of a flocking simulation** with prey–predator dynamics. Flocking models are widely used to reproduce the collective movement of animal groups such as schools of fish, flocks of birds, or herds. Each agent (boid) follows a few simple rules—**alignment, cohesion, and separation**—that together generate complex emergent group behavior.  

Our implementation combines **OpenGL** for visualization and **CUDA** for GPU acceleration. We first developed a **CPU-based version**, then redesigned and optimized it to leverage GPU parallelism, achieving real-time performance with larger and more dynamic groups.  

The project was inspired by a **University of Pennsylvania assignment outline**, which we used as a starting point and challenge to build our own full implementation.  

---

## Features  
- **Three types of agents**:  
  - **Prey** → follow flocking rules, evade predators, follow leaders  
  - **Leaders** → guide prey, avoid predators and other leaders  
  - **Predators** → chase prey, keep distance from other predators  
- **Obstacles (walls)** → force agents to dynamically avoid collisions and change paths  
- **CPU & GPU implementations** → fair comparison of performance  
- **Profiler** → measures execution times, FPS, exports results to CSV  

---

## GPU Optimizations  
- **Structure of Arrays (SoA)** for coalesced memory access  
- **Uniform spatial grid** with boid reordering → reduces neighbor checks from O(N²) to O(N·k)  
- **Shared memory caching** of local neighbors within grid cells  
- **CUDA streams** for overlapping kernel execution and asynchronous memory transfers  
- **Enhanced profiler** using CUDA events for fine-grained performance metrics  

---

## Performance  
- **CPU version**: quickly becomes bottlenecked by O(N²) interactions  
- **GPU version**: supports **thousands of agents in real time**  
- Profiling shows rendering remains light (<3 ms), while update logic dominates compute time  
- With CUDA optimizations, simulation scales significantly better compared to the CPU baseline  

---

