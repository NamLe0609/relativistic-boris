#ifndef PARTICLE_H
#define PARTICLE_H

#include <string>
#include <cuda_runtime.h>

struct Particle {
  float x;
  float y;
  float z;

  float px;
  float py;
  float pz;

  // Default constructor
  Particle() : x(), y(), z(), px(), py(), pz() {}

  // Allow function to run on both host and kernel
  __host__ __device__ void update(float x_pos, float y_pos, float z_pos,
                                  float x_momentum, float y_momentum,
                                  float z_momentum) {
    x = x_pos;
    y = y_pos;
    z = z_pos;
    px = x_momentum;
    py = y_momentum;
    pz = z_momentum;
  }

  const std::string print() {
    return "x: " + std::to_string(x) + " y: " + std::to_string(y) +
           " z: " + std::to_string(z) + " px: " + std::to_string(px) +
           " py: " + std::to_string(py) + " pz: " + std::to_string(pz);
  }
};


enum class ParticlePlacementType {
  UNIFORM, // Evenly spaced integer placement for xyz coordinates
};


void initialize_particles(Particle *particles, float3 num_of_particles,
                          int total_particle_count, float3 system_length,
                          ParticlePlacementType placement_type);


#endif // PARTICLE_H
