#include "particle.h"
#include <cuda_runtime.h>

void initialize_particles(Particle *particles, float3 num_of_particles,
                          int total_particle_count, float3 system_length,
                          ParticlePlacementType placement_type) {
  switch (placement_type) {
  case ParticlePlacementType::UNIFORM:
    float deltax = system_length.x / num_of_particles.x;
    float deltay = system_length.y / num_of_particles.y;
    float deltaz = system_length.z / num_of_particles.z;
    for (int i = 0; i < total_particle_count; i++) {
      float corrected_index = static_cast<float>(i) + 0.5f;
      particles[i].update(deltax * corrected_index, deltay * corrected_index,
                          deltaz * corrected_index, 10, 10, 10);
    }
    break;
  }
}
