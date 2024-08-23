#ifndef PARTICLE_PUSHER_H
#define PARTICLE_PUSHER_H

#include "particle.h"
#include "particle_history.h"
#include <cuda_runtime.h>

__device__ void push_particle(Particle &particle, const float3 &e_field,
                              const float3 &b_field, const float timestep,
                              const float charge, float mass);

__global__ void update_particles(Particle *particles,
                                 ParticleHistory *particle_histories,
                                 float3 e_field, float3 b_field, float charge,
                                 float mass, float timestep,
                                 int total_particle_count, int max_iter);

void launch_update_particles(Particle *particles,
                             ParticleHistory *particle_histories,
                             float3 e_field, float3 b_field, float charge,
                             float mass, float timestep,
                             int total_particle_count, int max_iter);

#endif // PARTICLE_PUSHER_H
