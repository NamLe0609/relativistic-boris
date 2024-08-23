#include "particle_pusher.cuh"
#include <cuda_runtime.h>

__device__ void push_particle(Particle &particle, const float3 &e_field,
                              const float3 &b_field, const float timestep,
                              const float charge, float mass) {
  // Pre-compute reused calculations
  float mass_square = mass * mass;
  float timestep_charge = charge * timestep;

  // half-step momentum from electric field
  float px_half = particle.px + timestep_charge * e_field.x / 2.0f;
  float py_half = particle.py + timestep_charge * e_field.y / 2.0f;
  float pz_half = particle.pz + timestep_charge * e_field.z / 2.0f;

  // lorentz factor for half-step momentum
  float lorentz = sqrtf(1.0f + px_half * px_half + py_half * py_half +
                        pz_half * pz_half / mass_square);

  // rotation vector from magnetic field
  float lorentz_double = 2.0f * lorentz; // Pre-compute to avoid recalculating
  float tx = timestep_charge * b_field.x / lorentz_double;
  float ty = timestep_charge * b_field.y / lorentz_double;
  float tz = timestep_charge * b_field.z / lorentz_double;
  float t_mag_square = tx * tx + ty * ty + tz * tz;

  // cross product of half p and t
  float px_prime = px_half + (py_half * tz - pz_half * ty);
  float py_prime = py_half + (pz_half * tx - px_half * tz);
  float pz_prime = pz_half + (px_half * ty - py_half * tx);

  // update momentum with effect from boris rotation and electric field
  float denominator = 1 + t_mag_square;
  float px_updated = px_half +
                     2 * (py_prime * tz - pz_prime * ty) / denominator +
                     timestep_charge * e_field.x / 2.0f;
  float py_updated = py_half +
                     2 * (pz_prime * tx - px_prime * tz) / denominator +
                     timestep_charge * e_field.y / 2.0f;
  float pz_updated = pz_half +
                     2 * (px_prime * ty - py_prime * tx) / denominator +
                     timestep_charge * e_field.z / 2.0f;

  // lorentz factor for updated momentum
  lorentz = sqrtf(1.0f + px_updated * px_updated + py_updated * py_updated +
                  pz_updated * pz_updated / mass_square);

  // update position using calculated velocity
  float lorentz_mass = mass * lorentz; // Pre-compute to avoid recalculating
  float x_updated = particle.x + timestep * px_updated / lorentz_mass;
  float y_updated = particle.y + timestep * py_updated / lorentz_mass;
  float z_updated = particle.z + timestep * pz_updated / lorentz_mass;

  // update the particle with new location and momentum
  particle.update(x_updated, y_updated, z_updated, px_updated, py_updated,
                  pz_updated);
}

__global__ void update_particles(Particle *particles,
                                 ParticleHistory *particle_histories,
                                 float3 e_field, float3 b_field, float charge,
                                 float mass, float timestep,
                                 int total_particle_count, int max_iter) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_particle_count) {
    for (int t = 0; t < max_iter; t++) {
      particle_histories[idx].x[t] = particles[idx].x;
      particle_histories[idx].y[t] = particles[idx].y;
      particle_histories[idx].z[t] = particles[idx].z;
      particle_histories[idx].px[t] = particles[idx].px;
      particle_histories[idx].py[t] = particles[idx].py;
      particle_histories[idx].pz[t] = particles[idx].pz;

      // printf("Particle %d at time %d - x: %.2f, y: %.2f, z: %.2f, px: %.2f, "
      //        "%.2f, pz: %.2f \n",
      //        idx, t, particle_histories[idx].x[t],
      //        particle_histories[idx].y[t], particle_histories[idx].z[t],
      //        particle_histories[idx].px[t], particle_histories[idx].py[t],
      //        particle_histories[idx].pz[t]);
      push_particle(particles[idx], e_field, b_field, timestep, charge, mass);
    }
  }
}

void launch_update_particles(Particle *particles,
                             ParticleHistory *particle_histories,
                             float3 e_field, float3 b_field, float charge,
                             float mass, float timestep,
                             int total_particle_count, int max_iter) {
  const int threads_per_block = 256;
  const int blocks =
      (total_particle_count + threads_per_block - 1) / threads_per_block;
  update_particles<<<blocks, threads_per_block>>>(
      particles, particle_histories, e_field, b_field, charge, mass, timestep,
      total_particle_count, max_iter);
  cudaDeviceSynchronize();
}
