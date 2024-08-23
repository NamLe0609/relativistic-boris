#include "particle_history.h"
#include "particle.h"
#include "particle_pusher.cuh"

#include <iostream>
#include <cuda_runtime.h>

int main() {
  // Choose number particles and dimension to simulate
  const float3 num_of_particle = make_float3(16.0f, 16.0f, 16.0f);
  const float3 system_length = make_float3(2.0f, 2.0f, 2.0f);
  const int total_particle_count = static_cast<int>(
      num_of_particle.x * num_of_particle.y * num_of_particle.z);
  std::cout << "Number of particles: " << total_particle_count << "\n";

  // Initialize E and B fields
  const float3 e_field = make_float3(0.5f, 0.5f, 0.5f);
  const float3 b_field = make_float3(0.75f, 0.75f, 0.75f);
  const float charge = 1.0f;
  const float mass = 1.0f;

  // choose arbitrary timesteps
  const float timestep = 0.025f;

  // Number of Boris pusher iteration run
  const int max_iter = 1024;

  // Declare values for memory size of some arrays
  int particle_history_mem_size =
      total_particle_count * sizeof(ParticleHistory);
  int float_arr_mem_size = max_iter * sizeof(float);

  // Declare and initialize particles with fixed data
  Particle *particles = new Particle[total_particle_count];
  initialize_particles(particles, num_of_particle, total_particle_count,
                       system_length, ParticlePlacementType::UNIFORM);

  ParticleHistory *particle_histories =
      new ParticleHistory[total_particle_count];
  initialize_particle_history(particle_histories,
                              total_particle_count, max_iter);

  // Allocate and copy particles from host to device
  int particle_mem_size = total_particle_count * sizeof(Particle);
  Particle *device_particles;
  cudaMalloc(&device_particles, particle_mem_size);
  cudaMemcpy(device_particles, particles, particle_mem_size,
             cudaMemcpyHostToDevice);

  // Allocate and copy particle_histories from host to device
  ParticleHistory *device_particle_histories;
  cudaMalloc(&device_particle_histories, particle_history_mem_size);
  cudaMalloc_particle_history(device_particle_histories, total_particle_count,
                              max_iter, float_arr_mem_size);

  // Run Boris pusher on GPU
  launch_update_particles(device_particles, device_particle_histories, e_field,
                          b_field, charge, mass, timestep, total_particle_count,
                          max_iter);

  // Create events to time the memcpy transfer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Get the device particle history pointer to index into the array
  // Otherwise we would get segfaults when indexing into it
  ParticleHistory *temp_particle_histories =
      new ParticleHistory[total_particle_count];
  cudaMemcpy(temp_particle_histories, device_particle_histories,
             particle_history_mem_size, cudaMemcpyDeviceToHost);

  // Copy out the particle_histories data from device to host
  cudaMemcpy_particle_history(particle_histories, temp_particle_histories,
                              total_particle_count, float_arr_mem_size);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float time_elapsed = 0;
  cudaEventElapsedTime(&time_elapsed, start, stop);

  // // Destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  for (int i = 0; i < total_particle_count; i++) {
    std::cout << "Particle " << i << " at time " << max_iter - 1
              << " - x: " << particle_histories[i].x[max_iter - 1]
              << " y: " << particle_histories[i].y[max_iter - 1]
              << " z: " << particle_histories[i].z[max_iter - 1]
              << " px: " << particle_histories[i].px[max_iter - 1]
              << " py: " << particle_histories[i].py[max_iter - 1]
              << " pz: " << particle_histories[i].pz[max_iter - 1] << '\n';
  }

  // Get total data transferred in MB
  double total_data_transferred =
      static_cast<double>(particle_history_mem_size) / (1024.0 * 1024.0);
  std::cout << "Data transfer size: " << total_data_transferred << " MB\n";
  std::cout << "Time taken to copy data: " << time_elapsed << " ms\n";

  cudaFree(device_particles);
  cudaFree(device_particle_histories);
  cudaFree_particle_history(temp_particle_histories, total_particle_count);
  delete[] particles;
  delete[] particle_histories;
  return 0.0;
}
