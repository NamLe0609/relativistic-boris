#include <algorithm>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <math.h>
#include <string>
#include <vector_functions.h>

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
  __host__ __device__
  void update(float x_pos, float y_pos, float z_pos, float x_momentum,
              float y_momentum, float z_momentum) {
    x = x_pos;
    y = y_pos;
    z = z_pos;
    px = x_momentum;
    py = y_momentum;
    pz = z_momentum;
  }

  std::string print() {
    return "x: " + std::to_string(x) + " y: " + std::to_string(y) +
           " z: " + std::to_string(z) + " px: " + std::to_string(px) +
           " py: " + std::to_string(py) + " pz: " + std::to_string(pz);
  }
};

struct ParticleHistory {
  float *x;
  float *y;
  float *z;
  float *px;
  float *py;
  float *pz;

  // Default constructor
  ParticleHistory()
      : x(nullptr), y(nullptr), z(nullptr), px(nullptr), py(nullptr),
        pz(nullptr) {}

  // Initialize the variable lists of size max_time
  ParticleHistory(int max_time) {
    x = new float[max_time];
    y = new float[max_time];
    z = new float[max_time];
    px = new float[max_time];
    py = new float[max_time];
    pz = new float[max_time];
  }

  std::string print(int time) {
    return "x: " + std::to_string(x[time]) + " y: " + std::to_string(y[time]) +
           " z: " + std::to_string(z[time]) +
           " px: " + std::to_string(px[time]) +
           " py: " + std::to_string(py[time]) +
           " pz: " + std::to_string(pz[time]);
  }

  ~ParticleHistory() {
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] px;
    delete[] py;
    delete[] pz;
  }
};

enum class ParticlePlacementType {
  UNIFORM, // Evenly spaced integer placement for xyz coordinates
};

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

void initialize_particle_history(ParticleHistory *particle_histories,
                                 int total_particle_count, int max_iter) {
  // Initialize for all particles
  for (int i = 0; i < total_particle_count; i++) {
    particle_histories[i] = ParticleHistory(max_iter);
  }
}

void cudaMalloc_particle_history(ParticleHistory *particle_histories,
                                 int total_particle_count, int max_iter) {
  // As this is going on the device, we must use cudaMalloc
  int floatarr_mem_size = max_iter * sizeof(float);
  for (int i = 0; i < total_particle_count; i++) {
    cudaMalloc(&particle_histories[i].x, floatarr_mem_size);
    cudaMalloc(&particle_histories[i].y, floatarr_mem_size);
    cudaMalloc(&particle_histories[i].z, floatarr_mem_size);
    cudaMalloc(&particle_histories[i].px, floatarr_mem_size);
    cudaMalloc(&particle_histories[i].py, floatarr_mem_size);
    cudaMalloc(&particle_histories[i].pz, floatarr_mem_size);
  }
}

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
                                 int total_particle_count, int latest_time) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_particle_count) {
    push_particle(particles[idx], e_field, b_field, timestep, charge, mass);
    particle_histories[idx].x[latest_time] = particles[idx].x;
    particle_histories[idx].y[latest_time] = particles[idx].y;
    particle_histories[idx].z[latest_time] = particles[idx].z;
    particle_histories[idx].px[latest_time] = particles[idx].px;
    particle_histories[idx].py[latest_time] = particles[idx].py;
    particle_histories[idx].pz[latest_time] = particles[idx].pz;
  }
}

void launch_update_particles(Particle *particles,
                             ParticleHistory *particle_histories,
                             float3 e_field, float3 b_field, float charge,
                             float mass, float timestep,
                             int total_particle_count, int iter) {
  const int threads_per_block = 256;
  const int blocks =
      (total_particle_count + threads_per_block - 1) / threads_per_block;
  update_particles<<<blocks, threads_per_block>>>(
      particles, particle_histories, e_field, b_field, charge, mass, timestep,
      total_particle_count, iter);
  cudaDeviceSynchronize();
}

int main() {
  // Choose number particles and dimension to simulate
  const float3 num_of_particle = make_float3(4.0f, 4.0f, 4.0f);
  const float3 system_length = make_float3(4.0f, 4.0f, 4.0f);
  const int total_particle_count = static_cast<int>(
      num_of_particle.x * num_of_particle.y * num_of_particle.z);

  // Initialize E and B fields
  const float3 e_field = make_float3(0.5f, 0.5f, 0.5f);
  const float3 b_field = make_float3(0.75f, 0.75f, 0.75f);
  const float charge = 1.0f;
  const float mass = 1.0f;

  // choose arbitrary timesteps
  const float timestep = 0.025f;

  // Number of Boris pusher iteration run
  const int max_iter = 100;

  // Declare and initialize particles with fixed data
  Particle *particles = new Particle[total_particle_count];
  ParticleHistory *particle_histories =
      new ParticleHistory[total_particle_count];
  initialize_particles(particles, num_of_particle, total_particle_count,
                       system_length, ParticlePlacementType::UNIFORM);
  initialize_particle_history(particle_histories, total_particle_count,
                              max_iter);

  // Allocate and copy particles to device
  int particle_mem_size = total_particle_count * sizeof(Particle);
  Particle *device_particles;
  cudaMalloc(&device_particles, particle_mem_size);
  cudaMemcpy(device_particles, particles, particle_mem_size,
             cudaMemcpyHostToDevice);

  // Allocate and copy particle_histories to device
  int particle_history_mem_size =
      total_particle_count * sizeof(ParticleHistory);
  ParticleHistory *device_particle_histories;
  cudaMalloc(&device_particle_histories, particle_history_mem_size);
  cudaMalloc_particle_history(device_particle_histories, total_particle_count,
                              max_iter);
  cudaMemcpy(device_particle_histories, particle_histories,
             particle_history_mem_size, cudaMemcpyHostToDevice);

  // Create events to time the memcpy transfer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for (int i = 0; i < max_iter; i++) {
    launch_update_particles(device_particles, device_particle_histories,
                            e_field, b_field, charge, mass, timestep,
                            total_particle_count, i);
  }
  cudaMemcpy(particle_histories, device_particle_histories,
             particle_history_mem_size, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float time_elapsed = 0;
  cudaEventElapsedTime(&time_elapsed, start, stop);

  std::cout << "Elapsed time: " << time_elapsed << "\n";

  // Destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // print particle
  for (int i = 0; i < total_particle_count; i++) {
    std::cout << "particle " << i << ": "
              << particle_histories[i].print(max_iter - 1) << "\n";
  }

  return 0.0;
}
