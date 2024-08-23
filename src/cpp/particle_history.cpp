#include "particle_history.h"
#include <cuda_runtime.h>

void initialize_particle_history(ParticleHistory *particle_histories,
                                 int total_particle_count,
                                 int max_iter) {
  // Initialize for all particles
  for (int i = 0; i < total_particle_count; i++) {
    particle_histories[i].initialize(max_iter);
  }
}

void cudaMalloc_particle_history(ParticleHistory *device_particle_histories,
                                 int total_particle_count, int max_iter,
                                 int float_arr_mem_size) {
  for (int i = 0; i < total_particle_count; i++) {
    float *device_x, *device_y, *device_z, *device_px, *device_py, *device_pz;

    // Allocate memory for each array on the device
    cudaMalloc(&device_x, float_arr_mem_size);
    cudaMalloc(&device_y, float_arr_mem_size);
    cudaMalloc(&device_z, float_arr_mem_size);
    cudaMalloc(&device_px, float_arr_mem_size);
    cudaMalloc(&device_py, float_arr_mem_size);
    cudaMalloc(&device_pz, float_arr_mem_size);

    // Copy from host the pointer of the float arrays
    cudaMemcpy(&(device_particle_histories[i].x), &device_x, sizeof(float *),
               cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_particle_histories[i].y), &device_y, sizeof(float *),
               cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_particle_histories[i].z), &device_z, sizeof(float *),
               cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_particle_histories[i].px), &device_px, sizeof(float *),
               cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_particle_histories[i].py), &device_py, sizeof(float *),
               cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_particle_histories[i].pz), &device_pz, sizeof(float *),
               cudaMemcpyHostToDevice);
  }
}

void cudaMemcpy_particle_history(ParticleHistory *particle_histories,
                                 ParticleHistory *temp_particle_histories,
                                 int total_particle_count,
                                 int float_arr_mem_size) {
  for (int i = 0; i < total_particle_count; i++) {
    // Copy from host the pointer of the float arrays
    cudaMemcpy(particle_histories[i].x, temp_particle_histories[i].x,
               float_arr_mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_histories[i].y, temp_particle_histories[i].y,
               float_arr_mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_histories[i].z, temp_particle_histories[i].z,
               float_arr_mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_histories[i].px, temp_particle_histories[i].px,
               float_arr_mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_histories[i].py, temp_particle_histories[i].py,
               float_arr_mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_histories[i].pz, temp_particle_histories[i].pz,
               float_arr_mem_size, cudaMemcpyDeviceToHost);
  }
}

void cudaFree_particle_history(ParticleHistory *device_particle_histories,
                               int total_particle_count) {
  for (int i = 0; i < total_particle_count; i++) {
    cudaFree(device_particle_histories[i].x);
    cudaFree(device_particle_histories[i].y);
    cudaFree(device_particle_histories[i].z);
    cudaFree(device_particle_histories[i].px);
    cudaFree(device_particle_histories[i].py);
    cudaFree(device_particle_histories[i].pz);
  }
}
