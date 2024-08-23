#ifndef PARTICLE_HISTORY_H
#define PARTICLE_HISTORY_H

#include <string>

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
  void initialize(int max_time) {
    x = new float[max_time]();
    y = new float[max_time]();
    z = new float[max_time]();
    px = new float[max_time]();
    py = new float[max_time]();
    pz = new float[max_time]();
  }

  // Print position and momentum data at specified time
  const std::string print(int time) {
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

void initialize_particle_history(ParticleHistory *particle_histories,
                                 int total_particle_count,
                                 int max_iter);

void cudaMalloc_particle_history(ParticleHistory *device_particle_histories,
                                 int total_particle_count, int max_iter,
                                 int float_arr_mem_size);

void cudaMemcpy_particle_history(ParticleHistory *particle_histories,
                                 ParticleHistory *temp_particle_histories,
                                 int total_particle_count,
                                 int float_arr_mem_size);

void cudaFree_particle_history(ParticleHistory *device_particle_histories,
                               int total_particle_count);

#endif // PARTICLE_HISTORY_H
