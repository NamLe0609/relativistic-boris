#include <vector>
#include <math.h>
#include <string>
#include <iostream>

struct Particle
{
    float x;
    float y;
    float z;

    float px;
    float py;
    float pz;

    // Default constructor
    Particle() : x(), y(), z(), px(), py(), pz() {}

    void update(float x_pos, float y_pos, float z_pos, float x_momentum, float y_momentum, float z_momentum)
    {
        x = x_pos;
        y = y_pos;
        z = z_pos;
        px = x_momentum;
        py = y_momentum;
        pz = z_momentum;
    }

    std::string print()
    {
        return "x: " + std::to_string(x) + " y: " + std::to_string(y) + " z: " + std::to_string(z) + " px: " + std::to_string(px) + " py: " + std::to_string(py) + " pz: " + std::to_string(pz);
    }
};

struct ParticleHistory
{
    float *x;
    float *y;
    float *z;
    float *px;
    float *py;
    float *pz;

    // Default constructor
    ParticleHistory() : x(nullptr), y(nullptr), z(nullptr), px(nullptr), py(nullptr), pz(nullptr) {}

    // Initialize the variable lists of size max_time
    ParticleHistory(int max_time)
    {
        x = new float[max_time];
        y = new float[max_time];
        z = new float[max_time];
        px = new float[max_time];
        py = new float[max_time];
        pz = new float[max_time];
    }

    ~ParticleHistory()
    {
        delete[] x;
        delete[] y;
        delete[] z;
        delete[] px;
        delete[] py;
        delete[] pz;
    }
};

struct Vec3
{
    float x;
    float y;
    float z;

    // Initializes the field attributes
    Vec3(float x_force, float y_force, float z_force) : x(x_force), y(y_force), z(z_force) {}
};

enum class ParticlePlacementType {
    UNIFORM, // Evenly spaced integer placement for xyz coordinates
};

void initialize_particles(Particle *particles, Vec3 num_of_particles, int total_particle_count, Vec3 system_length, ParticlePlacementType placement_type)
{
    switch (placement_type)
    {
    case ParticlePlacementType::UNIFORM:
        float deltax = system_length.x / num_of_particles.x;
        float deltay = system_length.y / num_of_particles.y;
        float deltaz = system_length.z / num_of_particles.z;
        for (int i = 0; i < total_particle_count; i++)
        {
            float corrected_index = static_cast<float>(i) + 0.5f;
            particles[i].update(deltax * corrected_index, 
            deltay * corrected_index,
            deltaz * corrected_index, 
            10, 10, 10);
        }
        break;
    }
}

void initialize_particle_history(ParticleHistory *particle_histories, int total_particle_count, int max_iter)
{
    for (int i = 0; i < total_particle_count; i++)
    {
        particle_histories[i] = ParticleHistory(max_iter);
    }
}

void push_particle(Particle &particle, const Vec3 &e_field, const Vec3 &b_field, const float timestep, const float charge, float mass)
{
    // Pre-compute reused calculations
    float mass_square = mass * mass;
    float timestep_charge = charge * timestep;

    // half-step momentum from electric field
    float px_half = particle.px + timestep_charge * e_field.x / 2.0f;
    float py_half = particle.py + timestep_charge * e_field.y / 2.0f;
    float pz_half = particle.pz + timestep_charge * e_field.z / 2.0f;

    // lorentz factor for half-step momentum
    float lorentz = sqrtf(1.0f + px_half * px_half + py_half * py_half + pz_half * pz_half / mass_square);

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
    float px_updated = px_half + 2 * (py_prime * tz - pz_prime * ty) / denominator + timestep_charge * e_field.x / 2.0f;
    float py_updated = py_half + 2 * (pz_prime * tx - px_prime * tz) / denominator + timestep_charge * e_field.y / 2.0f;
    float pz_updated = pz_half + 2 * (px_prime * ty - py_prime * tx) / denominator + timestep_charge * e_field.z / 2.0f;

    // lorentz factor for updated momentum
    lorentz = sqrtf(1.0f + px_updated * px_updated + py_updated * py_updated + pz_updated * pz_updated / mass_square);

    // update position using calculated velocity
    float lorentz_mass = mass * lorentz; // Pre-compute to avoid recalculating
    float x_updated = particle.x + timestep * px_updated / lorentz_mass;
    float y_updated = particle.y + timestep * py_updated / lorentz_mass;
    float z_updated = particle.z + timestep * pz_updated / lorentz_mass;

    // update the particle with new location and momentum
    particle.update(x_updated, y_updated, z_updated, px_updated, py_updated, pz_updated);
}

void update_particles(Particle *particles, ParticleHistory *particle_histories, Vec3 e_field, Vec3 b_field, float timestep, int total_particle_count, int latest_time)
{   
    for (int i = 0; i < total_particle_count; i++)
    {
        push_particle(particles[i], e_field, b_field, timestep, 1, 1);
        particle_histories[i].x[latest_time] = particles[i].x;
        particle_histories[i].y[latest_time] = particles[i].y;
        particle_histories[i].z[latest_time] = particles[i].z;
        particle_histories[i].px[latest_time] = particles[i].px;
        particle_histories[i].py[latest_time] = particles[i].py;
        particle_histories[i].pz[latest_time] = particles[i].pz;
    }
}

int main()
{
    // choose arbitrary timesteps
    const float timestep = 0.025f;

    // Choose number particles and dimension to simulate
    const Vec3 num_of_particle = {4, 4, 4};
    const Vec3 system_length = {4, 4, 4};
    const int total_particle_count = static_cast<int>(num_of_particle.x * num_of_particle.y * num_of_particle.z);

    // Number of Boris pusher iteration run
    const int max_iter = 100;

    // Initialize E and B fields
    const Vec3 e_field(0.5, 0.5, 0.5);
    const Vec3 b_field(0.75, 0.75, 0.75);

    // Declare and initialize particles with fixed data
    Particle *particles = new Particle[total_particle_count];
    ParticleHistory *particle_histories = new ParticleHistory[total_particle_count];
    initialize_particles(particles, num_of_particle, total_particle_count, system_length, ParticlePlacementType::UNIFORM);
    initialize_particle_history(particle_histories, total_particle_count, max_iter);
    
    // // Print particle initialize
    // for (int i = 0; i < total_particle_count; i++)
    // {
    //     std::cout << "Particle " << i << ": " << particles[i].print() << "\n";
    // }

    // Update particles with n iteration of Boris pusher
    for (int loop = 0; loop < max_iter; loop++)
    {
        update_particles(particles, particle_histories, e_field, b_field, timestep, total_particle_count, loop);
    }

    // // Print particle
    // for (int i = 0; i < total_particle_count; i++)
    // {
    //     std::cout << "Particle " << i << ": " << particles[i].print() << "\n";
    // }
    return 0.0;
}