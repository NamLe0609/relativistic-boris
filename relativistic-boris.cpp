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

    // Initializes the particle attributes
    Particle(float x_pos, float y_pos, float z_pos, float x_momentum, float y_momentum, float z_momentum)
        : x(x_pos), y(y_pos), z(z_pos), px(x_momentum), py(y_momentum), pz(z_momentum) {}

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
    float* x;
    float* y;
    float* z;
    float* px;
    float* py;
    float* pz;
    ParticleHistory(int max_time)
    {
        x = new float[max_time];
        y = new float[max_time];
        z = new float[max_time];
        px = new float[max_time];
        py = new float[max_time];
        pz = new float[max_time];
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

struct Field
{
    float x;
    float y;
    float z;

    // Initializes the field attributes
    Field(float x_force, float y_force, float z_force) : x(x_force), y(y_force), z(z_force) {}
};

void initialize_particles(Particle *particles, int num_of_particles)
{
    for (int i = 0; i < num_of_particles; i++) {
        // Placeholder particle placement routine 
        for (int x = 0; x < 16; x++) {
            for (int y = 0; y < 16; y++) {
                particles[i] = Particle(static_cast<float>(x), static_cast<float>(y), 0.0, 10.0, 10.0, 10.0);
            }
        }
    }
}

void initialize_particle_history(ParticleHistory* particle_histories, int num_of_particles, int max_iter) {
    for (int i = 0; i < num_of_particles; i++) {
        particle_histories[i] = ParticleHistory(max_iter);
    }
}

void push_particle(Particle &electron, const Field &e_field, const Field &b_field, float timestep)
{
    // Half-step momentum from Electric field
    float px_half = electron.px + timestep * e_field.x / 2.0f;
    float py_half = electron.py + timestep * e_field.y / 2.0f;
    float pz_half = electron.pz + timestep * e_field.z / 2.0f;

    // Lorentz factor for half-step momentum
    float lorentz = std::sqrt(1.0f + px_half * px_half + py_half * py_half + pz_half * pz_half);

    // Rotation vector from Magnetic field
    float tx = timestep * b_field.x / (2.0f * lorentz);
    float ty = timestep * b_field.y / (2.0f * lorentz);
    float tz = timestep * b_field.z / (2.0f * lorentz);
    float t_mag_square = tx * tx + ty * ty + tz * tz;

    // Cross product of half p and t
    float px_prime = px_half + (py_half * tz - pz_half * ty);
    float py_prime = py_half + (pz_half * tx - px_half * tz);
    float pz_prime = pz_half + (px_half * ty - py_half * tx);

    // Update momentum with effect from Boris rotation and Electric field
    float denominator = 1 + t_mag_square;
    float px_updated = px_half + 2 * (py_prime * tz - pz_prime * ty) / denominator + timestep * e_field.x / 2;
    float py_updated = py_half + 2 * (pz_prime * tx - px_prime * tz) / denominator + timestep * e_field.y / 2;
    float pz_updated = pz_half + 2 * (px_prime * ty - py_prime * tx) / denominator + timestep * e_field.z / 2;

    // Lorentz factor for updated momentum
    lorentz = std::sqrt(1.0f + px_updated * px_updated + py_updated * py_updated + pz_updated * pz_updated);

    // Update position using calculated velocity
    float x_updated = electron.x + timestep * px_updated / lorentz;
    float y_updated = electron.y + timestep * py_updated / lorentz;
    float z_updated = electron.z + timestep * pz_updated / lorentz;

    // Update the electron with new location and momentum
    electron.update(x_updated, y_updated, z_updated, px_updated, py_updated, pz_updated);
}

void update_particles(Particle* particles, ParticleHistory* particle_histories, Field e_field, Field b_field, float timestep, int num_of_particles, int latest_time) {
    for (int i = 0; i < num_of_particles; i++) {
        push_particle(particles[i], e_field, b_field, timestep);
        particle_histories[i].x[latest_time] = particles[i].x;
    }
}


int main()
{
    // Choose arbitrary timesteps
    float timestep = 0.025f;

    // Choose particles to simulate
    const int num_of_particles = 256;

    // Number of Boris pusher iteration run
    int max_iter = 100;

    // Initialize E and B fields
    Field e_field(0.5, 0.5, 0.5);
    Field b_field(0.75, 0.75, 0.75);

    // Declare and initialize particles with fixed data
    Particle* particles;
    ParticleHistory* particle_histories;
    initialize_particles(particles, num_of_particles);
    initialize_particle_history(particle_histories, num_of_particles, max_iter);

    // Update particles with n iteration of Boris pusher
    for (int loop = 0; loop < max_iter; loop++)
    {
        update_particles(particles, particle_histories, e_field, b_field, timestep, num_of_particles, loop);
    }

    // Print particle
    for (int i = 0; i < num_of_particles; i++) {
        std::cout << particles[i].print() << "\n";
    }
    return 0.0;
}
