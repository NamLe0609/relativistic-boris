#include <vector>
#include <math.h>
#include <string>
#include <iostream>

struct Particle {
    float x;
    float y;
    float z;

    float px;
    float py;
    float pz;

    // Initializes the particle attributes
    Particle(float x_pos, float y_pos, float z_pos, float x_momentum, float y_momentum, float z_momentum)
    : x(x_pos), y(y_pos), z(z_pos), px(x_momentum), py(y_momentum), pz(z_momentum) {}

    std::string print() {
        return "x: " + std::to_string(x) + " y: " + std::to_string(y) + " z: " + std::to_string(z) +" px: " + std::to_string(px) + " py: " + std::to_string(py) + " pz: " + std::to_string(pz);
    }
};

struct Field {
    float x;
    float y;
    float z;

    // Initializes the field attributes
    Field(float x_force, float y_force, float z_force): x(x_force), y(y_force), z(z_force) {}
};

Particle push_particle(const Particle &electron, const Field &e_field, const Field &b_field, float timestep) {
    // Half-step momentum from Electric field
    float px_half = electron.px + timestep * e_field.x / 2.0f;
    float py_half = electron.py + timestep * e_field.y / 2.0f;
    float pz_half = electron.pz + timestep * e_field.z / 2.0f;

    // Lorentz factor for half-step momentum
    float lorentz = std::sqrt(1.0f + px_half*px_half + py_half*py_half + pz_half*pz_half);

    // Rotation vector from Magnetic field
    float tx = timestep * b_field.x / (2.0f * lorentz);
    float ty = timestep * b_field.y / (2.0f * lorentz);
    float tz = timestep * b_field.z / (2.0f * lorentz);
    float t_mag_square = tx*tx + ty*ty + tz*tz;

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
    lorentz = std::sqrt(1.0f + electron.px*electron.px + electron.py*electron.py + electron.pz*electron.pz);

    // Update position using calculated velocity
    float x_updated = electron.x + timestep * electron.px / lorentz;
    float y_updated = electron.y + timestep * electron.py / lorentz;
    float z_updated = electron.z + timestep * electron.pz / lorentz;

    return Particle(x_updated, y_updated, z_updated, px_updated, py_updated, pz_updated);
}

int main() {
    // Choose arbitrary timesteps
    float timestep = 0.025f; 

    // Initialize E and B fields
    Field e_field(0.5, 0.5, 0.5);
    Field b_field(0.75, 0.75, 0.75);

    // Initialize particles with fixed data
    std::vector<Particle> particles;
    particles.reserve(256);
    for (size_t i = 0; i < 16; i++) {
        for (size_t j = 0; j < 16; j++) {
            // Static cast to prevent implicit conversion
            particles.push_back(Particle(static_cast<float>(i), static_cast<float>(j), 0, 10.0, 10.0, 10.0));
        }
    } 

    // Update particles with n iteration of Boris pusher
    size_t max_iter = 1;
    for (size_t i = 0; i < max_iter; i++) {
        for (Particle &p: particles) {
            p = push_particle(p, e_field, b_field, timestep); 
        } 
    }
    
    // Print particle
    for (Particle p: particles) {
        std::cout << p.print() << "\n";
    }

    return 0.0;
}

