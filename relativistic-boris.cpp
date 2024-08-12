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

    std::string print() {
        return "x: " + std::to_string(x) + " y: " + std::to_string(y) + " z: " + std::to_string(z) +" px: " + std::to_string(px) + " py: " + std::to_string(py) + " pz: " + std::to_string(pz);
    }
};

struct Field {
    float x;
    float y;
    float z;
};

void push_particle(Particle &electron, const Field &e_field, const Field &b_field, float timestep) {
    // Half-step momentum from Electric field
    electron.px += timestep * e_field.x / 2.0f;
    electron.py += timestep * e_field.y / 2.0f;
    electron.pz += timestep * e_field.z / 2.0f;

    // Lorentz factor for half-step momentum
    float lorentz = std::sqrt(1.0f + electron.px*electron.px + electron.py*electron.py + electron.pz*electron.pz);

    // Rotation vector from Magnetic field
    float tx = timestep * b_field.x / (2.0f * lorentz);
    float ty = timestep * b_field.y / (2.0f * lorentz);
    float tz = timestep * b_field.z / (2.0f * lorentz);
    float t_mag_square = tx*tx + ty*ty + tz*tz;

    // Cross product of half p and t
    float px_prime = electron.px + (electron.py * tz - electron.pz * ty); 
    float py_prime = electron.py + (electron.pz * tx - electron.px * tz); 
    float pz_prime = electron.pz + (electron.px * ty - electron.py * tx); 

    // Update momentum with effect from Boris rotation and Electric field 
    float denominator = 1 + t_mag_square;
    electron.px += 2 * (py_prime * tz - pz_prime * ty) / denominator + timestep * e_field.x / 2;
    electron.py += 2 * (pz_prime * tx - px_prime * tz) / denominator + timestep * e_field.y / 2;
    electron.pz += 2 * (px_prime * ty - py_prime * tx) / denominator + timestep * e_field.z / 2;
 
    // Lorentz factor for updated momentum
    lorentz = std::sqrt(1.0f + electron.px*electron.px + electron.py*electron.py + electron.pz*electron.pz);

    // Update position using calculated velocity
    electron.x += timestep * electron.px / lorentz;
    electron.y += timestep * electron.py / lorentz;
    electron.z += timestep * electron.pz / lorentz;
}

int main() {
    // Choose arbitrary timesteps
    float timestep = 0.025f; 

    // Initialize E and B fields
    Field e_field {0.5f, 0.5f, 0.5f};
    Field b_field {0.75f, 0.75f, 0.75f}; 

    // Initialize particle with fixed data
    Particle test = {0.25f, 0.25f, 0.25f, 10.0f, 10.0f, 10.0f};
    
    push_particle(test, e_field, b_field, timestep);
    
    // Print particle
    std::cout << test.print() << "\n";

    return 0.0;
}

