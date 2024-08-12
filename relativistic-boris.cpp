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

void relativisticBoris(Particle &electron, const Field &eField, const Field &bField, float timestep) {
    // Half-step momentum from Electric field
    electron.px += timestep * eField.x / 2.0f;
    electron.py += timestep * eField.y / 2.0f;
    electron.pz += timestep * eField.z / 2.0f;

    // Lorentz factor for half-step momentum
    float lorentz = std::sqrt(1.0f + electron.px*electron.px + electron.py*electron.py + electron.pz*electron.pz);

    // Rotation vector from Magnetic field
    float tx = timestep * bField.x / (2.0f * lorentz);
    float ty = timestep * bField.y / (2.0f * lorentz);
    float tz = timestep * bField.z / (2.0f * lorentz);
    float tMagSquare = tx*tx + ty*ty + tz*tz;

    // Cross product of half p and t
    float pxPrime = electron.px + (electron.py * tz - electron.pz * ty); 
    float pyPrime = electron.py + (electron.pz * tx - electron.px * tz); 
    float pzPrime = electron.pz + (electron.px * ty - electron.py * tx); 

    // Update momentum with effect from Boris rotation and Electric field 
    float denominator = 1 + tMagSquare;
    electron.px += 2 * (pyPrime * tz - pzPrime * ty) / denominator + timestep * eField.x / 2;
    electron.py += 2 * (pzPrime * tx - pxPrime * tz) / denominator + timestep * eField.y / 2;
    electron.pz += 2 * (pxPrime * ty - pyPrime * tx) / denominator + timestep * eField.z / 2;
 
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
    Field eField {0.5f, 0.5f, 0.5f};
    Field bField {0.75f, 0.75f, 0.75f}; 

    // Initialize particle with fixed data
    Particle test = {0.25f, 0.25f, 0.25f, 10.0f, 10.0f, 10.0f};
    
    relativisticBoris(test, eField, bField, timestep);
    
    // Print particle
    std::cout << test.print() << "\n";

    return 0.0;
}

