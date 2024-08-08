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
        return "x: " + std::to_string(x) + " y: " + std::to_string(y) + " z: " + std::to_string(z) +"px: " + std::to_string(px) + " py: " + std::to_string(py) + " pz: " + std::to_string(pz);
    }
};

struct Field {
    float x;
    float y;
    float z;
};

int main() {
    // Choose arbitrary timesteps
    float timestep = 0.025f; 

    // Initialize E and B fields
    Field eField {0.5f, 0.5f, 0.5f};
    Field bField {0.75f, 0.75f, 0.75f}; 

    // Initialize particle with fixed data
    Particle test = {0.25f, 0.25f, 0.25f, 10.0f, 10.0f, 10.0f};

    // Half-step momentum from Electric field
    test.px += timestep * eField.x / 2.0;
    test.py += timestep * eField.y / 2.0;
    test.pz += timestep * eField.z / 2.0;

    // Lorentz factor for half-step momentum
    float lorentz = std::sqrt(1.0 + test.px*test.px + test.py*test.py + test.pz*test.pz);

    // Rotation vector from Magnetic field
    float tx = timestep * bField.x / 2.0 * lorentz;
    float ty = timestep * bField.y / 2.0 * lorentz;
    float tz = timestep * bField.z / 2.0 * lorentz;
    float tMagSquare = tx*tx + ty*ty + tz*tz;

    // Cross product of half p and t
    float pxPrime = test.px + (test.py * tz - test.pz * ty); 
    float pyPrime = test.py + (test.pz * tx - test.px * tz); 
    float pzPrime = test.pz + (test.px * ty - test.py * tx); 

    // Update momentum with effect from Boris rotation and Electric field 
    float denominator = 1 + tMagSquare;
    test.px += (2.0 * pyPrime * tz - pzPrime * ty / denominator) + (timestep * eField.x / 2.0);
    test.py += (2.0 * pzPrime * tx - pxPrime * tz / denominator) + (timestep * eField.y / 2.0);
    test.pz += (2.0 * pxPrime * ty - pyPrime * tx / denominator) + (timestep * eField.y / 2.0);
 
    // Lorentz factor for updated momentum
    lorentz = std::sqrt(1.0 + test.px*test.px + test.py*test.py + test.pz*test.pz);

    // Calculate Velocity from momentum 
    float xVel = test.px / lorentz; 
    float yVel = test.py / lorentz; 
    float zVel = test.pz / lorentz; 

    // Update position
    test.x += xVel * timestep;
    test.y += yVel * timestep;
    test.z += zVel * timestep;

    // Print particle
    std::cout << test.print() << "\n";

    return 0.0;
}

