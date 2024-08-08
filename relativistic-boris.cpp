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
    Field eField {0.5, 0.5, 0.5};
    Field bField {0.75, 0.75, 0.75}; 

    // Initialize particle with fixed data
    Particle test = {0.25, 0.25, 0.25, 10, 10, 10};

    // Half-step momentum from Electric field
    float pxHalf = test.px + timestep * eField.x / 2;
    float pyHalf = test.py + timestep * eField.y / 2;
    float pzHalf = test.pz + timestep * eField.z / 2;

    // Lorentz factor for half-step momentum
    float lorentz = std::sqrt(1 + pxHalf*pxHalf + pyHalf*pyHalf + pzHalf*pzHalf);

    // Rotation vector from Magnetic field
    float tx = timestep * bField.x / 2*lorentz;
    float ty = timestep * bField.y / 2*lorentz;
    float tz = timestep * bField.z / 2*lorentz;
    float tMagSquare = tx*tx + ty*ty + tz*tz;

    // Cross product of half p and t
    float pxPrime = pxHalf + (pyHalf * tz - pzHalf * ty); 
    float pyPrime = pyHalf + (pzHalf * tx - pxHalf * tz); 
    float pzPrime = pzHalf + (pxHalf * ty - pyHalf * tx); 

    // Update momentum with effect from Boris rotation 
    float denominator = 1 + tMagSquare;
    float pxPlus = 2 * (pyPrime * tz - pzPrime * ty) / denominator;
    float pyPlus = 2 * (pzPrime * tx - pxPrime * tz) / denominator;
    float pzPlus = 2 * (pxPrime * ty - pyPrime * tx) / denominator;

    // Update momentum with effect from Electric field 
    test.px = pxPlus + timestep * eField.x / 2;
    test.py = pyPlus + timestep * eField.y / 2;
    test.px = pzPlus + timestep * eField.z / 2;

    // Lorentz factor for updated momentum
    lorentz = std::sqrt(1 + test.px*test.px + test.py*test.py + test.pz*test.pz);

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