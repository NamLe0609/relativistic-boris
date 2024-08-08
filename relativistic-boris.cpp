#include <vector>
#include <math.h>
#include <string>
#include <iostream>

template <typename T>
struct Particle {
    T x;
    T y;
    T z;

    T px;
    T py;
    T pz;

    std::string print() {
        return "x: " + std::to_string(x) + " y: " + std::to_string(y) + " z: " + std::to_string(z) +"px: " + std::to_string(px) + " py: " + std::to_string(py) + " pz: " + std::to_string(pz);
    }
};

template <typename T>
struct Field {
    T x;
    T y;
    T z;
};

template <typename T>
void relativisticBoris(Particle<T> &electron, const Field<T> &eField, const Field<T> &bField, T timestep) {
    // Half-step momentum from Electric field
    electron.px += timestep * eField.x / (T) 2.0;
    electron.py += timestep * eField.y / (T) 2.0;
    electron.pz += timestep * eField.z / (T) 2.0;

    // Lorentz factor for half-step momentum
    T lorentz = std::sqrt(1.0f + electron.px*electron.px + electron.py*electron.py + electron.pz*electron.pz);

    // Rotation vector from Magnetic field
    T tx = timestep * bField.x / ((T) 2.0 * lorentz);
    T ty = timestep * bField.y / ((T) 2.0 * lorentz);
    T tz = timestep * bField.z / ((T) 2.0 * lorentz);
    T tMagSquare = tx*tx + ty*ty + tz*tz;

    // Cross product of half p and t
    T pxPrime = electron.px + (electron.py * tz - electron.pz * ty); 
    T pyPrime = electron.py + (electron.pz * tx - electron.px * tz); 
    T pzPrime = electron.pz + (electron.px * ty - electron.py * tx); 

    // Update momentum with effect from Boris rotation and Electric field 
    T denominator = 1 + tMagSquare;
    electron.px += 2 * (pyPrime * tz - pzPrime * ty) / denominator + timestep * eField.x / 2;
    electron.py += 2 * (pzPrime * tx - pxPrime * tz) / denominator + timestep * eField.y / 2;
    electron.pz += 2 * (pxPrime * ty - pyPrime * tx) / denominator + timestep * eField.z / 2;
 
    // Lorentz factor for updated momentum
    lorentz = std::sqrt((T) 1.0 + electron.px*electron.px + electron.py*electron.py + electron.pz*electron.pz);

    // Update position using calculated velocity
    electron.x += timestep * electron.px / lorentz;
    electron.y += timestep * electron.py / lorentz;
    electron.z += timestep * electron.pz / lorentz;
}

int main() {
    // Choose arbitrary timesteps
    double timestep = 0.025; 

    // Initialize E and B fields
    Field<double> eField {0.5, 0.5, 0.5};
    Field<double> bField {0.75, 0.75, 0.75}; 

    // Initialize particle with fixed data
    Particle<double> test = {0.25, 0.25, 0.25, 10.0, 10.0, 10.0};
    
    relativisticBoris(test, eField, bField, timestep);
    
    // Print particle
    std::cout << test.print() << "\n";

    return 0.0;
}

