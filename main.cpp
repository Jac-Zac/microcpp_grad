#include "engine.hpp"

int main() {
    // Create two Value objects
    Value<double> a(1.0, 'a');
    Value<double> b(2.0, 'b');

    // Add the two Value objects
    Value<double> c = a + b;
    c.label = 'c';

    // Print the result
    std::cout << "c = " << c << std::endl;

    return 0;
}
