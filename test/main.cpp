#include "../micrograd/engine.hpp"

int main() {
    // Create two Value objects
    Value<double> a(3.5, 'a');
    Value<double> b(2.5, 'b');
    Value<double> d(0.1, 'd');

    // Add the two Value objects
    Value<double> c = a + b;
    c.label = 'c';

    a = a + d;

    // Print the result
    std::cout << "a = " << a << std::endl;
    std::cout << "c = " << c << std::endl;
    std::cout << "prev 0 which should be a = " << *(c.m_prev[0]) << std::endl;

    return 0;
}
