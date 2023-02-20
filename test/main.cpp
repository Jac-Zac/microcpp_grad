#include "../micrograd/engine.hpp"

int main() {
    double h = 0.0001;
    // Create two Value objects
    Value<double> a(2.0, 'a');
    Value<double> b(-3.0, 'b');
    Value<double> c(10, 'd');

    // First test
    auto d = a*b + c;
    d.label = 'd';

    b = b + h;

    auto e = a*b + c;
    d.label = 'e';

    // Print the result
    std::cout << "d = " << d << std::endl;
    std::cout << "e = " << e << std::endl;
    std::cout << "Slope " << ((d - e)/h) << std::endl;

    return 0;
}
