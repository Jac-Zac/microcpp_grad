#include "../micrograd/engine.hpp"
#include <__tuple>

int main() {
    double h = 0.0001;
    // Create two Value objects
    Value<double> a(2.0, 'a');
    Value<double> b(-3.0, 'b');
    Value<double> c(10, 'd');

    // First test
    auto d = a * b + c;
    d.label = 'd';

    a += h;

    auto e = a * b + c;
    e.label = 'e';

    // Print the result
    std::cout << "d = " << d << std::endl;
    std::cout << "e = " << e << std::endl;
    std::cout << "Slope " << ((e - d) / h) << std::endl;

    // getter function to print to screen the prev
    d.get_prev();

    return 0;
}
