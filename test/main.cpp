#include "../micrograd/engine.hpp"
#include <__tuple>

int main() {
    double h = 0.0001;
    // Create two Value objects
    Value<double> a(2.0, "a");
    Value<double> b(-3.0, "b");
    Value<double> c(10, "d");

    // First test
    auto d1 = a * b + c;
    d1.label = "d1";

    // Print the result
    std::cout << "result: " << d1 << std::endl;

    // getter function to print to screen the prev
    d1.get_prev();

    return 0;
}
