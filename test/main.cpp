#include "../micrograd/engine.hpp"

int main() {
    /* auto a = Value<double>(3.0, "a"); */
    /* auto b = a + a; */
    /* b.label = "b"; */
    /*  */
    /* // Grandina with respect to itself is 1 */
    /* b.backprop(); */
    /* b.print_graph(); */
    auto a = Value<double>(2.0, "a");
    auto b = Value<double>(4.0, "b");
    auto c = a + b; c.label = 'c';
    auto d = a + 4.0; d.label = 'd';
    std::cout << *(d.m_prev[0]) << '\n';
    std::cout << d.m_prev[0] << '\n';
    std::cout << &a << '\n';
    // This hash to print the value object a
    d.print_graph();
}
