#include "../micrograd/nn.hpp"

#define SIZE 2

void test() {

    std::array<double, SIZE> x = {2.0, 3.0};
    auto n = Neuron<SIZE>();
    n(x);
}
