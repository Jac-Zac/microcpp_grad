#include "../micrograd/nn.hpp"

#define SIZE 2

int main() {

    std::array<double, SIZE> x = {2.0, 3.0};
    auto n = Neuron<SIZE>();
    std::cout << n(x) << '\n';
}
