#include "../micrograd/nn.hpp"

#define NUMBER_OF_NEURONS 2

int main() {
    std::array<double, NUMBER_OF_NEURONS> x = {2.0, 3.0};
    auto n = Neuron<NUMBER_OF_NEURONS>();
    std::cout << n(x) << '\n';
}
