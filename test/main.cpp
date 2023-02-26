#include "../micrograd/nn.hpp"

#define NUMBER_OF_NEURONS 2

int main() {
    std::vector<double> x = {2.0, 3.0};
    auto n = Neuron<double>();
    std::cout << n(x) << '\n';
}
