#include "../micrograd/nn.hpp"

int main() {
    std::vector<double> x = {2.0, -4.5};
    auto n = Neuron<double>(2);
    std::cout << n(x) << '\n';
}
