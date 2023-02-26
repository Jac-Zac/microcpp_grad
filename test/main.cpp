#include "../micrograd/nn.hpp"

int main() {
    std::vector<double> x = {2.0, 3.0};
    auto l = Layer<double>(2,3);
    for(auto &output : l(x)){
        std::cout <<  output << '\n';
    }
}
