#include "../micrograd/nn.hpp"

#define INPUTS 3

typedef double TYPE;

int main() {
    // define the neural network
    auto n = MLP<TYPE, INPUTS>(INPUTS, {5,4});

    std::vector<std::vector<Value<TYPE>>> xs = {{2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5,1.0, 1.0,}, {1.0, 1.0, -1.0 }};

    // desired target
    std::vector<Value<TYPE>> ys = {1.0, -1.0, -1.0, 1.0};

    std::vector<Value<TYPE>> ypred;

    for (std::vector<Value<TYPE>> x : xs){
        ypred.emplace_back(n(x));
    }

    for (auto& results : ypred){
        std::cout<< results << '\n';
    }
}
