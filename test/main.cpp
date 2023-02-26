#include "../micrograd/nn.hpp"

#define INPUTS 3

typedef double TYPE;

int main() {
    // Binary classification

    // define the neural network
    auto n = MLP<TYPE, INPUTS>(INPUTS, {4, 4, 1});

    // std::cout << n; // to output the network shape

    std::vector<std::vector<Value<TYPE>>> xs = {{2.0, 3.0, -1.0},
                                                {3.0, -1.0, 0.5},
                                                {0.5, 1.0, 1.0},
                                                {1.0, 1.0, -1.0}};

    // desired target
    std::vector<Value<TYPE>> ys = {1.0, -1.0, -1.0, 1.0};

    std::vector<Value<TYPE>> ypred;

    // Run the 4 different example through the network
    for(size_t i = 0; i < 4 ; i++){
        ypred.emplace_back(n(xs[i])[0]);
    }

    for (auto &value : ypred) {
        std::cout << value << '\n';
    }
}
