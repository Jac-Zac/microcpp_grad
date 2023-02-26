#include "../micrograd/nn.hpp"

#define INPUTS 3

typedef double TYPE;

int main() {
    // Binary classification

    // define the neural network
    auto n = MLP<TYPE, INPUTS>(INPUTS, {4, 4, 1});

    std::vector<std::vector<Value<TYPE>>> xs = {{2.0, 3.0, -1.0},
                                                {3.0, -1.0, 0.5},
                                                {0.5, 1.0, 1.0},
                                                {1.0, 1.0, -1.0}};

    // desired target
    std::vector<Value<TYPE>> ys = {1.0, -1.0, -1.0, 1.0};

    std::vector<std::vector<Value<TYPE>>> ypred;

    for (auto& x : xs) {
        ypred.emplace_back(n(x));
    }

    for (auto &value : ypred) {
        std::cout << value[0] << '\n';
        value[0].draw_graph();
    }
}
