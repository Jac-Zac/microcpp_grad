#include <micrograd/nn.hpp>

using namespace nn;

#define SIZE 3
#define PASS_NUM 2

typedef double TYPE;

int main() {
    // Binary classification

    // define the neural network
    std::array<size_t, SIZE> n_neurons_for_layer = {4, 4, 1};
    auto model = MLP<TYPE, SIZE>(3, n_neurons_for_layer);

    // std::cout << n; // to output the network shape

    /* std::vector<Value_Vec<TYPE>> xs = { */
    /*     {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0,
     * -1.0}}; */
    std::vector<Value_Vec<TYPE>> xs = {{2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}};

    // desired target
    Value_Vec<TYPE> ys = {1.0, -1.0};

    std::vector<std::vector<Value_Vec<TYPE>>> ypred;

    auto loss = Value<TYPE>(0, "loss");

    for (size_t i = 0; i < PASS_NUM; i++) {
        ypred.emplace_back(model(xs[i]));
        // Mean Squared Error
        loss += (ypred[i][SIZE][0] - ys[i]) ^ 2;
        loss.backward();
        std::cout << loss << '\n';
        model.zero_grad();
    }

    std::cout << "The network hash: " << model.parameters().size() << " parameters\n";

    loss.draw_graph();
}
