#include <micrograd/nn.hpp>

using namespace nn;

#define SIZE 3
#define PASS_NUM 4

typedef double TYPE;

int main() {
    // Binary classification

    // define the neural network
    std::array<size_t, SIZE> n_neurons_for_layer = {4, 4, 1};
    auto model = MLP<TYPE, SIZE>(3, n_neurons_for_layer);

    std::vector<Value_Vec<TYPE>> xs = {
        {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};

    // desired target
    Value_Vec<TYPE> ys = {1.0, -1.0, -1.0, 1.0};

    std::cout << model; // to output the network shape

    std::cout << "\nThe network has: " << model.parameters().size()
              << " parameters\n\n";

    std::vector<std::vector<Value_Vec<TYPE>>> ypred;
    double step_size = 0.01;

    std::cout << "Starting Training\n";
    std::cout << "----------------------------\n\n";

    for (size_t x = 1; x <= 1; x++) {
        // Reset in case it is not the first loop
        auto loss = Value<TYPE>(0, "loss");
        ypred.clear();

        for (size_t i = 0; i < PASS_NUM; i++) {
            // Forward pass
            ypred.emplace_back(model(xs[i]));

            std::cout << (ypred[i][SIZE][0] - ys[i]) << '\n';

            // Mean Squared Error
            loss += (ypred[i][SIZE][0] - ys[i]) ^ 2;
            // backward pass
            loss.backward();
        }

        loss.draw_graph();

        // Update parameters thanks to the gradient
        for (auto &p : model.parameters()) {
            // Update parameter value
            p->data += -step_size * (p->grad);
        }

        std::cout << "The loss at step: " << x << " is: " << loss.data << '\n';

        // Zero grad
        model.zero_grad();
    }
}
