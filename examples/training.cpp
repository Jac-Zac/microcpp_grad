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
    Value_Vec<TYPE> tmp_loss;

    double step_size = 0.01;

    std::cout << "Starting Training\n";
    std::cout << "----------------------------\n\n";

    for (size_t x = 1; x <= 100; x++) {
        // Reset in case it is not the first loop
        ypred.clear();
        tmp_loss.clear();

        auto loss = Value<TYPE>(0, "loss");
        // Create a tmp variable that allows the full graph to be stored
        // Problem is with the ^ operator

        // Zero grad
        model.zero_grad();

        for (size_t i = 0; i < PASS_NUM; i++) {
            // Forward pass
            ypred.emplace_back(model(xs[i]));

            // Mean Squared Error
            tmp_loss.emplace_back(ypred[i][SIZE][0] - ys[i]);
            // This loss is not working also with other operation but the other is
            /* loss += (ypred[i][SIZE][0] - ys[i])^2; */
        }

        for(size_t i = 0; i < PASS_NUM; i++){
            loss += tmp_loss[i]^2;
        }

        // backward pass
        loss.backward();

        // Update parameters thanks to the gradient
        for (auto &p : model.parameters()) {
            // Update parameter value
            p->data += -step_size * (p->grad);
        }

        std::cout << "The loss at step: " << x << " is: " << loss.data << '\n';

        if (x == 100){
            loss.draw_graph();
        }
    }
}
