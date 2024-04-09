#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

class SingleLayerPerceptron {
private:
    [[maybe_unused]] const int dimension;
    const int num_classes;

    std::vector<std::vector<double>> weights;
    std::vector<double> bias_weight;
    double learning_rate;

    const double theta;

    /**
 * This function calculates the output of the activation function for a given data point, weights, and bias.
 * It calculates the dot product of the data and weights, adds the bias, and then applies the activation function.
 * The activation function is a step function that returns 1 if the net input is greater than a threshold theta,
 * -1 if the net input is less than theta - 1, and 0 otherwise.
 *
 * @param data A vector of integers representing a data point from the dataset.
 * @param weight A vector of doubles representing the weights of the model.
 * @param bias A double representing the bias of the model.
 * @return An integer representing the output of the activation function.
 */
    [[nodiscard]] int act_func(const std::vector<int> &data, const std::vector<double> &weight, double bias) const {
        // Calcula o produto escalar entre os dados e os pesos, e adiciona o bias
        double net = std::inner_product(data.begin(), data.end(), weight.begin(), bias);

        // Aplica a função de ativação e retorna o resultado
        return (net > theta) ? 1 : ((net >= theta - 1) ? 0 : -1);
    }

    /**
 * This function is responsible for updating the weights and bias of the model based on the difference between the predicted and actual output.
 * It checks if the predicted output is not equal to the actual output and if the learning rate and the actual output are not zero.
 * If these conditions are met, it updates the weights by adding the product of the learning rate, the actual output, and the data value to the current weight.
 * It also updates the bias by adding the product of the learning rate and the actual output to the current bias.
 *
 * @param data A vector of integers representing a data point from the dataset.
 * @param target An integer representing the actual output for the data point.
 * @param output An integer representing the predicted output for the data point.
 * @param weight A vector of doubles representing the weights of the model.
 * @param bias A double representing the bias of the model.
 */
    void ch_weights(const std::vector<int> &data, int target, int output, std::vector<double> &weight, double &bias) const {
        // Verifica se a saída prevista não é igual à saída esperada e se a taxa de aprendizagem e a saída esperada não são zero.
        if (output != target && learning_rate != 0 && target != 0) {
            // Atualiza os pesos adicionando o produto da taxa de aprendizagem, a saída esperada e o valor dos dados ao peso atual.
            std::transform(data.begin(), data.end(), weight.begin(), weight.begin(),
                           [&](int data_val, double weight_val) {
                               return weight_val + learning_rate * target * data_val;
                           });
            // Atualiza o bias adicionando o produto da taxa de aprendizagem e a saída real ao bias atual.
            bias += learning_rate * target;
        }
    }

    /**
 * This function is responsible for training the perceptron model.
 * It iterates over the dataset and updates the weights and bias of the model based on the difference between the predicted and actual output.
 * The function returns a boolean indicating whether the weights of the model were changed during the training process.
 *
 * @param dataset A 2D vector of integers representing the training data.
 * @param target A 2D vector of integers representing the target output for each data point in the dataset.
 * @return A boolean indicating whether the weights of the model were changed during the training process.
 */
    bool internal_train(const std::vector<std::vector<int>> &dataset, const std::vector<std::vector<int>> &target) {
        // Initialize a boolean to keep track of whether the weights were changed during the training process
        bool weights_changed = false;

        // Create an iterator for the target vector
        auto target_iter = target.begin();

        // Iterate over the dataset
        for (const auto &data: dataset) {
            // For each data point, iterate over the number of classes
            for (int i = 0; i < num_classes; ++i) {
                // Calculate the output of the activation function for the current data point and weights
                int output = act_func(data, weights[i], bias_weight[i]);

                // Update the weights and bias based on the difference between the predicted and actual output
                ch_weights(data, (*target_iter)[i], output, weights[i], bias_weight[i]);

                // If the predicted output does not match the actual output, set weights_changed to true
                if (output != (*target_iter)[i]) {
                    weights_changed = true;
                }
            }

            // Move to the next target output
            ++target_iter;
        }

        // Return whether the weights were changed during the training process
        return weights_changed;
    }

public:
    SingleLayerPerceptron(int dimension, int num_classes, double learning_rate, double theta)
            : dimension(dimension), num_classes(num_classes), learning_rate(learning_rate), theta(theta),
              weights(num_classes, std::vector<double>(dimension, 0.0)),
              bias_weight(num_classes, 0.0) {}

    void train(const std::vector<std::vector<int>> &dataset, const std::vector<std::vector<int>> &target) {
        while (internal_train(dataset, target));
    }

    std::vector<int> predict(const std::vector<int> &data) {
        std::vector<int> output(num_classes);
        for (int i = 0; i < num_classes; ++i) {
            output[i] = act_func(data, weights[i], bias_weight[i]);
        }
        return output;
    }

    void print_weights() const {
        int neuron_num = 1;
        for (const auto &weight: weights) {
            std::cout << "Neuronio " << neuron_num++ << ":" << std::endl;
            std::cout << "Peso: ";
            std::copy(weight.begin(), weight.end(), std::ostream_iterator<double>(std::cout, ", "));
            std::cout << std::endl;
            std::cout << "Peso do bias: " << bias_weight[neuron_num - 2] << std::endl;
        }
    }
};

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> readData(const std::string& filename, int num_data_columns) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<int>> data;
    std::vector<std::vector<int>> labels;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string value;
        std::vector<int> row;
        std::vector<int> label;

        for (int i = 0; i < num_data_columns; ++i) {
            std::getline(ss, value, ',');
            // Check for and remove BOM
            if (!value.empty() && value[0] == '\xEF' && value[1] == '\xBB' && value[2] == '\xBF') {
                value = value.substr(3);
            }
            row.push_back(std::stoi(value));
        }

        while (std::getline(ss, value, ',')) {
            // Check for and remove BOM
            if (!value.empty() && value[0] == '\xEF' && value[1] == '\xBB' && value[2] == '\xBF') {
                value = value.substr(3);
            }
            label.push_back(std::stoi(value));
        }

        data.push_back(row);
        labels.push_back(label);
    }

    return {data, labels};
}

int main() {
    std::vector<std::vector<int>> dataset = {
            {1, 1},
            {1, 0},
            {0, 1},
            {0, 0}
    };

    std::vector<std::vector<int>> target = {
            {1,  1},
            {1,  -1},
            {-1, 1},
            {-1, -1}
    };

    SingleLayerPerceptron slp(2, 2, 1.0, 0.2);
    slp.print_weights();
    slp.train(dataset, target);
    slp.print_weights();

    auto [data, labels] = readData("caracteres-limpo.csv", 63);

    SingleLayerPerceptron slp_letras(63, static_cast<int>(labels[0].size()), 1, 0.2);
    slp_letras.train(data, labels);
    slp_letras.print_weights();

    auto [test_data, test_labels] = readData("caracteres-ruido.csv", 63);

    for (int i = 0; i < test_data.size(); ++i) {
        auto output = slp_letras.predict(test_data[i]);
        std::cout << "Predicao: ";
        std::copy(output.begin(), output.end(), std::ostream_iterator<int>(std::cout, ", "));
        std::cout << "\tEsperado: ";
        std::copy(test_labels[i].begin(), test_labels[i].end(), std::ostream_iterator<int>(std::cout, ", "));
        std::cout << std::endl;
    }

    return 0;
}
