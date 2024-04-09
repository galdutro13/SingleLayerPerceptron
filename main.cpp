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
      * Esta função calcula a saída da função de ativação para um determinado ponto de dados, pesos e bias.
      * Ela calcula o produto escalar entre os dados e os pesos, adiciona o bias e, em seguida, aplica a função de ativação.
      * A função de ativação é uma função de passo (step function) que retorna 1 se a entrada líquida for maior que um limite theta,
      * -1 se a entrada líquida for menor que theta - 1, e 0 nos demais casos.
      *
      * @param data Um vetor de inteiros representando um ponto de dados do conjunto de dados.
      * @param weight Um vetor de números decimais representando os pesos do modelo.
      * @param bias Um número decimal representando o bias do modelo.
      * @return Um inteiro representando a saída da função de ativação.
      */
    [[nodiscard]] int act_func(const std::vector<int> &data, const std::vector<double> &weight, double bias) const {
        // Calcula o produto escalar entre os dados e os pesos, e adiciona o bias
        double net = std::inner_product(data.begin(), data.end(), weight.begin(), bias);

        // Aplica a função de ativação e retorna o resultado
        return (net > theta) ? 1 : ((net >= theta - 1) ? 0 : -1);
    }

    /**
      * Esta função é responsável por atualizar os pesos e o bias do modelo com base na distância entre a saída prevista e a saída real.
      * Ela verifica se a saída prevista não é igual à saída real e se a taxa de aprendizado e a saída real não são zero.
      * Caso estas condições sejam atendidas, atualiza os pesos adicionando o produto da taxa de aprendizado, a saída real e o valor dos dados ao peso atual.
      * Também atualiza o bias adicionando o produto da taxa de aprendizado e a saída real ao bias atual.
      *
      * @param data Um vetor de inteiros representando um ponto de dado do conjunto de dados.
      * @param target Um inteiro representando a saída real para o ponto de dado.
      * @param output Um inteiro representando a saída prevista para o ponto de dado.
      * @param weight Um vetor de números decimais representando os pesos do modelo.
      * @param bias Um número decimal representando o bias do modelo.
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
      * Essa função é responsável por treinar o modelo perceptron.
      * Ela itera através do conjunto de dados e atualiza os pesos e o bias do modelo baseando-se na distância entre as saídas previstas e reais.
      * A função retorna um valor booleano indicando se os pesos do modelo foram alterados durante o processo de treinamento.
      *
      * @param dataset Um vetor 2D de números inteiros representando os dados de treinamento.
      * @param target Um vetor 2D de números inteiros representando a saída desejada para cada ponto de dados no dataset.
      * @return Um valor booleano indicando se os pesos do modelo foram alterados durante o processo de treinamento.
      */
    bool internal_train(const std::vector<std::vector<int>> &dataset, const std::vector<std::vector<int>> &target) {
        // Inicializa uma variável booleana para acompanhar se os pesos foram alterados durante o processo de treinamento.
        bool weights_changed = false;

        // Cria um iterador para o vetor target (alvo)
        auto target_iter = target.begin();

        // Itera sobre o conjunto de dados
        for (const auto &data: dataset) {
            // Para cada ponto de dados, itera sobre o número de classes
            for (int i = 0; i < num_classes; ++i) {
                // Calcula a saída da função de ativação para o ponto de dados atual e os pesos
                int output = act_func(data, weights[i], bias_weight[i]);

                // Atualiza os pesos e o bias com base na diferença entre a saída prevista e a saída real
                ch_weights(data, (*target_iter)[i], output, weights[i], bias_weight[i]);

                // Se a saída prevista não corresponder à saída real, define 'weights_changed' como verdadeiro
                if (output != (*target_iter)[i]) {
                    weights_changed = true;
                }
            }

            // Passa para a próxima saída alvo
            ++target_iter;
        }

        // Retorna se os pesos foram alterados durante o processo de treinamento
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
