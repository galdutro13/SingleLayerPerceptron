#include <iostream>
#include <vector>
#include <numeric>

class SingleLayerPerceptron {
private:
    const int dimention;
    const int numb_classes;

    std::vector<std::vector<double>> weights;
    std::vector<double> bweight;
    double learning_rate;

    const double theta;
    const double bias = 1.0;

    // Função responsavel por atualizar os pesos da rede neural
    bool ch_weights(const std::vector<int>& OUT, const std::vector<int>& target, const std::vector<int>& data){
        bool wchanged = false; // Variavel que indica se os pesos foram alterados
        for(int i = 0; i < numb_classes; ++i) { // Loop para cada classe
            if(OUT[i] != target[i] && learning_rate != 0 && target[i] != 0) { // Verifica se a saida da rede neural é diferente do target
                wchanged = true; // Indica que os pesos foram alterados
                for(int j = 0; j < dimention; ++j) { // Loop para cada dimensão
                    weights[i][j] += learning_rate * target[i] * data[j]; // Atualiza os pesos
                }
                bweight[i] += learning_rate * target[i]; // Atualiza os pesos do bias
            }
        }
        return wchanged; // Retorna se os pesos foram alterados
    }

    // Função de ativação
    [[nodiscard]] inline int act_func(double y_in) const {
        return (y_in > theta) ? 1 : ((y_in >= theta - 1) ? 0 : -1);
    }

    // Essa função implementa o loop interno do algoritmo de treinamento
    bool i_train(const std::vector<std::vector<int>>& dataset, const std::vector<std::vector<int>>& target) {
        bool wchanged = false; // Variavel que indica se os pesos foram alterados
        int j = 0;

        // Loop do algoritmo de treinamento
        for(const auto& data : dataset) {
            std::vector<int> OUT(numb_classes); // Vetor de saida da rede neural

            // Calculo da saida da rede neural
            for(int i = 0; i < numb_classes; ++i) {
                // Calculo do dotproduct entre os pesos e os dados de entrada
                double NET = std::inner_product(data.begin(), data.end(), weights[i].begin(), bweight[i] * bias);
                OUT[i] = act_func(NET); // Função de ativação
            }

            // Atualização dos pesos
            wchanged |= ch_weights(OUT, target[j], data);
            ++j;
        }
        return wchanged;
    }

public:
    // Construtor
    SingleLayerPerceptron(int dimention, int numb_classes, double learning_rate, double theta)
            : dimention(dimention), numb_classes(numb_classes), learning_rate(learning_rate), theta(theta),
              weights(numb_classes, std::vector<double>(dimention, 0.0)),
              bweight(numb_classes, 0.0) {}

    void train(const std::vector<std::vector<int>>& dataset, const std::vector<std::vector<int>>& target) {
        while (i_train(dataset, target));
    }

    std::vector<int> predict(const std::vector<int>& data) {
        std::vector<int> OUT;
        OUT.reserve(numb_classes);
        for(int i = 0; i < numb_classes; ++i) {
            double NET = std::inner_product(data.begin(), data.end(), weights[i].begin(), bweight[i] * bias);
            OUT.emplace_back(act_func(NET));
        }
        return OUT;
    }

    void print_weights() const {
        for(int i = 0; i < numb_classes; ++i) {
            std::cout << "Neuron " << i + 1 << ":\n";
            std::cout << "Weights: ";
            for(auto w : weights[i]) {
                std::cout << w << ", ";
            }
            std::cout << "\nBias weight: " << bweight[i] << "\n";
        }
    }
};

int main() {
    std::vector<std::vector<int>> dataset = {
            {1, 1},
            {1, 0},
            {0, 1},
            {0, 0}
    };

    std::vector<std::vector<int>> target = {
            {1, 1},
            {1, -1},
            {-1, 1},
            {-1, -1}
    };

    SingleLayerPerceptron slp(2, 2, 1.0, 0.2);
    slp.print_weights();
    slp.train(dataset, target);
    slp.print_weights();

    return 0;
}
