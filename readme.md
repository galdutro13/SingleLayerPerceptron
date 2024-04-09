# Single Layer Perceptron
Este projeto é uma implementação de um Perceptron de Camada Única (Single Layer Perceptron) em C++. O Perceptron é um dos modelos mais simples de rede neural, utilizado para classificação binária.
## Como funciona
O código é composto por uma classe chamada SingleLayerPerceptron que possui métodos para treinar o modelo e fazer previsões. A classe tem os seguintes atributos:
- ```dimension```: a dimensão dos dados de entrada.
- ```num_classes```: o número de classes para classificação.
- ```weights```: os pesos do modelo.
- ```bias_weight```: o peso do bias do modelo.
- ```learning_rate```: a taxa de aprendizado do modelo.
- ```theta```: o limiar da função de ativação.

A classe ```SingleLayerPerceptron``` possui os seguintes métodos:
- ```act_func```: calcula a saída da função de ativação para um determinado ponto de dados, pesos e bias.
- ```ch_weights```: atualiza os pesos e o bias do modelo com base na distância entre a saída prevista e a saída real.
- ```internal_train```: treina o modelo perceptron.
- ```train```: treina o modelo até que os pesos não sejam mais alterados.
- ```predict```: faz uma previsão para um dado ponto de dados.
- ```print_weights```: imprime os pesos e o bias do modelo.

O código também inclui uma função ```readData``` para ler os dados de um arquivo CSV.
