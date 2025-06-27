#ifndef NEURALNET_H
#define NEURALNET_H
#include "matrix.h"

typedef struct {
    int input_size, hidden_size, output_size;
    Matrix *weights_ih, *weights_ho;
    Matrix *bias_h, *bias_o;
    float learning_rate;
} NeuralNet;

NeuralNet* nn_create(int input, int hidden, int output, float lr);
void nn_free(NeuralNet* nn);
Matrix* nn_predict(NeuralNet* nn, Matrix* input);
void nn_train(NeuralNet* nn, Matrix* input, Matrix* target);

#endif // NEURALNET_H
