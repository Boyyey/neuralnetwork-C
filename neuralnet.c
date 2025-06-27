#include "neuralnet.h"
#include <stdlib.h>
#include <math.h>

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float dsigmoid(float y) { return y * (1.0f - y); }

NeuralNet* nn_create(int input, int hidden, int output, float lr) {
    NeuralNet* nn = (NeuralNet*)malloc(sizeof(NeuralNet));
    nn->input_size = input;
    nn->hidden_size = hidden;
    nn->output_size = output;
    nn->learning_rate = lr;
    nn->weights_ih = matrix_create(hidden, input);
    nn->weights_ho = matrix_create(output, hidden);
    nn->bias_h = matrix_create(hidden, 1);
    nn->bias_o = matrix_create(output, 1);
    matrix_randomize(nn->weights_ih, -1, 1);
    matrix_randomize(nn->weights_ho, -1, 1);
    matrix_randomize(nn->bias_h, -1, 1);
    matrix_randomize(nn->bias_o, -1, 1);
    return nn;
}

void nn_free(NeuralNet* nn) {
    if (nn) {
        matrix_free(nn->weights_ih);
        matrix_free(nn->weights_ho);
        matrix_free(nn->bias_h);
        matrix_free(nn->bias_o);
        free(nn);
    }
}

Matrix* nn_predict(NeuralNet* nn, Matrix* input) {
    Matrix* hidden = matrix_dot(nn->weights_ih, input);
    Matrix* hidden_b = matrix_add(hidden, nn->bias_h);
    matrix_apply(hidden_b, sigmoid);
    Matrix* output = matrix_dot(nn->weights_ho, hidden_b);
    Matrix* output_b = matrix_add(output, nn->bias_o);
    matrix_apply(output_b, sigmoid);
    matrix_free(hidden);
    matrix_free(hidden_b);
    matrix_free(output);
    return output_b;
}

void nn_train(NeuralNet* nn, Matrix* input, Matrix* target) {
    // Forward
    Matrix* hidden = matrix_dot(nn->weights_ih, input);
    Matrix* hidden_b = matrix_add(hidden, nn->bias_h);
    matrix_apply(hidden_b, sigmoid);
    Matrix* output = matrix_dot(nn->weights_ho, hidden_b);
    Matrix* output_b = matrix_add(output, nn->bias_o);
    matrix_apply(output_b, sigmoid);
    // Error
    Matrix* output_error = matrix_sub(target, output_b);
    // dOutput = error * dsigmoid(output)
    Matrix* d_output = matrix_create(output_b->rows, output_b->cols);
    for (int i = 0; i < output_b->rows * output_b->cols; i++)
        d_output->data[i] = output_error->data[i] * dsigmoid(output_b->data[i]);
    // Update weights_ho and bias_o
    Matrix* hidden_b_T = matrix_transpose(hidden_b);
    Matrix* delta_ho = matrix_dot(d_output, hidden_b_T);
    for (int i = 0; i < nn->weights_ho->rows * nn->weights_ho->cols; i++)
        nn->weights_ho->data[i] += nn->learning_rate * delta_ho->data[i];
    for (int i = 0; i < nn->bias_o->rows * nn->bias_o->cols; i++)
        nn->bias_o->data[i] += nn->learning_rate * d_output->data[i];
    // Hidden error
    Matrix* weights_ho_T = matrix_transpose(nn->weights_ho);
    Matrix* hidden_error = matrix_dot(weights_ho_T, d_output);
    // dHidden = hidden_error * dsigmoid(hidden)
    Matrix* d_hidden = matrix_create(hidden_b->rows, hidden_b->cols);
    for (int i = 0; i < hidden_b->rows * hidden_b->cols; i++)
        d_hidden->data[i] = hidden_error->data[i] * dsigmoid(hidden_b->data[i]);
    // Update weights_ih and bias_h
    Matrix* input_T = matrix_transpose(input);
    Matrix* delta_ih = matrix_dot(d_hidden, input_T);
    for (int i = 0; i < nn->weights_ih->rows * nn->weights_ih->cols; i++)
        nn->weights_ih->data[i] += nn->learning_rate * delta_ih->data[i];
    for (int i = 0; i < nn->bias_h->rows * nn->bias_h->cols; i++)
        nn->bias_h->data[i] += nn->learning_rate * d_hidden->data[i];
    // Free
    matrix_free(hidden);
    matrix_free(hidden_b);
    matrix_free(output);
    matrix_free(output_b);
    matrix_free(output_error);
    matrix_free(d_output);
    matrix_free(hidden_b_T);
    matrix_free(delta_ho);
    matrix_free(weights_ho_T);
    matrix_free(hidden_error);
    matrix_free(d_hidden);
    matrix_free(input_T);
    matrix_free(delta_ih);
}
