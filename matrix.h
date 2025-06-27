#ifndef MATRIX_H
#define MATRIX_H

// Matrix structure and operations for neural network

typedef struct {
    int rows, cols;
    float* data;
} Matrix;

Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix* m);
void matrix_fill(Matrix* m, float value);
void matrix_randomize(Matrix* m, float min, float max);
Matrix* matrix_dot(Matrix* a, Matrix* b);
Matrix* matrix_add(Matrix* a, Matrix* b);
Matrix* matrix_sub(Matrix* a, Matrix* b);
Matrix* matrix_transpose(Matrix* m);
void matrix_apply(Matrix* m, float (*func)(float));
void matrix_print(Matrix* m);

#endif // MATRIX_H
