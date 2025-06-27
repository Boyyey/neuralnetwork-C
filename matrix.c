#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Matrix* matrix_create(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (float*)calloc(rows * cols, sizeof(float));
    return m;
}

void matrix_free(Matrix* m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

void matrix_fill(Matrix* m, float value) {
    for (int i = 0; i < m->rows * m->cols; i++)
        m->data[i] = value;
}

void matrix_randomize(Matrix* m, float min, float max) {
    for (int i = 0; i < m->rows * m->cols; i++)
        m->data[i] = min + ((float)rand() / RAND_MAX) * (max - min);
}

Matrix* matrix_dot(Matrix* a, Matrix* b) {
    if (a->cols != b->rows) return NULL;
    Matrix* r = matrix_create(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0;
            for (int k = 0; k < a->cols; k++)
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            r->data[i * r->cols + j] = sum;
        }
    }
    return r;
}

Matrix* matrix_add(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) return NULL;
    Matrix* r = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++)
        r->data[i] = a->data[i] + b->data[i];
    return r;
}

Matrix* matrix_sub(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) return NULL;
    Matrix* r = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++)
        r->data[i] = a->data[i] - b->data[i];
    return r;
}

Matrix* matrix_transpose(Matrix* m) {
    Matrix* r = matrix_create(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++)
            r->data[j * r->cols + i] = m->data[i * m->cols + j];
    return r;
}

void matrix_apply(Matrix* m, float (*func)(float)) {
    for (int i = 0; i < m->rows * m->cols; i++)
        m->data[i] = func(m->data[i]);
}

void matrix_print(Matrix* m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++)
            printf("%6.3f ", m->data[i * m->cols + j]);
        printf("\n");
    }
}
