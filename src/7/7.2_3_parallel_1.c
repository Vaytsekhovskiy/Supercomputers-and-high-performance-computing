// 1) количество потоков является входным
//параметром, при этом размерность матриц может быть не кратна количеству потоков
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
    Вычисление определителя матрицы методом разложения по первой строке.
    Используется рекурсивно для миноров размерности (n-1) x (n-1).
*/
static double determinant(int n, const double *a)
{
    if (n == 1)
        return a[0];

    if (n == 2)
        return a[0] * a[3] - a[1] * a[2];

    double result = 0.0;

    for (int j = 0; j < n; ++j) {
        int minor_size = n - 1;
        double *minor = (double *)malloc((size_t)minor_size * minor_size * sizeof(double));
        if (minor == NULL) {
            fprintf(stderr, "Ошибка выделения памяти для минора\n");
            exit(EXIT_FAILURE);
        }

        int row = 0;
        int col = 0;

        for (int r = 1; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                if (c == j)
                    continue;
                minor[row * minor_size + col++] = a[r * n + c];
                if (col == minor_size) {
                    col = 0;
                    ++row;
                }
            }
        }

        double sign = ((j % 2) == 0) ? 1.0 : -1.0;
        result += sign * a[j] * determinant(minor_size, minor);
        free(minor);
    }

    return result;
}

/*
    Вычисление алгебраических дополнений для всех элементов матрицы.
    Каждый элемент (i, j) независим от остальных, поэтому их можно распределить по потокам.

    Число задач равно n * n. Если n не кратно числу потоков, OpenMP нормально обработает
    остаток: schedule(static) распределит итерации по потокам, и часть потоков получит
    на одну итерацию больше, чем остальные.
*/
static void compute_cofactors_parallel(const double *a, int n, double *cofactor)
{
    if (n == 1) {
        cofactor[0] = 1.0;
        return;
    }

    int idx;
    // Каждый элемент A[i][j] обрабатывается независимо
    // распараллеливание сделано по индексам элементов матрицы
    /*
        Почему это корректно:

        каждый (i, j) вычисляется отдельно, без общих данных;
        записи в разные элементы массива cofactor не конфликтуют;
        schedule(static) автоматически распределяет итерации между потоками, даже если n не кратно числу потоков;
        это и есть правильный способ для случая “размерность матриц может быть не кратна количеству потоков”.
     */
#pragma omp parallel for default(none) shared(a, n, cofactor) schedule(static)
    for (idx = 0; idx < n * n; ++idx) {
        int i = idx / n;
        int j = idx % n;

        int minor_size = n - 1;
        double *minor = (double *)malloc((size_t)minor_size * minor_size * sizeof(double));
        if (minor == NULL) {
            fprintf(stderr, "Ошибка выделения памяти для минора\n");
            exit(EXIT_FAILURE);
        }

        int k = 0;
        for (int r = 0; r < n; ++r) {
            if (r == i)
                continue;
            for (int c = 0; c < n; ++c) {
                if (c == j)
                    continue;
                minor[k++] = a[r * n + c];
            }
        }

        double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        cofactor[i * n + j] = sign * determinant(minor_size, minor);
        free(minor);
    }
}

int main(void)
{
    int n;
    int threads;

    printf("Введите размерность матрицы n: ");
    if (scanf("%d", &n) != 1 || n <= 0) {
        fprintf(stderr, "Размерность должна быть положительным целым числом.\n");
        return EXIT_FAILURE;
    }

    printf("Введите количество потоков: ");
    if (scanf("%d", &threads) != 1 || threads <= 0) {
        fprintf(stderr, "Количество потоков должно быть положительным целым числом.\n");
        return EXIT_FAILURE;
    }

    omp_set_num_threads(threads);

    double *a = (double *)malloc((size_t)n * n * sizeof(double));
    if (a == NULL) {
        fprintf(stderr, "Ошибка выделения памяти для матрицы.\n");
        return EXIT_FAILURE;
    }

    printf("Введите матрицу %d x %d:\n", n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (scanf("%lf", &a[i * n + j]) != 1) {
                fprintf(stderr, "Ошибка чтения элемента матрицы.\n");
                free(a);
                return EXIT_FAILURE;
            }
        }
    }

    double *cofactor = (double *)malloc((size_t)n * n * sizeof(double));
    if (cofactor == NULL) {
        fprintf(stderr, "Ошибка выделения памяти для матрицы дополнений.\n");
        free(a);
        return EXIT_FAILURE;
    }

    compute_cofactors_parallel(a, n, cofactor);

    printf("\nМатрица алгебраических дополнений:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%8.4f ", cofactor[i * n + j]);
        }
        printf("\n");
    }

    free(a);
    free(cofactor);
    return EXIT_SUCCESS;
}
/*
    Составить схему потоков.

  Поток 0 (main)
  │
  ├─ Ввод n и threads
  ├─ omp_set_num_threads(threads)
  ├─ Выделение памяти под матрицу A[n*n] и cofactor[n*n]
  │
  ├─ Чтение матрицы A[i][j]
  │
  ├─ #pragma omp parallel for schedule(static)
  │   │
  │   ├─ Поток T0: получает свой диапазон индексoв idx = [0 .. k1]
  │   │   ├─ i = idx / n
  │   │   ├─ j = idx % n
  │   │   ├─ строит минор M[i][j] (без i-й строки и j-го столбца)
  │   │   ├─ вычисляет det(M[i][j]) рекурсивно
  │   │   ├─ вычисляет sign = (-1)^(i+j)
  │   │   └─ cofactor[i*n + j] = sign * det(M[i][j])
  │   │
  │   ├─ Поток T1: получает свой диапазон индексoв idx = [k1+1 .. k2]
  │   │   ├─ i = idx / n
  │   │   ├─ j = idx % n
  │   │   ├─ строит минор M[i][j]
  │   │   ├─ вычисляет det(M[i][j])
  │   │   ├─ вычисляет sign
  │   │   └─ cofactor[i*n + j] = sign * det(M[i][j])
  │   │
  │   ├─ ...
  │   │
  │   └─ Поток Tk: получает свой диапазон индексoв idx = [last .. n*n-1]
  │       ├─ i = idx / n
  │       ├─ j = idx % n
  │       ├─ строит минор M[i][j]
  │       ├─ вычисляет det(M[i][j])
  │       ├─ вычисляет sign
  │       └─ cofactor[i*n + j] = sign * det(M[i][j])
  │
  ├─ (неявный барьер OpenMP в конце parallel for)
  │
  ├─ Последовательно:
  │   ├─ вывод матрицы алгебраических дополнений
  │   └─ освобождение памяти
  │
  └─ Конец программы
*/