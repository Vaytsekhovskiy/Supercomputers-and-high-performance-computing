// 2) количество потоков заранее неизвестно и не является параметром задачи.
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
    Задача: для каждой позиции (i, j) матрицы A найти алгебраическое дополнение:
        C[i][j] = (-1)^(i+j) * det(M[i][j])
    где M[i][j] — минор, полученный после удаления i-й строки и j-го столбца.

    В этом варианте количество потоков не задаётся пользователем.
    Оно определяется автоматически средой OpenMP:
        - либо значением переменной окружения OMP_NUM_THREADS;
        - либо настройками OpenMP по умолчанию;
        - либо системным максимумом для данной машины.

    То есть число потоков является внутренним параметром запуска, а не параметром задачи.
*/

static double determinant(int n, const double *a)
{
    if (n == 1)
        return a[0];

    if (n == 2)
        return a[0] * a[3] - a[1] * a[2];

    double result = 0.0;

    /*
        Разложение определителя по первой строке:
            det(A) = sum_{j=0}^{n-1} (-1)^(0+j) * a[0][j] * det(M[0][j])
        Здесь a[0][j] — элемент первой строки, а M[0][j] — минор без первой строки и j-го столбца.
        Для каждого j строим такой минор и рекурсивно вычисляем его определитель.
    */
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

        - число потоков не передаётся как параметр функции;
        - оно выбирается OpenMP автоматически;
        - при этом может оказаться, что n не кратно числу потоков.

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
    int threads_count;

    /*
        Размерность матрицы вводится как обычный входной параметр задачи.
        Число потоков не входной параметр: OpenMP сам определит его.
    */
    printf("Введите размерность матрицы n: ");
    if (scanf("%d", &n) != 1 || n <= 0) {
        fprintf(stderr, "Размерность должна быть положительным целым числом.\n");
        return EXIT_FAILURE;
    }

    /*
        Явно фиксированный режим работы OpenMP.
        Чтобы не допустить динамического изменения числа потоков при выполнении,
        выключаем динамическое изменение числа потоков.
    */
    omp_set_dynamic(0);

    /*
        Получаем количество потоков, которое OpenMP использует по умолчанию.
        Это нужно только для вывода информации, а не как входной параметр вычисления.
    */
    threads_count = omp_get_max_threads();
    printf("Количество потоков, выбранное OpenMP: %d\n", threads_count);

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

    /*
        Вызов вычисления алгебраических дополнений в параллельном режиме.
        Количество потоков определяется системой OpenMP, а не вводом пользователя.
    */
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
