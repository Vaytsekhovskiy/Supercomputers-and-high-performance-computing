#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>

/*
    Программа:

    - берёт квадратную матрицу A размером n x n;
    - для каждого элемента A[i][j] строит минор M[i][j] (матрицу без i-й строки и j-го столбца);
    - вычисляет детерминант этого минора;
    - добавляет знак (-1)^(i+j);
    - результат записывает в матрицу алгебраических дополнений.

    Формула:
        A_ij = (-1)^(i+j) * det(M_ij)

    Почему так:
    алгебраическое дополнение элемента матрицы - это определитель минора,
    умноженный на знак, который зависит от положения элемента.
*/

static double determinant(int n, const double *a)
{
    if (n == 1)
        return a[0];
    if (n == 2)
        return a[0] * a[3] - a[1] * a[2];

    double result = 0.0;
    int i;

    /*
        Разложение определителя по первой строке:
        det(A) = sum_{j=0}^{n-1} a[0][j] * (-1)^(0+j) * det(M_{0j})
        Здесь каждый элемент первой строки умножается на определитель подходящего минора.
        Именно так в математике обычно вычисляют детерминант матрицы порядка n > 2.
    */
    for (i = 0; i < n; ++i) {
        int minor_size = n - 1;
        double *minor = (double *)malloc((size_t)minor_size * minor_size * sizeof(double));
        if (minor == NULL) {
            fprintf(stderr, "Ошибка выделения памяти для минора\n");
            exit(EXIT_FAILURE);
        }

        int row = 0;
        int col = 0;

        /*
            Формируем минор M[0][i]:
            - пропускаем первую строку, потому что она уже использована в разложении;
            - пропускаем i-й столбец, чтобы удалить этот элемент;
            - остальные элементы переписываем в minor[] по порядку.
        */
        for (int r = 1; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                if (c == i)
                    continue;
                minor[row * minor_size + col++] = a[r * n + c];
                if (col == minor_size) {
                    col = 0;
                    ++row;
                }
            }
        }

        double sign = ((i % 2) == 0) ? 1.0 : -1.0;
        result += sign * a[i] * determinant(minor_size, minor);
        free(minor);
    }

    return result;
}

static void compute_cofactors(const double *a, int n, double *cofactor)
{
    if (n == 1) {
        cofactor[0] = 1.0;
        return;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            /*
                Для элемента A[i][j] строим минор M[i][j]:
                удаляем i-ю строку и j-й столбец,
                затем вычисляем его определитель.
                После этого добавляем знак (-1)^(i+j).
            */
            int minor_size = n - 1;
            double *minor = (double *)malloc((size_t)minor_size * minor_size * sizeof(double));
            if (minor == NULL) {
                fprintf(stderr, "Ошибка выделения памяти для минора\n");
                exit(EXIT_FAILURE);
            }

            int idx = 0;
            for (int r = 0; r < n; ++r) {
                if (r == i)
                    continue;
                for (int c = 0; c < n; ++c) {
                    if (c == j)
                        continue;
                    minor[idx++] = a[r * n + c];
                }
            }

            double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            cofactor[i * n + j] = sign * determinant(minor_size, minor);
            free(minor);
        }
    }
}

int main(void)
{
    printf("Введите размерность матрицы n: ");
    int n;
    if (scanf("%d", &n) != 1 || n <= 0) {
        fprintf(stderr, "Размерность должна быть положительным целым числом.\n");
        return EXIT_FAILURE;
    }

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
        fprintf(stderr, "Ошибка выделения памяти для массива алгебраических дополнений.\n");
        free(a);
        return EXIT_FAILURE;
    }

    compute_cofactors(a, n, cofactor);

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
