/*
    ЛР 12. Умножение матрицы на вектор (y = A * x).

    ПРОГРАММА ПОДГОТОВКИ ДАННЫХ (последовательная, без MPI).

    Что делает:
    1) Генерирует случайную матрицу A размера m x n и вектор x размера n.
    2) Делит матрицу A на p блоков строк (block-распределение) --
       так, как её потом будут независимо читать p MPI-процессов.
    3) Записывает данные на диск в текстовые файлы:
         vector_m{m}_n{n}_p{p}.txt        - вектор x (общий для всех процессов)
         part_m{m}_n{n}_p{p}_r{rank}.txt  - своя часть матрицы A для процесса rank
         expected_m{m}_n{n}_p{p}.txt      - эталонный результат y, посчитанный
                                              последовательно (для проверки корректности)

    Как правильно поделить m строк на p процессов почти поровну:
        base = m / p          - целая часть
        rem  = m % p          - остаток
        Первые `rem` процессов получают (base + 1) строк,
        остальные -- ровно `base` строк.
        Это гарантирует, что разница в нагрузке между процессами не превышает 1 строку.

    Запуск:
        prepare_data.exe <m> <n> <p> [seed]

    Пример:
        prepare_data.exe 1000 1000 4
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Использование: %s m n p [seed]\n", argv[0]);
        fprintf(stderr, "  m    - число строк матрицы A\n");
        fprintf(stderr, "  n    - число столбцов матрицы A (и размер вектора x)\n");
        fprintf(stderr, "  p    - число процессов, для которых готовятся данные\n");
        fprintf(stderr, "  seed - (необязательно) зерно генератора случайных чисел\n");
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int p = atoi(argv[3]);
    unsigned int seed = (argc >= 5) ? (unsigned int)atoi(argv[4]) : 42u;

    if (m <= 0 || n <= 0 || p <= 0) {
        fprintf(stderr, "Ошибка: m, n, p должны быть положительными целыми числами.\n");
        return 1;
    }
    if (p > m) {
        fprintf(stderr, "Ошибка: число процессов p (%d) не должно превышать число строк m (%d).\n", p, m);
        return 1;
    }

    srand(seed);

    /* 1) Генерация вектора x и его запись в файл */

    double *x = (double *)malloc((size_t)n * sizeof(double));
    if (!x) { fprintf(stderr, "Ошибка выделения памяти под вектор x.\n"); return 1; }

    for (int j = 0; j < n; ++j) {
        /* случайное значение в диапазоне [-10.0, 10.0] */
        x[j] = -10.0 + 20.0 * ((double)rand() / (double)RAND_MAX);
    }

    char vec_filename[256];
    snprintf(vec_filename, sizeof(vec_filename), "vector_m%d_n%d_p%d.txt", m, n, p);
    FILE *fv = fopen(vec_filename, "w");
    if (!fv) { fprintf(stderr, "Не удалось создать файл %s\n", vec_filename); return 1; }

    fprintf(fv, "%d\n", n);
    for (int j = 0; j < n; ++j) {
        fprintf(fv, "%.10f\n", x[j]);
    }
    fclose(fv);

    /* 2) Вычисляем распределение строк по процессам*/

    int base = m / p;
    int rem  = m % p;

    /* rows_count[r]  - сколько строк получит процесс r
       start_row[r]   - с какой строки матрицы A начинается его блок */
    int *rows_count = (int *)malloc((size_t)p * sizeof(int));
    int *start_row  = (int *)malloc((size_t)p * sizeof(int));

    int offset = 0;
    for (int r = 0; r < p; ++r) {
        rows_count[r] = base + (r < rem ? 1 : 0);
        start_row[r] = offset;
        offset += rows_count[r];
    }
    /* offset теперь должен быть равен m -- проверка на всякий случай */
    if (offset != m) {
        fprintf(stderr, "Внутренняя ошибка распределения строк.\n");
        return 1;
    }

    /* 3) Генерация матрицы построчно, сразу раскладывая по файлам */

    /* Для проверки корректности параллельно считаем эталонный результат y = A*x
       обычным последовательным способом. */
    double *y_expected = (double *)malloc((size_t)m * sizeof(double));
    if (!y_expected) { fprintf(stderr, "Ошибка выделения памяти под y_expected.\n"); return 1; }

    for (int r = 0; r < p; ++r) {
        char part_filename[256];
        snprintf(part_filename, sizeof(part_filename), "part_m%d_n%d_p%d_r%d.txt", m, n, p, r);
        FILE *fp = fopen(part_filename, "w");
        if (!fp) { fprintf(stderr, "Не удалось создать файл %s\n", part_filename); return 1; }

        /* Первая строка файла -- заголовок: сколько строк и сколько столбцов внутри */
        fprintf(fp, "%d %d\n", rows_count[r], n);

        for (int i = 0; i < rows_count[r]; ++i) {
            int global_row = start_row[r] + i;

            double dot = 0.0;
            for (int j = 0; j < n; ++j) {
                double a_ij = -5.0 + 10.0 * ((double)rand() / (double)RAND_MAX);
                fprintf(fp, "%.10f ", a_ij);
                dot += a_ij * x[j];
            }
            fprintf(fp, "\n");

            y_expected[global_row] = dot;
        }

        fclose(fp);
        printf("Записан файл %s (%d строк)\n", part_filename, rows_count[r]);
    }

    /* 4) Запись эталонного результата для проверки корректности */

    char expected_filename[256];
    snprintf(expected_filename, sizeof(expected_filename), "expected_m%d_n%d_p%d.txt", m, n, p);
    FILE *fe = fopen(expected_filename, "w");
    if (!fe) { fprintf(stderr, "Не удалось создать файл %s\n", expected_filename); return 1; }

    fprintf(fe, "%d\n", m);
    for (int i = 0; i < m; ++i) {
        fprintf(fe, "%.10f\n", y_expected[i]);
    }
    fclose(fe);

    printf("Записан файл %s (эталонный результат для проверки)\n", expected_filename);
    printf("Записан файл %s (вектор x)\n", vec_filename);
    printf("Подготовка данных завершена: m=%d, n=%d, p=%d\n", m, n, p);

    free(x);
    free(y_expected);
    free(rows_count);
    free(start_row);

    return 0;
}