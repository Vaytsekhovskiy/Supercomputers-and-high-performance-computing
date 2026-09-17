/*
    лр 12. Умножение матрицы на вектор (y = A * x).

    ПАРАЛЛЕЛЬНАЯ ПРОГРАММА (MPI).

    Идея алгоритма (параллелизм по данным):
    - Матрица A заранее (программой prepare_data) разбита на p блоков строк
      и разложена по отдельным файлам -- part_..._r0.txt, part_..._r1.txt и т.д.
    - Каждый MPI-процесс с номером rank читает СВОЙ файл part_..._r{rank}.txt
      и общий файл с вектором x.
    - Каждый процесс независимо вычисляет свою часть результата:
      скалярные произведения "своих" строк матрицы на вектор x.
      Это не требует никакого обмена данными между процессами во время счёта --
      каждая строка обрабатывается полностью независимо от остальных.
    - В конце все процессы передают свои куски результата на процесс 0
      с помощью MPI_Gatherv (Gatherv -- потому что число строк у процессов
      может отличаться на 1, если m не делится на p нацело).
    - Процесс 0 сохраняет итоговый вектор y и печатает время выполнения.

    ВАЖНО: формула распределения строк по процессам (base/rem) здесь
    ДОЛЖНА СОВПАДАТЬ с формулой в prepare_data.c, иначе процесс прочитает
    файл с одним числом строк, а Gatherv будет ждать другое количество.
    Совпадение гарантируется тем, что обе программы используют одну и ту же
    математическую формулу деления m на p (см. комментарий ниже).

    Запуск:
        mpiexec -n <p> 12_3.exe <m> <n> <p>

    Пример:
        mpiexec -n 3 12_3.exe 6 4 3
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Использование: mpiexec -n <p> %s <m> <n> <p>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int p = atoi(argv[3]);

    /* p, переданное в аргументах, должно совпадать с реальным числом MPI-процессов --
       иначе файлы, подготовленные для p процессов, не подойдут для другого числа. */
    if (p != size) {
        if (rank == 0) {
            fprintf(stderr, "Ошибка: аргумент p=%d не совпадает с числом MPI-процессов (%d).\n", p, size);
            fprintf(stderr, "Запустите: mpiexec -n %d %s %d %d %d\n", p, argv[0], m, n, p);
        }
        MPI_Finalize();
        return 1;
    }

    /* 1) Каждый процесс читает общий вектор x*/

    char vec_filename[256];
    snprintf(vec_filename, sizeof(vec_filename), "vector_m%d_n%d_p%d.txt", m, n, p);
    FILE *fv = fopen(vec_filename, "r");
    if (!fv) {
        fprintf(stderr, "rank %d: не удалось открыть файл %s (сначала запустите prepare_data)\n",
                rank, vec_filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n_from_file;
    if (fscanf(fv, "%d", &n_from_file) != 1 || n_from_file != n) {
        fprintf(stderr, "rank %d: файл %s повреждён или не совпадает с n=%d\n", rank, vec_filename, n);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    double *x = (double *)malloc((size_t)n * sizeof(double));
    for (int j = 0; j < n; ++j) {
        if (fscanf(fv, "%lf", &x[j]) != 1) {
            fprintf(stderr, "rank %d: ошибка чтения вектора x\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }
    fclose(fv);

    /*2) Каждый процесс читает свою часть матрицы A */

    char part_filename[256];
    snprintf(part_filename, sizeof(part_filename), "part_m%d_n%d_p%d_r%d.txt", m, n, p, rank);
    FILE *fp = fopen(part_filename, "r");
    if (!fp) {
        fprintf(stderr, "rank %d: не удалось открыть файл %s\n", rank, part_filename);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    int rows_count, n_check;
    if (fscanf(fp, "%d %d", &rows_count, &n_check) != 2 || n_check != n) {
        fprintf(stderr, "rank %d: файл %s повреждён\n", rank, part_filename);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }

    double *A_local = (double *)malloc((size_t)rows_count * n * sizeof(double));
    for (int idx = 0; idx < rows_count * n; ++idx) {
        if (fscanf(fp, "%lf", &A_local[idx]) != 1) {
            fprintf(stderr, "rank %d: ошибка чтения матрицы из %s\n", rank, part_filename);
            MPI_Abort(MPI_COMM_WORLD, 6);
        }
    }
    fclose(fp);

    /*  3)  вычисление: локальные скалярные произведения  */

    /* Синхронизируем старт замера времени у всех процессов */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    double *y_local = (double *)malloc((size_t)rows_count * sizeof(double));
    for (int i = 0; i < rows_count; ++i) {
        double sum = 0.0;
        const double *row = &A_local[(size_t)i * n];
        for (int j = 0; j < n; ++j) {
            sum += row[j] * x[j];
        }
        y_local[i] = sum;
    }

    double t_compute_end = MPI_Wtime();

    /* 4) Сбор результата на процессе 0 через MPI_Gatherv  */

    /* Чтобы собрать результат в правильном порядке, процесс 0 должен знать,
       сколько элементов пришлёт каждый процесс (recvcounts) и с какого места
       в итоговом массиве их разместить (displs). Эта информация вычисляется
       по ТОЙ ЖЕ формуле распределения, что и в prepare_data.c -- поэтому
       читать её из файла не нужно, достаточно пересчитать. */

    int *recvcounts = NULL;
    int *displs = NULL;
    double *y = NULL;

    if (rank == 0) {
        recvcounts = (int *)malloc((size_t)size * sizeof(int));
        displs = (int *)malloc((size_t)size * sizeof(int));

        int base = m / size;
        int rem = m % size;
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            recvcounts[r] = base + (r < rem ? 1 : 0);
            displs[r] = offset;
            offset += recvcounts[r];
        }

        y = (double *)malloc((size_t)m * sizeof(double));
    }

    MPI_Gatherv(y_local, rows_count, MPI_DOUBLE,
                y, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();

    /* 5) Замер и сбор времени выполнения */

    double local_compute_time = t_compute_end - t_start; /* время только вычислений */
    double local_total_time = t_end - t_start;            /* время вычислений + сбор результата */

    double max_compute_time, max_total_time;
    MPI_Reduce(&local_compute_time, &max_compute_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /*  6) Процесс 0: сохранение результата, проверка корректности, вывод времени*/

    if (rank == 0) {
        char result_filename[256];
        snprintf(result_filename, sizeof(result_filename), "result_m%d_n%d_p%d.txt", m, n, p);
        FILE *fr = fopen(result_filename, "w");
        fprintf(fr, "%d\n", m);
        for (int i = 0; i < m; ++i) {
            fprintf(fr, "%.10f\n", y[i]);
        }
        fclose(fr);

        /* Проверка корректности: сравниваем с эталоном, посчитанным prepare_data.c */
        char expected_filename[256];
        snprintf(expected_filename, sizeof(expected_filename), "expected_m%d_n%d_p%d.txt", m, n, p);
        FILE *fe = fopen(expected_filename, "r");

        double max_abs_diff = -1.0;
        if (fe != NULL) {
            int m_check;
            fscanf(fe, "%d", &m_check);
            if (m_check == m) {
                max_abs_diff = 0.0;
                for (int i = 0; i < m; ++i) {
                    double expected_val;
                    fscanf(fe, "%lf", &expected_val);
                    double diff = y[i] - expected_val;
                    if (diff < 0) diff = -diff;
                    if (diff > max_abs_diff) max_abs_diff = diff;
                }
            }
            fclose(fe);
        }

        printf("=== Умножение матрицы на вектор (MPI) ===\n");
        printf("m=%d, n=%d, p=%d (процессов: %d)\n", m, n, p, size);
        printf("Время вычислений (max по процессам): %.6f сек\n", max_compute_time);
        printf("Время вычислений + сбор результата : %.6f сек\n", max_total_time);
        if (max_abs_diff >= 0.0) {
            printf("Проверка корректности: максимальное отклонение от эталона = %.10e\n", max_abs_diff);
            printf("  (значение должно быть близко к 0 -- отличия только из-за погрешности double)\n");
        } else {
            printf("Файл с эталонным результатом не найден -- проверка корректности пропущена.\n");
        }
        printf("Результат сохранён в файл: %s\n", result_filename);

        free(y);
        free(recvcounts);
        free(displs);
    }

    free(x);
    free(A_local);
    free(y_local);

    MPI_Finalize();
    return 0;
}
/*
Выводы программы:
1. Маленький пример для проверки правильности вручную

PS C:\projects\supercomps> ./src/12/prepare_data.exe 6 4 3
Записан файл part_m6_n4_p3_r0.txt (2 строк)
Записан файл part_m6_n4_p3_r1.txt (2 строк)
Записан файл part_m6_n4_p3_r2.txt (2 строк)
Записан файл expected_m6_n4_p3.txt (эталонный результат для проверки)
Записан файл vector_m6_n4_p3.txt (вектор x)
Подготовка данных завершена: m=6, n=4, p=3

PS C:\projects\supercomps> mpiexec -n 3 ./cmake-build-debug/12_3.exe 6 4 3
=== Умножение матрицы на вектор (MPI) ===
m=6, n=4, p=3 (процессов: 3)
Время вычислений (max по процессам): 0.000001 сек
Время вычислений + сбор результата : 0.000390 сек
Проверка корректности: максимальное отклонение от эталона = 7.2549255492e-10
  (значение должно быть близко к 0 -- отличия только из-за погрешности double)
Результат сохранён в файл: result_m6_n4_p3.txt

2. Проверка на "неровном" делении (m не делится на p нацело) -
важно проверить именно этот случай, чтобы убедиться, что MPI_Gatherv работает правильно:

PS C:\projects\supercomps> ./src/12/prepare_data.exe 10 5 3
Записан файл part_m10_n5_p3_r0.txt (4 строк)
Записан файл part_m10_n5_p3_r1.txt (3 строк)
Записан файл part_m10_n5_p3_r2.txt (3 строк)
Записан файл expected_m10_n5_p3.txt (эталонный результат для проверки)
Записан файл vector_m10_n5_p3.txt (вектор x)
Подготовка данных завершена: m=10, n=5, p=3
PS C:\projects\supercomps> mpiexec -n 3 ./cmake-build-debug/12_3.exe 10 5 3
=== Умножение матрицы на вектор (MPI) ===
m=10, n=5, p=3 (процессов: 3)
Время вычислений (max по процессам): 0.000001 сек
Время вычислений + сбор результата : 0.000368 сек
Проверка корректности: максимальное отклонение от эталона = 6.6953020905e-10
  (значение должно быть близко к 0 -- отличия только из-за погрешности double)
Результат сохранён в файл: result_m10_n5_p3.txt

3. Проверка на минимальном p (p=2, как указано в задании 2<=p<<m,n):

PS C:\projects\supercomps> ./src/12/prepare_data.exe 100 100 2
Записан файл part_m100_n100_p2_r0.txt (50 строк)
Записан файл part_m100_n100_p2_r1.txt (50 строк)
Записан файл expected_m100_n100_p2.txt (эталонный результат для проверки)
Записан файл vector_m100_n100_p2.txt (вектор x)
Подготовка данных завершена: m=100, n=100, p=2
PS C:\projects\supercomps> mpiexec -n 2 ./cmake-build-debug/12_3.exe 100 100 2
=== Умножение матрицы на вектор (MPI) ===
m=100, n=100, p=2 (процессов: 2)
Время вычислений (max по процессам): 0.000011 сек
Время вычислений + сбор результата : 0.000028 сек
Проверка корректности: максимальное отклонение от эталона = 6.4852656578e-09
  (значение должно быть близко к 0 -- отличия только из-за погрешности double)
Результат сохранён в файл: result_m100_n100_p2.txt

4. Для каждой комбинации размеров данных запустим prepare_data один раз,
затем 12_3 - с разным числом процессов p:

PS C:\projects\supercomps> ./src/12/prepare_data.exe 5000 5000 1
Записан файл part_m5000_n5000_p1_r0.txt (5000 строк)
Записан файл expected_m5000_n5000_p1.txt (эталонный результат для проверки)
Записан файл vector_m5000_n5000_p1.txt (вектор x)
Подготовка данных завершена: m=5000, n=5000, p=1
PS C:\projects\supercomps> mpiexec -n 1 ./cmake-build-debug/12_3.exe 5000 5000 1
=== Умножение матрицы на вектор (MPI) ===
m=5000, n=5000, p=1 (процессов: 1)
Время вычислений (max по процессам): 0.052292 сек
Время вычислений + сбор результата : 0.052332 сек
Проверка корректности: максимальное отклонение от эталона = 4.9497373311e-08
  (значение должно быть близко к 0 -- отличия только из-за погрешности double)
Результат сохранён в файл: result_m5000_n5000_p1.txt

PS C:\projects\supercomps> ./src/12/prepare_data.exe 5000 5000 2
Записан файл part_m5000_n5000_p2_r0.txt (2500 строк)
Записан файл part_m5000_n5000_p2_r1.txt (2500 строк)
Записан файл expected_m5000_n5000_p2.txt (эталонный результат для проверки)
Записан файл vector_m5000_n5000_p2.txt (вектор x)
Подготовка данных завершена: m=5000, n=5000, p=2
PS C:\projects\supercomps> mpiexec -n 2 ./cmake-build-debug/12_3.exe 5000 5000 2
=== Умножение матрицы на вектор (MPI) ===
m=5000, n=5000, p=2 (процессов: 2)
Время вычислений (max по процессам): 0.025631 сек
Время вычислений + сбор результата : 0.025686 сек
Проверка корректности: максимальное отклонение от эталона = 4.9497373311e-08
  (значение должно быть близко к 0 -- отличия только из-за погрешности double)
Результат сохранён в файл: result_m5000_n5000_p2.txt

PS C:\projects\supercomps> ./src/12/prepare_data.exe 5000 5000 4
Записан файл part_m5000_n5000_p4_r0.txt (1250 строк)
Записан файл part_m5000_n5000_p4_r1.txt (1250 строк)
Записан файл part_m5000_n5000_p4_r2.txt (1250 строк)
Записан файл part_m5000_n5000_p4_r3.txt (1250 строк)
Записан файл expected_m5000_n5000_p4.txt (эталонный результат для проверки)
Записан файл vector_m5000_n5000_p4.txt (вектор x)
Подготовка данных завершена: m=5000, n=5000, p=4
PS C:\projects\supercomps> mpiexec -n 4 ./cmake-build-debug/12_3.exe 5000 5000 4
=== Умножение матрицы на вектор (MPI) ===
m=5000, n=5000, p=4 (процессов: 4)
Время вычислений (max по процессам): 0.013282 сек
Время вычислений + сбор результата : 0.013515 сек
Проверка корректности: максимальное отклонение от эталона = 4.9497373311e-08
  (значение должно быть близко к 0 -- отличия только из-за погрешности double)
Результат сохранён в файл: result_m5000_n5000_p4.txt

PS C:\projects\supercomps> ./src/12/prepare_data.exe 5000 5000 8
Записан файл part_m5000_n5000_p8_r0.txt (625 строк)
Записан файл part_m5000_n5000_p8_r1.txt (625 строк)
Записан файл part_m5000_n5000_p8_r2.txt (625 строк)
Записан файл part_m5000_n5000_p8_r3.txt (625 строк)
Записан файл part_m5000_n5000_p8_r4.txt (625 строк)
Записан файл part_m5000_n5000_p8_r5.txt (625 строк)
Записан файл part_m5000_n5000_p8_r6.txt (625 строк)
Записан файл part_m5000_n5000_p8_r7.txt (625 строк)
Записан файл expected_m5000_n5000_p8.txt (эталонный результат для проверки)
Записан файл vector_m5000_n5000_p8.txt (вектор x)
Подготовка данных завершена: m=5000, n=5000, p=8
PS C:\projects\supercomps> mpiexec -n 8 ./cmake-build-debug/12_3.exe 5000 5000 8
=== Умножение матрицы на вектор (MPI) ===
m=5000, n=5000, p=8 (процессов: 8)
Время вычислений (max по процессам): 0.010638 сек
Время вычислений + сбор результата : 0.010775 сек
Проверка корректности: максимальное отклонение от эталона = 4.9497373311e-08
  (значение должно быть близко к 0 -- отличия только из-за погрешности double)
Результат сохранён в файл: result_m5000_n5000_p8.txt
 */