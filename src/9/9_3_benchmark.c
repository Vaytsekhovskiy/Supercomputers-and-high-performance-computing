/*
    Лабораторная работа по MPI.

    Задание 3.
    Реализовать алгоритм с применением функций двухточечного обмена MPI
    и протестировать его на нескольких примерах.

    Важный момент:
    - В MPI обмен "точка-точка" выполняется функциями MPI_Send и MPI_Recv.
    - В этой программе мы используем типовую схему: один процесс (root)
      принимает данные от всех остальных, а затем отправляет им ответ.
    - Это уже классический двухточечный обмен, потому что сообщения идут
      между конкретными процессами, а не через коллективные операции.

    Дополнительно в программе реализованы:
    - таймеры MPI_Wtime();
    - несколько тестовых наборов данных;
    - расчет ускорения и эффективности для каждого запуска.

    Как пользоваться:
      1) Запустить программу на 1 процессе:
         mpiexec -n 1 .\9_3.exe
      2) Запустить программу на 2, 4, 8 процессах:
         mpiexec -n 2 .\9_3.exe
         mpiexec -n 4 .\9_3.exe
         mpiexec -n 8 .\9_3.exe
      3) Сравнить полученные времена T1 и Tp и вычислить:
         S = T1 / Tp
         E = S / p = T1 / (p * Tp)
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/*
    В этой задаче мы моделируем обмен данными между процессами:
    - rank 0 (root) получает массивы от остальных процессов;
    - затем root суммирует все данные и посылает результат обратно;
    - это простой и понятный пример point-to-point обмена.

    Для каждой тестовой конфигурации считается время выполнения,
    а после этого выдаются формулы ускорения и эффективности.
*/

static void fill_vector(double *v, int n, int rank)
{
    for (int i = 0; i < n; ++i) {
        /*
            Значение зависит от номера процесса и индекса элемента.
            Это делает данные различными для каждого процесса и упрощает проверку.
        */
        v[i] = (double)(rank + 1) * 100.0 + (double)i;
    }
}

static double compute_sum(const double *v, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += v[i];
    }
    return s;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            printf("Для этого benchmark нужна MPI-среда минимум с 2 процессами.\n");
            printf("Например: mpiexec -n 2 .\\9_3.exe\n");
        }
        MPI_Finalize();
        return 0;
    }

    /*
        Несколько наборов данных, на которых будем проверять программу.
        Набор можно легко расширить, добавив новые значения n.
    */
    const int test_sizes[] = { 1000, 100000, 1000000 };
    const int test_count = 3;
    const int repeats = 20;

    if (rank == 0) {
        printf("=== MPI point-to-point benchmark ===\n");
        printf("Processes: %d\n", size);
        printf("Each test runs %d repetitions.\n\n", repeats);
    }

    for (int t = 0; t < test_count; ++t) {
        int n = test_sizes[t];

        /*
            Для каждого размера массива на каждом процессе проводится benchmark.
            Внутри цикла выполняется следующая логика:

            - worker процессы создают массив и отправляют его в root (MPI_Send);
            - root принимает массивы от всех workers (MPI_Recv);
            - root суммирует все полученные данные;
            - root отправляет результат назад каждому worker (MPI_Send).

            Комментарий по схеме:
            root = 0,
            worker = 1..size-1.
        */
        double start = MPI_Wtime();

        for (int r = 0; r < repeats; ++r) {
            if (rank == 0) {
                /*
                    root принимает данные от каждого worker по точке-точке.
                    Это типичный пример обмена MPI_Send/MPI_Recv.
                */
                double *recv_buf = (double *)malloc((size_t)n * sizeof(double));
                double total = 0.0;

                for (int src = 1; src < size; ++src) {
                    MPI_Status status;
                    MPI_Recv(recv_buf, n, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &status);
                    total += compute_sum(recv_buf, n);
                }

                /*
                    root отправляет обратно каждому worker итоговую сумму.
                    Обмен всё ещё точка-точка: конкретный процесс -> конкретный процесс.
                */
                for (int dst = 1; dst < size; ++dst) {
                    MPI_Send(&total, 1, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD);
                }

                free(recv_buf);
            } else {
                double *buf = (double *)malloc((size_t)n * sizeof(double));
                double result = 0.0;

                fill_vector(buf, n, rank);

                /*
                    worker отправляет свой массив root.
                    Используется MPI_Send, то есть явный обмен между двумя процессами.
                */
                MPI_Send(buf, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

                /*
                    worker получает ответ от root.
                    Это второй обмен типа point-to-point.
                */
                MPI_Recv(&result, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                free(buf);
            }
        }

        double elapsed = MPI_Wtime() - start;

        /*
            Для сравнения результатов разных запусков удобно получить максимальное время
            среди всех процессов. На root это будет время всего параллельного фрагмента.
        */
        double max_elapsed = 0.0;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("n = %d, time = %.6f s\n", n, max_elapsed / repeats);
            printf("  Формула для ускорения: S = T1 / Tp\n");
            printf("  Формула для эффективности: E = S / p = T1 / (p * Tp)\n");
            printf("  Здесь p = %d, Tp = %.6f s (время этого запуска)\n\n", size, max_elapsed / repeats);
        }
    }

    if (rank == 0) {
        printf("=== Как считать ускорение и эффективность ===\n");
        printf("1) Запустите программу на 1 процессе: mpiexec -n 1 .\\9_3.exe\n");
        printf("2) Затем на 2, 4, 8 процессах: mpiexec -n 2 .\\9_3.exe, mpiexec -n 4 .\\9_3.exe, ...\n");
        printf("3) Для каждого запуска возьмите T1 и Tp из таблицы выше.\n");
        printf("4) Скорость (speedup): S = T1 / Tp\n");
        printf("5) Эффективность: E = S / p = T1 / (p * Tp)\n");
        printf("6) Чем ближе E к 1, тем лучше масштабируемость программы.\n");
    }

    MPI_Finalize();
    return 0;
}

/*
    Пример запуска:

      mpiexec -n 1 .\9_3.exe
      mpiexec -n 2 .\9_3.exe
      mpiexec -n 4 .\9_3.exe

    В каждом запуске программа печатает время для нескольких размеров n:
      n = 1000, time = ...
      n = 100000, time = ...
      n = 1000000, time = ...

    Затем нужно сравнить T1 и Tp:
      S = T1 / Tp
      E = T1 / (p * Tp)

    Это и есть требуемое ускорение и эффективность для MPI-программы.
*/
