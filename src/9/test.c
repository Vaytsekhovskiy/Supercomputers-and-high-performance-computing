#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/*
    Этот файл служит для измерения времени работы MPI-программы
    при различных размерах входных данных и различном количестве процессов.

    Как использовать:
      1) Последовательный запуск (база для сравнения):
             mpiexec -n 1 .\test.exe 1000000

      2) Параллельный запуск:
             mpiexec -n 2 .\test.exe 1000000
             mpiexec -n 4 .\test.exe 1000000
             mpiexec -n 8 .\test.exe 1000000

      3) После этого считать:
             ускорение S = T1 / Tp
             эффективность E = S / P
         где:
             T1  - время работы при 1 процессе (последовательная версия)
             Tp  - время работы при P процессах
             P   - число процессов

    В этом примере мы не используем MPI_Reduce, а используем лишь point-to-point
    обмены MPI_Send / MPI_Recv, чтобы показать, как измерять время в MPI и
    как сравнивать разные режимы запуска.
*/

static double local_work(const double *data, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        /*
            Простая вычислительная нагрузка:
            каждый элемент обрабатывается независимо и суммируется.
            Так можно легко увеличивать объём вычислений.
        */
        sum += data[i] * data[i] + 1.0;
    }
    return sum;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Использование: mpiexec -n <P> .\\test.exe <N>\n");
            fprintf(stderr, "Где N - число элементов данных для обработки.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        if (rank == 0) {
            fprintf(stderr, "N должно быть положительным числом.\n");
        }
        MPI_Finalize();
        return 1;
    }

    /*
        Равномерное распределение данных между процессами.
        Если N не делится на size, часть процессов получает на 1 элемент больше.
    */
    int base = n / size;
    int rem = n % size;
    int local_n = base + (rank < rem ? 1 : 0);

    int offset = rank * base + (rank < rem ? rank : rem);

    double *local_data = (double *)malloc((size_t)local_n * sizeof(double));
    if (local_data == NULL) {
        if (rank == 0) {
            fprintf(stderr, "Ошибка выделения памяти для локальных данных.\n");
        }
        MPI_Finalize();
        return 1;
    }

    /*
        Заполняем локальный массив данными.
        Значения зависят от номера процесса и смещения, чтобы не было одинаковых данных у всех.
    */
    for (int i = 0; i < local_n; ++i) {
        local_data[i] = (double)(offset + i + 1) * (rank + 1.0);
    }

    /*
        Засекаем время в каждом процессе локально.
        В MPI время обычно измеряют через MPI_Wtime(), которое возвращает время в секундах.
    */
    double t_start = MPI_Wtime();
    double local_sum = local_work(local_data, local_n);
    double t_end_local = MPI_Wtime();

    /*
        Каждый процесс отправляет свой локальный результат в process 0.
        Это типичная схема "точка-точка": информация передаётся напрямую,
        без коллективной операции вроде MPI_Reduce.
    */
    double total_sum = 0.0;
    double partial = 0.0;

    if (rank == 0) {
        total_sum = local_sum;
        for (int src = 1; src < size; ++src) {
            MPI_Recv(&partial, 1, MPI_DOUBLE, src, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += partial;
        }

        double t_end_total = MPI_Wtime();

        printf("MPI benchmark\n");
        printf("Processes: %d\n", size);
        printf("N: %d\n", n);
        printf("Total sum: %.6f\n", total_sum);
        printf("Elapsed time: %.6f sec\n", t_end_total - t_start);
        printf("\n");
        printf("Чтобы посчитать ускорение и эффективность, нужно сделать два запуска:\n");
        printf("  1) mpiexec -n 1 .\\test.exe %d\n", n);
        printf("  2) mpiexec -n %d .\\test.exe %d\n", size, n);
        printf("\n");
        printf("Формулы:\n");
        printf("  Speedup = T1 / Tp\n");
        printf("  Efficiency = Speedup / P\n");
    } else {
        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}

/*
    Пример расчёта ускорения и эффективности:

      Если при 1 процессе время T1 = 2.0 c,
      а при 4 процессах время Tp = 0.6 c,
      то:
          Speedup = 2.0 / 0.6 = 3.333...
          Efficiency = 3.333 / 4 = 0.8333 = 83.33%

    Чем ближе efficiency к 1, тем лучше масштабируемость программы.
    На практике эффективность может быть меньше 1 из-за коммуникативной и организационной накладной.
*/
