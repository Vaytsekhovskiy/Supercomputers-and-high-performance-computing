#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
    БЕНЧМАРК MPI-программы "банки через Большой Банк" (point-to-point).

    Что измеряем:
    1) Время выполнения при разных наборах данных (разный объём расписок).
    2) Время при разном количестве процессов.
    3) Ускорение и эффективность.

    Формулы:
      Speedup    S = T1 / Tp
      Efficiency E = S / P

    Где:
      T1 - время при 1 процессе,
      Tp - время при P процессах,
      P  - число процессов.
*/

#define BIG_BANK 0
#define CURRENCIES 3
enum { CNY = 0, USD = 1, GBP = 2 };

typedef struct {
    int from;
    int to;
    int cur;
    double amount;
} Receipt;

/* Два тестовых набора фиксированных данных */
static const Receipt DATASET_SMALL[] = {
    {1, 2, CNY, 1200.0}, {1, 3, USD, 300.0},  {1, 2, GBP, 50.0},
    {2, 1, USD, 200.0},  {2, 3, CNY, 700.0},  {2, 1, GBP, 20.0},
    {3, 1, CNY, 400.0},  {3, 2, USD, 100.0},  {3, 1, GBP, 30.0}
};

static const Receipt DATASET_MEDIUM[] = {
    {1,2,CNY, 1000},{1,3,USD,200},{1,4,GBP,50},{1,5,CNY,130},{1,6,USD,70},{1,7,GBP,40},
    {2,1,CNY, 300},{2,3,USD,180},{2,4,GBP,20},{2,5,CNY,260},{2,6,USD,30},{2,7,GBP,10},
    {3,1,CNY, 150},{3,2,USD,120},{3,4,GBP,90},{3,5,CNY,80},{3,6,USD,60},{3,7,GBP,35},
    {4,1,CNY, 500},{4,2,USD,220},{4,3,GBP,70},{4,5,CNY,40},{4,6,USD,55},{4,7,GBP,25},
    {5,1,CNY, 410},{5,2,USD,140},{5,3,GBP,65},{5,4,CNY,95},{5,6,USD,88},{5,7,GBP,12},
    {6,1,CNY, 205},{6,2,USD,333},{6,3,GBP,17},{6,4,CNY,77},{6,5,USD,48},{6,7,GBP,29},
    {7,1,CNY, 99}, {7,2,USD,44}, {7,3,GBP,11},{7,4,CNY,23},{7,5,USD,67},{7,6,GBP,31}
};

static const char* currency_name(int c) {
    if (c == CNY) return "CNY";
    if (c == USD) return "USD";
    return "GBP";
}

/*
    Выбираем набор данных:
      argv[1] = small | medium
*/
static void choose_dataset(
    const char *name,
    const Receipt **dataset,
    int *count
) {
    if (name != NULL && strcmp(name, "medium") == 0) {
        *dataset = DATASET_MEDIUM;
        *count = (int)(sizeof(DATASET_MEDIUM) / sizeof(DATASET_MEDIUM[0]));
    } else {
        *dataset = DATASET_SMALL;
        *count = (int)(sizeof(DATASET_SMALL) / sizeof(DATASET_SMALL[0]));
    }
}

/*
    Получить ID банка из записи r по стороне отправителя/получателя.
*/
static int get_bank_id(const Receipt *r, int by_sender) {
    return by_sender ? r->from : r->to;
}

/*
    Локальная обработка расписок:
    каждый процесс проходит только свою часть [start, end) общего массива записей.
*/
static void process_local_chunk(
    const Receipt *dataset, int receipts_count,
    int rank, int size,
    int *local_max_bank_id,
    double **local_flat /* output: [banks+1][banks+1][3] в плоском виде */
) {
    int chunk = receipts_count / size;
    int rem = receipts_count % size;

    int local_n = chunk + (rank < rem ? 1 : 0);
    int start = rank * chunk + (rank < rem ? rank : rem);
    int end = start + local_n;

    int max_id = 0;
    for (int i = start; i < end; ++i) {
        if (dataset[i].from > max_id) max_id = dataset[i].from;
        if (dataset[i].to > max_id) max_id = dataset[i].to;
    }

    *local_max_bank_id = max_id;

    /* Временно пока неизвестен глобальный max_id, сам буфер создадим позже */
    *local_flat = NULL;
}

/*
    Заполнить локальный 3D-буфер (плоский) после того, как известен banks_count.
*/
static void fill_local_receipts_flat(
    const Receipt *dataset, int receipts_count,
    int rank, int size,
    int banks_count,
    double *flat /* size=(banks+1)*(banks+1)*3 */
) {
    int total = (banks_count + 1) * (banks_count + 1) * CURRENCIES;
    for (int i = 0; i < total; ++i) flat[i] = 0.0;

    int chunk = receipts_count / size;
    int rem = receipts_count % size;
    int local_n = chunk + (rank < rem ? 1 : 0);
    int start = rank * chunk + (rank < rem ? rank : rem);
    int end = start + local_n;

    for (int i = start; i < end; ++i) {
        int from = dataset[i].from;
        int to = dataset[i].to;
        int cur = dataset[i].cur;
        double amount = dataset[i].amount;

        int idx = ((from * (banks_count + 1) + to) * CURRENCIES) + cur;
        flat[idx] += amount;
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const Receipt *dataset = NULL;
    int receipts_count = 0;
    const char *dataset_name = (argc >= 2 ? argv[1] : "small");
    choose_dataset(dataset_name, &dataset, &receipts_count);

    /*
        Этап 1: определить глобальное число банков по выбранному набору.
        Убрана жёсткая проверка if (size != BANK_COUNT + 1):
        теперь процессоров может быть любое число >= 1.
    */
    int local_max_bank = 0;
    double *tmp = NULL;
    process_local_chunk(dataset, receipts_count, rank, size, &local_max_bank, &tmp);

    int global_max_bank = 0;
    if (rank == BIG_BANK) {
        global_max_bank = local_max_bank;
        for (int src = 1; src < size; ++src) {
            int x = 0;
            MPI_Recv(&x, 1, MPI_INT, src, 10, MPI_COMM_WORLD, &status);
            if (x > global_max_bank) global_max_bank = x;
        }
        for (int dst = 1; dst < size; ++dst) {
            MPI_Send(&global_max_bank, 1, MPI_INT, dst, 11, MPI_COMM_WORLD);
        }
    } else {
        MPI_Send(&local_max_bank, 1, MPI_INT, BIG_BANK, 10, MPI_COMM_WORLD);
        MPI_Recv(&global_max_bank, 1, MPI_INT, BIG_BANK, 11, MPI_COMM_WORLD, &status);
    }

    int banks_count = global_max_bank;
    if (banks_count <= 0) {
        if (rank == BIG_BANK) {
            printf("Нет данных для обработки.\n");
        }
        MPI_Finalize();
        return 0;
    }

    /*
        Этап 2: бенчмарк.
        Время замеряем на участке:
          - локальное формирование расписок из своей части;
          - point-to-point обмен с Большим Банком;
          - клиринг;
          - рассылка итогов.
    */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int flat_size = (banks_count + 1) * (banks_count + 1) * CURRENCIES;
    double *local_flat = (double*)malloc((size_t)flat_size * sizeof(double));
    if (!local_flat) {
        fprintf(stderr, "rank %d: Ошибка выделения памяти local_flat\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    fill_local_receipts_flat(dataset, receipts_count, rank, size, banks_count, local_flat);

    double *my_net = (double*)malloc((size_t)CURRENCIES * sizeof(double));
    if (!my_net) {
        fprintf(stderr, "rank %d: Ошибка выделения памяти my_net\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    for (int c = 0; c < CURRENCIES; ++c) my_net[c] = 0.0;

    if (rank == BIG_BANK) {
        double *receipts = (double*)malloc((size_t)flat_size * sizeof(double));
        if (!receipts) {
            fprintf(stderr, "rank 0: Ошибка выделения памяти receipts\n");
            MPI_Abort(MPI_COMM_WORLD, 4);
        }

        for (int i = 0; i < flat_size; ++i) receipts[i] = local_flat[i];

        for (int src = 1; src < size; ++src) {
            double *buf = (double*)malloc((size_t)flat_size * sizeof(double));
            if (!buf) {
                fprintf(stderr, "rank 0: Ошибка выделения памяти buf\n");
                MPI_Abort(MPI_COMM_WORLD, 5);
            }
            MPI_Recv(buf, flat_size, MPI_DOUBLE, src, 20, MPI_COMM_WORLD, &status);
            for (int i = 0; i < flat_size; ++i) receipts[i] += buf[i];
            free(buf);
        }

        double *net = (double*)malloc((size_t)(banks_count + 1) * CURRENCIES * sizeof(double));
        if (!net) {
            fprintf(stderr, "rank 0: Ошибка выделения памяти net\n");
            MPI_Abort(MPI_COMM_WORLD, 6);
        }
        for (int i = 0; i < (banks_count + 1) * CURRENCIES; ++i) net[i] = 0.0;

        for (int from = 1; from <= banks_count; ++from) {
            for (int to = 1; to <= banks_count; ++to) {
                if (from == to) continue;
                for (int c = 0; c < CURRENCIES; ++c) {
                    int idx = ((from * (banks_count + 1) + to) * CURRENCIES) + c;
                    double x = receipts[idx];
                    net[to * CURRENCIES + c] += x;
                    net[from * CURRENCIES + c] -= x;
                }
            }
        }

        for (int b = 1; b <= banks_count; ++b) {
            if (b % size == BIG_BANK) {
                /* Этот банк "живёт" на самом rank 0 — просто копируем, без MPI_Send/Recv */
                for (int c = 0; c < CURRENCIES; ++c) {
                    my_net[c] += net[b * CURRENCIES + c];
                }
            } else {
                MPI_Send(&net[b * CURRENCIES], CURRENCIES, MPI_DOUBLE, b % size, 21 + b, MPI_COMM_WORLD);
            }
        }

        free(net);
        free(receipts);
    } else {
        MPI_Send(local_flat, flat_size, MPI_DOUBLE, BIG_BANK, 20, MPI_COMM_WORLD);

        /* rank может "обслуживать" несколько банков: получаем их итоги */
        for (int b = 1; b <= banks_count; ++b) {
            if (b % size == rank) {
                double buf[CURRENCIES];
                MPI_Recv(buf, CURRENCIES, MPI_DOUBLE, BIG_BANK, 21 + b, MPI_COMM_WORLD, &status);
                for (int c = 0; c < CURRENCIES; ++c) my_net[c] += buf[c];
            }
        }
    }

    double t1 = MPI_Wtime();
    double local_elapsed = t1 - t0;

    /*
        Собираем максимальное время среди процессов на rank 0 через point-to-point.
        Это и есть корректное время параллельного шага (берём самый медленный процесс).
    */
    double elapsed_max = local_elapsed;
    if (rank == BIG_BANK) {
        for (int src = 1; src < size; ++src) {
            double x = 0.0;
            MPI_Recv(&x, 1, MPI_DOUBLE, src, 30, MPI_COMM_WORLD, &status);
            if (x > elapsed_max) elapsed_max = x;
        }

        /*
            T1: запускаем ту же программу с -n 1.
            В этой же строке считаем только Tp.
            Speedup/Efficiency считаем, если T1 передали вторым аргументом:
              mpiexec -n P .\9_3_benchmark.exe medium 0.123456
        */
        double T1 = -1.0;
        if (argc >= 3) T1 = atof(argv[2]);

        printf("=== BENCHMARK 9_3 (point-to-point) ===\n");
        printf("Dataset: %s\n", dataset_name);
        printf("Receipts: %d\n", receipts_count);
        printf("Banks detected: %d\n", banks_count);
        printf("Processes (P): %d\n", size);
        printf("Tp (max over ranks): %.6f sec\n", elapsed_max);

        if (T1 > 0.0) {
            double S = T1 / elapsed_max;
            double E = S / (double)size;
            printf("T1: %.6f sec\n", T1);
            printf("Speedup S = T1/Tp: %.6f\n", S);
            printf("Efficiency E = S/P: %.6f (%.2f%%)\n", E, E * 100.0);
        } else {
            printf("Для расчёта S и E сначала получите T1:\n");
            printf("  mpiexec -n 1 .\\9_3_benchmark.exe %s\n", dataset_name);
            printf("Затем запустите с P>1 и передайте T1 третьим параметром:\n");
            printf("  mpiexec -n %d .\\9_3_benchmark.exe %s <T1>\n", size, dataset_name);
        }

        /*
            Краткая схема взаимодействия процессов для бенчмарка:
            1) Все rank>0 -> rank0: локальные таблицы расписок (MPI_Send / MPI_Recv)
            2) rank0: клиринг
            3) rank0 -> rank'и: итоговые net по банкам (MPI_Send / MPI_Recv)
            4) Все rank>0 -> rank0: локальное время (MPI_Send / MPI_Recv)
        */
    } else {
        MPI_Send(&local_elapsed, 1, MPI_DOUBLE, BIG_BANK, 30, MPI_COMM_WORLD);
    }

    free(my_net);
    free(local_flat);

    MPI_Finalize();
    return 0;
}
/*
PS C:\projects\supercomps> mpiexec -n 1 ./cmake-build-debug/9_3_benchmark.exe small
=== BENCHMARK 9_3 (point-to-point) ===
Dataset: small
Receipts: 9
Banks detected: 3
Processes (P): 1
Tp (max over ranks): 0.000004 sec
Для расчёта S и E сначала получите T1:
  mpiexec -n 1 .\9_3_benchmark.exe small
Затем запустите с P>1 и передайте T1 третьим параметром:
  mpiexec -n 1 .\9_3_benchmark.exe small <T1>
PS C:\projects\supercomps> mpiexec -n 10 ./cmake-build-debug/9_3_benchmark.exe small 0.000004
=== BENCHMARK 9_3 (point-to-point) ===
Dataset: small
Receipts: 9
Banks detected: 3
Processes (P): 10
Tp (max over ranks): 0.000141 sec
T1: 0.000004 sec
Speedup S = T1/Tp: 0.028470
Efficiency E = S/P: 0.002847 (0.28%)
PS C:\projects\supercomps> mpiexec -n 100 ./cmake-build-debug/9_3_benchmark.exe small 0.000004
=== BENCHMARK 9_3 (point-to-point) ===
Dataset: small
Receipts: 9
Banks detected: 3
Processes (P): 100
Tp (max over ranks): 0.005230 sec
T1: 0.000004 sec
Speedup S = T1/Tp: 0.000765
Efficiency E = S/P: 0.000008 (0.00%)

ДЛЯ MEDIUM
PS C:\projects\supercomps> mpiexec -n 1 ./cmake-build-debug/9_3_benchmark.exe medium
=== BENCHMARK 9_3 (point-to-point) ===
Dataset: medium
Receipts: 42
Banks detected: 7
Processes (P): 1
Tp (max over ranks): 0.000005 sec
Для расчёта S и E сначала получите T1:
  mpiexec -n 1 .\9_3_benchmark.exe medium
Затем запустите с P>1 и передайте T1 третьим параметром:
  mpiexec -n 1 .\9_3_benchmark.exe medium <T1>
PS C:\projects\supercomps> mpiexec -n 10 ./cmake-build-debug/9_3_benchmark.exe medium 0.000005
=== BENCHMARK 9_3 (point-to-point) ===
Dataset: medium
Receipts: 42
Banks detected: 7
Processes (P): 10
Tp (max over ranks): 0.000272 sec
T1: 0.000005 sec
Speedup S = T1/Tp: 0.018403
Efficiency E = S/P: 0.001840 (0.18%)
PS C:\projects\supercomps> mpiexec -n 100 ./cmake-build-debug/9_3_benchmark.exe medium 0.000005
=== BENCHMARK 9_3 (point-to-point) ===
Dataset: medium
Receipts: 42
Banks detected: 7
Processes (P): 100
Tp (max over ranks): 0.008260 sec
T1: 0.000005 sec
Speedup S = T1/Tp: 0.000605
Efficiency E = S/P: 0.000006 (0.00%)
 */