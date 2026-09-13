#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define BIG_BANK 0
#define BANK_COUNT 3
#define CURRENCIES 3

enum { CNY = 0, USD = 1, GBP = 2 };

/*
    9_3_packed делает ту же модель межбанковского клиринга, что и 9_3,
    но все обмены между процессами выполняются в упакованном виде через MPI_Pack/MPI_Unpack.
    Логика не меняется: каждый банк формирует свои обязательства, отправляет их в Большой Банк,
    центральный процесс сводит расчёты по трём валютам и возвращает каждому банку итоговую чистую позицию (получить/отдать).
    Смысл версии packed - проверить, как влияет упаковка сообщений на производительность.
    Время измеряется через MPI_Wtime() на основном участке работы,
    а итоговое Tp берётся как максимум по процессам, потому что общее завершение определяется самым медленным rank.
*/

static const char *currency_name(int c) {
    if (c == CNY) return "CNY";
    if (c == USD) return "USD";
    return "GBP";
}

/* Упаковать буфер расписок и отправить */
static void send_receipts_packed(int dst_rank, int tag,
                                 const double *flat, int flat_count,
                                 MPI_Comm comm)
{
    int sz_data = 0;
    int sz_count = 0;
    MPI_Pack_size(1, MPI_INT, comm, &sz_count);
    MPI_Pack_size(flat_count, MPI_DOUBLE, comm, &sz_data);

    int pack_size = sz_count + sz_data;
    char *pack_buf = (char *)malloc((size_t)pack_size);
    if (!pack_buf) {
        fprintf(stderr, "Ошибка выделения памяти send pack_buf\n");
        MPI_Abort(comm, 2);
    }

    int pos = 0;
    MPI_Pack((void *)&flat_count, 1, MPI_INT, pack_buf, pack_size, &pos, comm);
    MPI_Pack((void *)flat, flat_count, MPI_DOUBLE, pack_buf, pack_size, &pos, comm);

    MPI_Send(pack_buf, pos, MPI_PACKED, dst_rank, tag, comm);
    free(pack_buf);
}

/* Принять упакованный буфер расписок и распаковать */
static void recv_receipts_packed(int src_rank, int tag,
                                 double **flat_out, int *flat_count_out,
                                 MPI_Comm comm)
{
    MPI_Status st;
    MPI_Probe(src_rank, tag, comm, &st);

    int packed_nbytes = 0;
    MPI_Get_count(&st, MPI_PACKED, &packed_nbytes);

    char *pack_buf = (char *)malloc((size_t)packed_nbytes);
    if (!pack_buf) {
        fprintf(stderr, "Ошибка выделения памяти recv pack_buf\n");
        MPI_Abort(comm, 3);
    }

    MPI_Recv(pack_buf, packed_nbytes, MPI_PACKED, src_rank, tag, comm, &st);

    int pos = 0;
    int flat_count = 0;
    MPI_Unpack(pack_buf, packed_nbytes, &pos, &flat_count, 1, MPI_INT, comm);

    double *flat = (double *)malloc((size_t)flat_count * sizeof(double));
    if (!flat) {
        fprintf(stderr, "Ошибка выделения памяти recv flat\n");
        free(pack_buf);
        MPI_Abort(comm, 4);
    }

    MPI_Unpack(pack_buf, packed_nbytes, &pos, flat, flat_count, MPI_DOUBLE, comm);

    free(pack_buf);
    *flat_out = flat;
    *flat_count_out = flat_count;
}

/* Упаковать и отправить вектор net[3] */
static void send_net_packed(int dst_rank, int tag, const double net3[CURRENCIES], MPI_Comm comm)
{
    int sz_data = 0;
    int sz_count = 0;
    MPI_Pack_size(1, MPI_INT, comm, &sz_count);
    MPI_Pack_size(CURRENCIES, MPI_DOUBLE, comm, &sz_data);

    int pack_size = sz_count + sz_data;
    char *pack_buf = (char *)malloc((size_t)pack_size);
    if (!pack_buf) {
        fprintf(stderr, "Ошибка выделения памяти send net pack_buf\n");
        MPI_Abort(comm, 5);
    }

    int count = CURRENCIES;
    int pos = 0;
    MPI_Pack(&count, 1, MPI_INT, pack_buf, pack_size, &pos, comm);
    MPI_Pack((void *)net3, CURRENCIES, MPI_DOUBLE, pack_buf, pack_size, &pos, comm);

    MPI_Send(pack_buf, pos, MPI_PACKED, dst_rank, tag, comm);
    free(pack_buf);
}

/* Принять и распаковать вектор net[3] */
static void recv_net_packed(int src_rank, int tag, double net3[CURRENCIES], MPI_Comm comm)
{
    MPI_Status st;
    MPI_Probe(src_rank, tag, comm, &st);

    int packed_nbytes = 0;
    MPI_Get_count(&st, MPI_PACKED, &packed_nbytes);

    char *pack_buf = (char *)malloc((size_t)packed_nbytes);
    if (!pack_buf) {
        fprintf(stderr, "Ошибка выделения памяти recv net pack_buf\n");
        MPI_Abort(comm, 6);
    }

    MPI_Recv(pack_buf, packed_nbytes, MPI_PACKED, src_rank, tag, comm, &st);

    int pos = 0;
    int count = 0;
    MPI_Unpack(pack_buf, packed_nbytes, &pos, &count, 1, MPI_INT, comm);
    if (count != CURRENCIES) {
        fprintf(stderr, "Некорректный размер net: %d\n", count);
        free(pack_buf);
        MPI_Abort(comm, 7);
    }
    MPI_Unpack(pack_buf, packed_nbytes, &pos, net3, CURRENCIES, MPI_DOUBLE, comm);

    free(pack_buf);
}

/* Отправка локального времени через packed */
static void send_time_packed(int dst_rank, int tag, double t, MPI_Comm comm)
{
    int sz = 0;
    MPI_Pack_size(1, MPI_DOUBLE, comm, &sz);
    char *buf = (char *)malloc((size_t)sz);
    if (!buf) {
        fprintf(stderr, "Ошибка выделения памяти send time pack\n");
        MPI_Abort(comm, 8);
    }
    int pos = 0;
    MPI_Pack(&t, 1, MPI_DOUBLE, buf, sz, &pos, comm);
    MPI_Send(buf, pos, MPI_PACKED, dst_rank, tag, comm);
    free(buf);
}

/* Приём локального времени через packed */
static double recv_time_packed(int src_rank, int tag, MPI_Comm comm)
{
    MPI_Status st;
    MPI_Probe(src_rank, tag, comm, &st);
    int nbytes = 0;
    MPI_Get_count(&st, MPI_PACKED, &nbytes);

    char *buf = (char *)malloc((size_t)nbytes);
    if (!buf) {
        fprintf(stderr, "Ошибка выделения памяти recv time pack\n");
        MPI_Abort(comm, 9);
    }

    MPI_Recv(buf, nbytes, MPI_PACKED, src_rank, tag, comm, &st);

    int pos = 0;
    double t = 0.0;
    MPI_Unpack(buf, nbytes, &pos, &t, 1, MPI_DOUBLE, comm);
    free(buf);
    return t;
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double t_start = MPI_Wtime();
    if (size != BANK_COUNT + 1) {
        if (rank == BIG_BANK) {
            printf("Ошибка: запустите программу ровно на %d процессах.\n", BANK_COUNT + 1);
            printf("Пример: mpiexec -n %d .\\9_3_packed.exe\n", BANK_COUNT + 1);
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);


    if (rank == BIG_BANK) {
        double receipts[BANK_COUNT + 1][BANK_COUNT + 1][CURRENCIES];
        for (int i = 0; i <= BANK_COUNT; ++i) {
            for (int j = 0; j <= BANK_COUNT; ++j) {
                for (int c = 0; c < CURRENCIES; ++c) {
                    receipts[i][j][c] = 0.0;
                }
            }
        }

        for (int from = 1; from <= BANK_COUNT; ++from) {
            double *flat_in = NULL;
            int flat_count = 0;

            recv_receipts_packed(from, 100 + from, &flat_in, &flat_count, MPI_COMM_WORLD);

            int expected = (BANK_COUNT + 1) * (BANK_COUNT + 1) * CURRENCIES;
            if (flat_count != expected) {
                fprintf(stderr, "Некорректный размер packed receipts от банка %d: %d\n", from, flat_count);
                free(flat_in);
                MPI_Abort(MPI_COMM_WORLD, 10);
            }

            int idx = 0;
            for (int i = 0; i <= BANK_COUNT; ++i) {
                for (int j = 0; j <= BANK_COUNT; ++j) {
                    for (int c = 0; c < CURRENCIES; ++c) {
                        receipts[i][j][c] += flat_in[idx++];
                    }
                }
            }

            free(flat_in);
        }

        double net[BANK_COUNT + 1][CURRENCIES];
        for (int b = 0; b <= BANK_COUNT; ++b) {
            for (int c = 0; c < CURRENCIES; ++c) {
                net[b][c] = 0.0;
            }
        }

        for (int from = 1; from <= BANK_COUNT; ++from) {
            for (int to = 1; to <= BANK_COUNT; ++to) {
                if (from == to) continue;
                for (int c = 0; c < CURRENCIES; ++c) {
                    double x = receipts[from][to][c];
                    net[to][c] += x;
                    net[from][c] -= x;
                }
            }
        }

        printf("=== Большой Банк: итог клиринга (packed) ===\n");
        for (int b = 1; b <= BANK_COUNT; ++b) {
            printf("Банк %d:\n", b);
            for (int c = 0; c < CURRENCIES; ++c) {
                if (net[b][c] > 0.0) {
                    printf("  Получить %.2f %s\n", net[b][c], currency_name(c));
                } else if (net[b][c] < 0.0) {
                    printf("  Отдать   %.2f %s\n", -net[b][c], currency_name(c));
                } else {
                    printf("  Баланс   0.00 %s\n", currency_name(c));
                }
            }
        }

        for (int b = 1; b <= BANK_COUNT; ++b) {
            send_net_packed(b, 200 + b, net[b], MPI_COMM_WORLD);
        }

    } else {
        int my_bank = rank;

        double debts[BANK_COUNT + 1][CURRENCIES];
        for (int to = 0; to <= BANK_COUNT; ++to) {
            for (int c = 0; c < CURRENCIES; ++c) {
                debts[to][c] = 0.0;
            }
        }

        /* Фиксированные тестовые данные */
        if (my_bank == 1) {
            debts[2][CNY] = 1200.0;
            debts[3][USD] = 300.0;
            debts[2][GBP] = 50.0;
        } else if (my_bank == 2) {
            debts[1][USD] = 200.0;
            debts[3][CNY] = 700.0;
            debts[1][GBP] = 20.0;
        } else if (my_bank == 3) {
            debts[1][CNY] = 400.0;
            debts[2][USD] = 100.0;
            debts[1][GBP] = 30.0;
        }

        int flat_count = (BANK_COUNT + 1) * (BANK_COUNT + 1) * CURRENCIES;
        double *out = (double *)malloc((size_t)flat_count * sizeof(double));
        if (!out) {
            fprintf(stderr, "Ошибка выделения памяти out\n");
            MPI_Abort(MPI_COMM_WORLD, 11);
        }

        for (int i = 0; i < flat_count; ++i) out[i] = 0.0;

        int idx = 0;
        for (int from = 0; from <= BANK_COUNT; ++from) {
            for (int to = 0; to <= BANK_COUNT; ++to) {
                for (int c = 0; c < CURRENCIES; ++c) {
                    if (from == my_bank && to >= 1 && to <= BANK_COUNT && to != my_bank) {
                        out[idx] = debts[to][c];
                    } else {
                        out[idx] = 0.0;
                    }
                    ++idx;
                }
            }
        }

        send_receipts_packed(BIG_BANK, 100 + my_bank, out, flat_count, MPI_COMM_WORLD);
        free(out);

        double my_net[CURRENCIES];
        recv_net_packed(BIG_BANK, 200 + my_bank, my_net, MPI_COMM_WORLD);

        printf("Банк %d получил итог от Большого Банка (packed):\n", my_bank);
        for (int c = 0; c < CURRENCIES; ++c) {
            if (my_net[c] > 0.0) {
                printf("  Получить %.2f %s\n", my_net[c], currency_name(c));
            } else if (my_net[c] < 0.0) {
                printf("  Отдать   %.2f %s\n", -my_net[c], currency_name(c));
            } else {
                printf("  Баланс   0.00 %s\n", currency_name(c));
            }
        }
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    if (rank == BIG_BANK) {
        double max_time = local_time;
        for (int src = 1; src < size; ++src) {
            double t = 0.0;
            MPI_Recv(&t, 1, MPI_DOUBLE, src, 999, MPI_COMM_WORLD, &status);
            if (t > max_time) max_time = t;
        }
        printf("\n=== Замер времени MPI ===\n");
        printf("Процессов: %d\n", size);
        printf("Время выполнения (максимум по процессам): %.6f сек\n", max_time);
    } else {
        MPI_Send(&local_time, 1, MPI_DOUBLE, BIG_BANK, 999, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
/*
Банк 1 получил итог от Большого Банка (packed):
  Отдать   800.00 CNY
  Отдать   100.00 USD
  Баланс   0.00 GBP
=== Большой Банк: итог клиринга (packed) ===
Банк 1:
  Отдать   800.00 CNY
  Отдать   100.00 USD
  Баланс   0.00 GBP
Банк 2:
  Получить 500.00 CNY
  Отдать   100.00 USD
  Получить 30.00 GBP
Банк 3:
  Получить 300.00 CNY
  Получить 200.00 USD
  Отдать   30.00 GBP

=== Замер времени MPI ===
Процессов: 4
Время выполнения (максимум по процессам): 0.000860 сек

Итак, сравнивая программу с упаковкой/распаковкой (9_3.с) и ту же самую прогрмму,
но без упаковки/распаковки (9_3_packed.c) выяснилось следующее:
время выполнения для 9_3.c : в среднем 0.000649 сек
время выполнения для 9_3_packed.c : в среднем 0.000860 сек

Это ожидаемый результат: у 9_3_packed время больше,
потому что для вашего объёма данных упаковка добавляет накладные расходы, а выигрыша почти не даёт.
В 9_3 данные уже отправляются в простом непрерывном виде (double-массивы),
и MPI может передать их напрямую. В 9_3_packed перед каждой пересылкой добавляются MPI_Pack и MPI_Unpack,
то есть дополнительная работа на CPU и копирование памяти в промежуточные буферы и обратно.
При таких маленьких сообщениях и очень коротком общем времени
(доли миллисекунды) стоимость этих лишних операций заметнее,
чем потенциальная польза от «унифицированного формата» сообщения,
поэтому среднее время стало хуже: 0.000860>0.000649.
Итого, packing полезен, когда нужно передавать сложные/разрозненные структуры одним сообщением
или снижать количество пересылок; в данной задаче с маленькими плотными буферами он чаще даёт именно оверхед.

 */