/*
    Разработать алгоритм решения задания, с учетом разделения вычислений между несколькими процессорами.
    Составить схему взаимодействия процессов.

    На маленькой улице ЧжуаньСю в городе Гонконг живут двести тысяч китайцев и находятся три банка.
    Каждый из этих банков принимает деньги от вкладчиков в трех валютах –
    китайских юанях, американских долларах и английских фунтах стерлингов.
    При этом если вкладчик хочет взять деньги в одном банке на улице ЧжуаньСю и положить в другой,
    то ему в первом банке выдается только расписка,
    которую он и относит во второй банк. В пятницу вечером банки подсчитывают,
    сколько денег и в какой валюте они должны соседям и отправляют инкассаторов отнести эти деньги.
    Написать программу, моделирующую обмен деньгами в пятницу вечером на улице Чжуань-Сю с дополнительным условием,
    что все пятничные расчеты банки проводят через Большой Банк, находящийся на улице Чжуань-Го.
    Использовать метод передачи информации «точка-точка».
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/*
    Модель:
    - Процесс 0: Большой Банк (центр расчетов).
    - Процессы 1,2,3: три банка на улице ЧжуаньСю.

    Валюты:
    0 - CNY (юани)
    1 - USD (доллары)
    2 - GBP (фунты)

    Идея обмена через Большой Банк:
    1) Каждый банк i считает, сколько он должен каждому банку j по каждой валюте.
       Эти суммы (расписки) он отправляет в Большой Банк.
    2) Большой Банк агрегирует все расписки и вычисляет чистую позицию каждого банка:
         net[получатель] += сумма
         net[отправитель] -= сумма
       Для каждой валюты отдельно.
    3) Большой Банк отправляет каждому банку итог:
       - если net > 0, банк должен ПОЛУЧИТЬ деньги;
       - если net < 0, банк должен ОТДАТЬ деньги;
       - если net = 0, баланс по валюте нулевой.

    Используется только point-to-point:
    MPI_Send / MPI_Recv
*/

#define BIG_BANK 0
#define BANK_COUNT 3
#define CURRENCIES 3

enum { CNY = 0, USD = 1, GBP = 2 };

const char *currency_name(int c) {
    if (c == CNY) return "CNY";
    if (c == USD) return "USD";
    return "GBP";
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double t_start = MPI_Wtime();
    if (size != BANK_COUNT + 1) {
        if (rank == BIG_BANK) {
            printf("Ошибка: запустите программу ровно на %d процессах.\n", BANK_COUNT + 1);
            printf("Пример: mpiexec -n %d .\\9_3.exe\n", BANK_COUNT + 1);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == BIG_BANK) {
        /* =========================
           ЛОГИКА БОЛЬШОГО БАНКА
           ========================= */

        /*
            receipts[from][to][cur]:
            сколько банк from должен банку to в валюте cur.
            Индексы банков from/to: 1..3 (0 не используется для обычных банков).
        */
        double receipts[BANK_COUNT + 1][BANK_COUNT + 1][CURRENCIES];
        for (int i = 0; i <= BANK_COUNT; ++i) {
            for (int j = 0; j <= BANK_COUNT; ++j) {
                for (int c = 0; c < CURRENCIES; ++c) {
                    receipts[i][j][c] = 0.0;
                }
            }
        }

        /* Получаем расписки от банков 1..3 */
        for (int from = 1; from <= BANK_COUNT; ++from) {
            /* Каждый банк отправляет матрицу 3x3x3, строка самого себя там ненулевая, остальне строки нулевые */
            // “плоская” версия 3D-таблицы [i][j][currency].
            double buffer[(BANK_COUNT + 1) * (BANK_COUNT + 1) * CURRENCIES];
            MPI_Recv(buffer, (BANK_COUNT + 1) * (BANK_COUNT + 1) * CURRENCIES, MPI_DOUBLE,
                     from, 100 + from, MPI_COMM_WORLD, &status);

            int idx = 0;
            // Пробегает buffer в том же порядке, в котором банк-отправитель его заполнял.
            for (int i = 0; i <= BANK_COUNT; ++i) {
                for (int j = 0; j <= BANK_COUNT; ++j) {
                    for (int c = 0; c < CURRENCIES; ++c) {
                        // Добавляет полученные значения в общую сводную таблицу главного банка.
                        // += (а не =), потому что цикл идёт по трём банкам -
                        // нужно накопить вклад каждого отправителя.
                        receipts[i][j][c] += buffer[idx++];
                    }
                }
            }
            /*
            Итог после внешнего цикла:
                receipts[i][j][c] содержит общую сумму по валюте 'c',
                которую банк 'i' должен банку 'j', с учётом всех полученных сообщений.
             */
        }

        /*
            Считаем чистую позицию net[bank][cur]:
            > 0 : банк должен получить
            < 0 : банк должен отдать
        */
        double net[BANK_COUNT + 1][CURRENCIES];
        // Заполняем net нулями.
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
                    net[to][c] += x;   /* получатель должен получить */
                    net[from][c] -= x; /* отправитель должен отдать */
                }
            }
        }

        printf("=== Большой Банк: итог клиринга ===\n");
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

        /* Отправляем каждому банку его чистую позицию */
        for (int b = 1; b <= BANK_COUNT; ++b) {
            MPI_Send(net[b], CURRENCIES, MPI_DOUBLE, b, 200 + b, MPI_COMM_WORLD);
        }

    } else {
        /* =========================
           ЛОГИКА ОБЫЧНОГО БАНКА
           rank = 1..3
           ========================= */

        int my_bank = rank;

        /*
            Пример исходных расписок.
            debts[to][cur] = сколько МОЙ банк должен банку to в валюте cur.
            Можно менять значения для экспериментов.
        */
        double debts[BANK_COUNT + 1][CURRENCIES];
        for (int to = 0; to <= BANK_COUNT; ++to) {
            for (int c = 0; c < CURRENCIES; ++c) {
                debts[to][c] = 0.0;
            }
        }

        /* Чтобы пример был наглядным, зададим простые фиксированные данные */
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

        /*
            Формируем буфер расписок формата [from][to][currency].
            Этот банк заполняет только свою строку from = my_bank.
        */
        double out[(BANK_COUNT + 1) * (BANK_COUNT + 1) * CURRENCIES];
        for (int i = 0; i < (BANK_COUNT + 1) * (BANK_COUNT + 1) * CURRENCIES; ++i) {
            out[i] = 0.0;
        }

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

        /* Отправляем расписки в Большой Банк */
        MPI_Send(out, (BANK_COUNT + 1) * (BANK_COUNT + 1) * CURRENCIES, MPI_DOUBLE,
                 BIG_BANK, 100 + my_bank, MPI_COMM_WORLD);

        /* Получаем итог клиринга от Большого Банка */
        double my_net[CURRENCIES];
        MPI_Recv(my_net, CURRENCIES, MPI_DOUBLE, BIG_BANK, 200 + my_bank, MPI_COMM_WORLD, &status);

        printf("Банк %d получил итог от Большого Банка:\n", my_bank);
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
Схема взаимодействия процессов (point-to-point):

    P1 (Банк 1) ----\
                      \
    P2 (Банк 2) -------> P0 (Большой Банк) -----> P1 (итог net для банка 1)
                      /                          -----> P2 (итог net для банка 2)
    P3 (Банк 3) ----/                           -----> P3 (итог net для банка 3)

Где:
- P1,P2,P3 отправляют P0 свои "расписки" (кто кому сколько должен по 3 валютам),
- P0 выполняет клиринг (сводный расчет),
- P0 отправляет каждому банку его чистую позицию.

Пример вывода программы:
Банк 1 получил итог от Большого Банка:
  Отдать   800.00 CNY
  Отдать   100.00 USD
  Баланс   0.00 GBP
Банк 2 получил итог от Большого Банка:
  Получить 500.00 CNY
  Отдать   100.00 USD
  Получить 30.00 GBP
Банк 3 получил итог от Большого Банка:
  Получить 300.00 CNY
  Получить 200.00 USD
  Отдать   30.00 GBP
=== Большой Банк: итог клиринга ===
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
Время выполнения (максимум по процессам): 0.000649 сек
*/