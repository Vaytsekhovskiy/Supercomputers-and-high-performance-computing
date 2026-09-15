#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/*

    В данной задаче необходимо реализовать рассылку значения n от корневого процесса
    ко всем остальным процессам двумя способами:

    1) вручную, с помощью двухточечных обменов MPI_Send / MPI_Recv;
    2) с помощью коллективной функции MPI_Bcast().

    После этого сравниваются времена выполнения этих двух способов.
    Это позволяет оценить, насколько оптимизирована библиотечная функция MPI_Bcast
    по сравнению с собственной реализацией на уровне point-to-point обменов.

    Важный момент: мы сравниваем не всю программу целиком, а именно стоимость рассылки
    одного значения n ко всем процессам. Поэтому основная работа здесь - это коммуникация,
    а не вычисления. Такое сравнение корректно, потому что обе реализации выполняют одну и ту же задачу:
    все процессы должны получить одно и то же значение n.
*/

static void broadcast_point_to_point(int root, int rank, int size, int *value)
{
    /*
        Эта функция реализует рассылку значения value от root ко всем остальным процессам
        без использования MPI_Bcast.

        Логика работы следующая:
        - если текущий процесс является корневым, он отправляет значение всем процессам,
          кроме самого себя, через отдельный MPI_Send;
        - если текущий процесс не корневой, он ждет только одного сообщения от root и принимает его
          через MPI_Recv.

        Такая схема корректна, потому что задача состоит в распространении одного значения
        ко всем процессам, а не в обмене между произвольными парами. Для этого достаточно одной
        корневой вершины и нескольких точечных отправок.
    */
    if (rank == root) {
        for (int dest = 0; dest < size; ++dest) {
            if (dest == root) continue;
            MPI_Send(value, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(value, 1, MPI_INT, root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    int n = 0; // значение, которое будет рассылаться
    /*
    Значения времени для передачи одного int будут порядка долей микросекунды —
    это уровень погрешности самого таймера и системного шума. Одно измерение здесь совершенно не показательно.
    Нужно усреднение по многим повторам:
     */
    const int repeats = 100000;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); // кол-во процессов

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    /*
        Если аргумент командной строки не передан, то корневой процесс задаёт значение n самостоятельно.
        Это нужно только для удобства запуска программы без промежуточного ввода.
    */
    if (rank == 0) {
        if (n <= 0) {
            n = 123456;
        }
    }

    /*
        Первый этап: собственная рассылка значения n через точечные обмены.
        Здесь измеряется именно время, которое требуется на отправку сообщений root-ом
        всем остальным процессам и их получение.
    */
    int p2p_value = 0;
    MPI_Barrier(MPI_COMM_WORLD);  // синхронизация старта у всех процессов
    double t_start_p2p = MPI_Wtime();
    for (int r = 0; r < repeats; ++r) {
        p2p_value = (rank == 0) ? n : 0;
        broadcast_point_to_point(0, rank, size, &p2p_value);
    }
    MPI_Barrier(MPI_COMM_WORLD);  // ждём, пока ВСЕ процессы завершат приём
    double t_end_p2p = MPI_Wtime();


    /*
        Второй этап: та же самая рассылка, но через сетевую библиотечную функцию MPI_Bcast.
        Здесь root передаёт значение всем процессам одним коллективным вызовом =>
        Стандартная коллективной операцией MPI.
    */
    int bcast_value = 0;
    if (rank == 0) {
        bcast_value = n;
    }

    MPI_Barrier(MPI_COMM_WORLD);  // синхронизация старта у всех процессов
    double t_start_bcast = MPI_Wtime();
    for (int r = 0; r < repeats; ++r) {
        MPI_Bcast(&bcast_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);  // ждём, пока ВСЕ процессы завершат приём
    double t_end_bcast = MPI_Wtime();

    /*
        В данном примере проверка корректности следующая: после рассылки каждый процесс
        должен получить одно и то же значение n. Для прозрачности мы не используем сложную
        дополнительную логику: достаточно убедиться, что оба способа передают одинаковое значение.
    */

    int local_ok = (p2p_value == bcast_value);
    int global_ok = 0;
    MPI_Reduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Correctness check: %s\n", global_ok ? "OK — все процессы получили верное значение" : "FAILED");
    }

    /*
        После замера времени сравниваем результаты двух реализаций.
        Ожидаемое поведение:
          - T_p2p обычно больше, чем T_bcast;
          - MPI_Bcast оптимизирована для коллективных обменов и обычно работает быстрее;
          - если t_p2p / t_bcast > 1, значит библиотечная функция выигрывает по скорости.

        Формулы:
          speedup = t_p2p / t_bcast
          relative_efficiency = speedup

        Здесь мы рассматриваем MPI_Bcast как эталон.
        Если speedup > 1, то собственная реализация медленнее MPI_Bcast;
        если speedup < 1, то собственная реализация оказалась быстрее.
    */
    double t_p2p = t_end_p2p - t_start_p2p;
    double t_bcast = t_end_bcast - t_start_bcast;

    if (rank == 0) {
        printf("MPI broadcast benchmark: point-to-point vs MPI_Bcast\n");
        printf("n = %d, processes = %d\n", n, size);
        printf("Time point-to-point: %.12f sec\n", t_p2p);
        printf("Time MPI_Bcast     : %.12f sec\n", t_bcast);

        if (t_bcast > 0.0) {
            double speedup = t_p2p / t_bcast;
            printf("Relative speedup (P2P / Bcast): %.6f\n", speedup);

            /*
                Время выполнения point-to-point на минимально осмысленном числе процессов,
                где рассылка вообще происходит, то есть P = 2 (root + 1 получатель)
             */
            double T_p2p_2 = -1.0;
            if (argc >= 3) T_p2p_2 = atof(argv[2]);
            /*
                Время выполнения bcast на минимально осмысленном числе процессов,
                где рассылка вообще происходит, то есть P = 2 (root + 1 получатель)
            */
            double T_bcast_2 = -1.0;
            if (argc >= 4) T_bcast_2 = atof(argv[3]);

            if (T_p2p_2 > 0.0 && T_bcast_2 > 0.0) {
                double S_p2p = T_p2p_2 / t_p2p;
                double E_p2p = S_p2p * 2.0 / (double)size; // делим на P/2, так как базой служит P=2, а не P=1
                printf("T_p2p_2: %.6f sec\n", T_p2p_2);
                printf("Speedup S_p2p = T_p2p_2/Tp: %.6f\n", S_p2p);
                printf("Efficiency E_p2p = S_p2p/P: %.6f (%.2f%%)\n", E_p2p, E_p2p * 100.0);

                double S_bcast = T_bcast_2 / t_bcast;
                double E_bcast = S_bcast * 2.0 / (double)size; // делим на P/2, так как базой служит P=2, а не P=1
                printf("T_bcast_2: %.6f sec\n", T_bcast_2);
                printf("Speedup S_bcast = T_bcast_2/Tp: %.6f\n", S_bcast);
                printf("Efficiency E_bcast = S_bcast/P: %.6f (%.2f%%)\n", E_bcast, E_bcast * 100.0);
            } else {
                printf("Для расчёта эффективности сначала получите базовые времена при P=2:\n");
                printf("  mpiexec -n 2 .\\10_3.exe %d\n", n);
                printf("Затем запустите с нужным P, передав оба базовых времени:\n");
                printf("  mpiexec -n %d .\\10_3.exe %d <T_p2p_2> <T_bcast_2>\n", size, n);
            }
        } else {
            printf("Relative speedup is undefined because t_bcast == 0\n");
        }
    }

    /*
        В конце программы печатаем значения, чтобы показать, что рассылка успешно завершилась.
        Переменная bcast_value на всех процессах должна содержать то же число, что и n у root.
    */
    if (rank == 0) {
        printf("Root value = %d\n", n);
        printf("Bcast value = %d\n", bcast_value);
    }

    MPI_Finalize();
    return 0;
}
/*
    Выводы программы:
    P = 2:
PS C:\projects\supercomps> mpiexec -n 2 ./cmake-build-debug/10_3.exe
Correctness check: OK — все процессы получили верное значение
MPI broadcast benchmark: point-to-point vs MPI_Bcast
n = 123456, processes = 2
Time point-to-point: 0.020433300000 sec
Time MPI_Bcast     : 0.019394600000 sec
Relative speedup (P2P / Bcast): 1.053556
Для расчёта эффективности сначала получите базовые времена при P=2:
  mpiexec -n 2 .\10_3.exe 123456
Затем запустите с нужным P, передав оба базовых времени:
  mpiexec -n 2 .\10_3.exe 123456 <T_p2p_2> <T_bcast_2>
Root value = 123456
Bcast value = 123456

    P = 4:
PS C:\projects\supercomps> mpiexec -n 4 ./cmake-build-debug/10_3.exe 123456 0.020433300000 0.019394600000
Correctness check: OK — все процессы получили верное значение
MPI broadcast benchmark: point-to-point vs MPI_Bcast
n = 123456, processes = 4
Time point-to-point: 0.039823300000 sec
Time MPI_Bcast     : 0.029698799999 sec
Relative speedup (P2P / Bcast): 1.340906
T_p2p_2: 0.020433 sec
Speedup S_p2p = T_p2p_2/Tp: 0.513099
Efficiency E_p2p = S_p2p/P: 0.256550 (25.65%)
T_bcast_2: 0.019395 sec
Speedup S_bcast = T_bcast_2/Tp: 0.653043
Efficiency E_bcast = S_bcast/P: 0.326522 (32.65%)
Root value = 123456
Bcast value = 123456

    P = 8:
PS C:\projects\supercomps> mpiexec -n 8 ./cmake-build-debug/10_3.exe 123456 0.020433300000 0.0193946000000
Correctness check: OK — все процессы получили верное значение
MPI broadcast benchmark: point-to-point vs MPI_Bcast
n = 123456, processes = 8
Time point-to-point: 0.108210200000 sec
Time MPI_Bcast     : 0.055306299999 sec
Relative speedup (P2P / Bcast): 1.956562
T_p2p_2: 0.020433 sec
Speedup S_p2p = T_p2p_2/Tp: 0.188830
Efficiency E_p2p = S_p2p/P: 0.047207 (4.72%)
T_bcast_2: 0.019395 sec
Speedup S_bcast = T_bcast_2/Tp: 0.350676
Efficiency E_bcast = S_bcast/P: 0.087669 (8.77%)
Root value = 123456
Bcast value = 123456

    P = 16:
PS C:\projects\supercomps> mpiexec -n 16 ./cmake-build-debug/10_3.exe 123456 0.020433300000 0.0193946000000
Correctness check: OK — все процессы получили верное значение
MPI broadcast benchmark: point-to-point vs MPI_Bcast
n = 123456, processes = 16
Time point-to-point: 0.335364999999 sec
Time MPI_Bcast     : 0.124698400001 sec
Relative speedup (P2P / Bcast): 2.689409
T_p2p_2: 0.020433 sec
Speedup S_p2p = T_p2p_2/Tp: 0.060929
Efficiency E_p2p = S_p2p/P: 0.007616 (0.76%)
T_bcast_2: 0.019395 sec
Speedup S_bcast = T_bcast_2/Tp: 0.155532
Efficiency E_bcast = S_bcast/P: 0.019442 (1.94%)
Root value = 123456
Bcast value = 123456

    P = 32:
PS C:\projects\supercomps> mpiexec -n 32 ./cmake-build-debug/10_3.exe 123456 0.020433300000 0.0193946000000
Correctness check: OK — все процессы получили верное значение
MPI broadcast benchmark: point-to-point vs MPI_Bcast
n = 123456, processes = 32
Time point-to-point: 0.844303300000 sec
Time MPI_Bcast     : 0.312263600001 sec
Relative speedup (P2P / Bcast): 2.703816
T_p2p_2: 0.020433 sec
Speedup S_p2p = T_p2p_2/Tp: 0.024201
Efficiency E_p2p = S_p2p/P: 0.001513 (0.15%)
T_bcast_2: 0.019395 sec
Speedup S_bcast = T_bcast_2/Tp: 0.062110
Efficiency E_bcast = S_bcast/P: 0.003882 (0.39%)
Root value = 123456
Bcast value = 123456

    P = 64:
PS C:\projects\supercomps> mpiexec -n 64 ./cmake-build-debug/10_3.exe 123456 0.020433300000 0.0193946000000
Correctness check: OK — все процессы получили верное значение
MPI broadcast benchmark: point-to-point vs MPI_Bcast
n = 123456, processes = 64
Time point-to-point: 1.837642400000 sec
Time MPI_Bcast     : 2.170725000000 sec
Relative speedup (P2P / Bcast): 0.846557
T_p2p_2: 0.020433 sec
Speedup S_p2p = T_p2p_2/Tp: 0.011119
Efficiency E_p2p = S_p2p/P: 0.000347 (0.03%)
T_bcast_2: 0.019395 sec
Speedup S_bcast = T_bcast_2/Tp: 0.008935
Efficiency E_bcast = S_bcast/P: 0.000279 (0.03%)
Root value = 123456
Bcast value = 123456

Сводная таблица результатов
P	    T_p2p (сек)	    T_bcast (сек)	Speedup(P2P/Bcast)	    E_p2p (%)	    E_bcast (%)
2	    0.020433	    0.019395	    1.054	                100.00 (база)	100.00 (база)
4	    0.039823	    0.029699	    1.341	                25.65	        32.65
8	    0.108210	    0.055306	    1.957	                4.72	        8.77
16	    0.335365	    0.124698	    2.689	                0.76	        1.94
32	    0.844303	    0.312264	    2.704	                0.15	        0.39
64	    1.837642	    2.170725	    0.847	                0.03	        0.03
 */
