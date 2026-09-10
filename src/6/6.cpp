#include <iostream>
#include <omp.h>
#include <stdio.h>
int main(int argc, char** argv)
{
    int maxThreads = omp_get_num_procs();  // 1) приложение определяет максимум потоков, доступных системе
    int threadsCount = (maxThreads < 4) ? maxThreads : 4; // 2) если максимальное количество нитей меньше четырех,
    // то параллельная секция должна выполняться с максимальным количеством
    // нитей, иначе установить количество нитей для параллельной секции равным 4;

    std::cout << "Максимально доступно потоков: " << maxThreads << std::endl;
    std::cout << "Будет использоваться потоков: " << threadsCount << std::endl;

    omp_set_dynamic(0);          // отключаем dynamic adjustment
    omp_set_num_threads(threadsCount);

#pragma omp parallel
    {
        int count = omp_get_thread_num();
        int ItsMe = omp_get_num_threads();
#pragma omp critical // 3) синхронизировать вывод потоков, что бы доступ к окну консоли был исключающим.
        {
            printf("Hello, OpenMP! I am %d of %d\n", count,ItsMe);
        }
    }
    return 0;
}