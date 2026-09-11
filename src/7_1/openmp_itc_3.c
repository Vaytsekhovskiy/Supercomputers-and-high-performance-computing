#include <stdio.h>
#include <math.h>
#include <omp.h>
#define N 1000
int main(int argc, char** argv)
{
    double b[N];
    double s=0;
    int i;
    #pragma omp parallel for
    for (i=0; i<N; i++)
    {
        b[i]=i*tan(i*3.14/N);
    }
    int j;
    for (j=0; j<N; j++)
    {
        s+=b[j];
    }
    printf("%f",s);
    return 0;
}
/*
Было:
#pragma omp parallel for
for (i=0; i<N; i++)
{
b[i]=i*tan(i*3.14/N);
s+=b[i];
}

Стало:
#pragma omp parallel for
for (i=0; i<N; i++)
{
b[i]=i*tan(i*3.14/N);
}
int j;
for (j=0; j<N; j++)
{
s+=b[j];
}

Было исправлено, так как переменная s является общей для всех потоков,
и при параллельном доступе к ней возникает гонка данных.
Например:

1. было s = 100
2. поток A хочет добавить 5, поток B — 7
3. оба прочитали 100
4. A записал 105
5. B записал 107
Итог 107 вместо правильного 112. Гонка данных, произошла потеря обновления.

Теперь каждый поток записывает свои значения в массив b,
а затем в последовательном цикле суммируются все элементы массива b в переменную s.
*/