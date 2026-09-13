#include <mpi.h>
#include <stdio.h>
#define leng 20 //length of WR-string
int main( int argc, char **argv )
{
    double t,t2;
    int i,rank,size;
    char WR[leng];
    MPI_Status status;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    sprintf(WR,"Hello from %d",rank); // формирование сообщения
    t=MPI_Wtime(); // фиксация времени «начала посылки»,
    // локально для каждого процесса
    MPI_Send(WR,leng,MPI_CHAR,size-(rank+1),rank,MPI_COMM_WORLD); // (1)
    MPI_Recv(WR,leng,MPI_CHAR,size-(rank+1),size-(rank+1),MPI_COMM_WORLD,&status); // (2)
    t2=MPI_Wtime(); // фиксация времени «окончания приема»,
    // локально для каждого процесса
     printf("\n From processor %d\n WR=%s\n",rank,WR); // вывод сообщения
    printf("\n From processor %d\n Time=%le\n",rank,(t2-t)/100); // вывод времени,
    // затраченного на обмен данным процессором
    MPI_Finalize();
    return 0;
}
/*
1. Изучить рассмотренный ниже пример. Запустить программу на двух
процессорах. Поменять местами строки, помеченные в комментариях (1) и
(2), и запустить программу еще раз. Просмотреть выходной файл и файл
ошибок.

До смены строк местами:
mpiexec -n 2 .\cmake-build-debug\example.exe

From processor 1
WR=Hello from 0

From processor 1
Time=1.523000e-06

From processor 0
WR=Hello from 1

From processor 0
Time=1.372000e-06

После смены строк местами:
1. Происходит взаимная блокировка процессов (deadlock), так как оба процесса пытаются выполнить MPI_Recv до
того, как другой процесс выполнит MPI_Send.
В результате оба процесса ожидают друг друга и не могут продолжить выполнение,
что приводит к зависанию программы.

2. При попытке прервать программу (Ctrl+C) выводится сообщение об ошибке:
mpiexec aborting job...

job aborted:
[ranks] message

[0] job terminated by the user

[1] terminated

---- error analysis -----

[0] on COMP
ctrl-c was hit. job aborted by the user.

---- error analysis -----
*/