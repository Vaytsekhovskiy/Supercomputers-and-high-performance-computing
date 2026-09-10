/*
3. Дана последовательность натуральных чисел {a0…an–1}. Создать
    OpenMP-приложение для вычисления общей суммы и всех промежуточных
    сумм простых чисел последовательности.

    Распараллелить программу при помощи директив OpenMP, откомпилировать и отладить в среде OpenMP.
 */

#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

static bool isPrime(int x) { // Функция для проверки, является ли число простым
    if (x < 2) return false;
    if (x == 2) return true;
    if (x % 2 == 0) return false;
    for (int d = 3; d * d <= x; d += 2) {
        if (x % d == 0) return false;
    }
    return true;
}

int main() {
    int n; // длина массива
    cin >> n;

    vector<int> a(n); // массив чисел
    for (int i = 0; i < n; ++i) cin >> a[i];

    vector<long long> primeValues(n, 0);

#pragma omp parallel for schedule(static)
// Разбиваем итерации цикла for между потоками,
// диапазон индексов делится заранее на примерно равные непрерывные куски.
    for (int i = 0; i < n; ++i) {
        if (isPrime(a[i])) {
            primeValues[i] = a[i];
        }
    }
// Подсчёт totalSum и prefixSums оставлен последовательным,
// потому что префиксные суммы имеют зависимость по данным
// (prefix[i] зависит от prefix[i-1]).
    vector<long long> prefixSums(n);
    long long totalSum = 0;
    for (int i = 0; i < n; ++i) {
        totalSum += primeValues[i];
        prefixSums[i] = totalSum;
    }

    cout << "Общая сумма простых чисел в последовательности: " << totalSum << '\n';
    cout << "Промежуточные суммы простых чисел на каждой итерации: ";
    for (int i = 0; i < n; ++i) {
        if (i) cout << ' ';
        cout << prefixSums[i];
    }
    cout << '\n';

    return 0;
}
/*
 5. Составить схему потоков.

 Поток 0 (main)
 │
 ├─ Ввод n и массива a[0..n-1]
 ├─ Инициализация primeValues[n] = 0
 │
 ├─ #pragma omp parallel for schedule(static)
 │   ├─ Поток T0: обрабатывает свой диапазон i, проверяет isPrime(a[i]), пишет primeValues[i]
 │   ├─ Поток T1: обрабатывает свой диапазон i, проверяет isPrime(a[i]), пишет primeValues[i]
 │   ├─ ...
 │   └─ Поток Tk: обрабатывает свой диапазон i, проверяет isPrime(a[i]), пишет primeValues[i]
 │
 ├─ (неявный барьер OpenMP в конце parallel for)
 │
 ├─ Последовательно:
 │   ├─ totalSum += primeValues[i]
 │   └─ prefixSums[i] = totalSum
 │
 └─ Вывод totalSum и prefixSums