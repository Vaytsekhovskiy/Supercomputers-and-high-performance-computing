/*
    3. Дана последовательность натуральных чисел {a0…an–1}. Создать
    OpenMP-приложение для вычисления общей суммы и всех промежуточных
    сумм простых чисел последовательности.
    Реализовать алгоритм в последовательной программе
 */

#include <iostream>
#include <vector>

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

    vector<long long> prefixSums(n);
    long long totalSum = 0;

    for (int i = 0; i < n; ++i) {
        if (isPrime(a[i])) totalSum += a[i];
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
