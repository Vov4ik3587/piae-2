# Лабораторная работа №2
# Вариант 6
# Модель двухфакторная, квадратичная, [-1; 1]
# Начальный план - сетка, шаг 0.1 по каждому фактору. веса равные
# Строить Ф2 оптимальный план, неполный вариант комбинированного алгоритма

import numpy as np


def model(x1, x2):
    return np.matrix([[1], [x1], [x2], [x1 * x2], [x1 * x1], [x2 * x2]])


def F2_optim(M):
    return -0.5 * np.trace(np.linalg.matrix_power(np.linalg.inv(M), 2))


def check_optimal_plan(D):
    max_functional = 0
    n = 41
    for i in range(0, n):
        for j in range(0, n):
            cur_functional = np.trace(np.dot(
                np.linalg.matrix_power(D, 3),
                np.vstack(model(-1 + 0.05 * i, -1 + 0.05 * j)) @ np.vstack(model(-1 + 0.05 * i, -1 + 0.05 * j)).T
            ))
            if cur_functional > max_functional:
                max_functional = cur_functional
    return max_functional


# Функция градиентного спуска по весам плана
def gradient_descent(n, x, p):
    proj = np.zeros(n)
    grad = np.zeros(n)
    out = open("output.txt", "w")
    solution = 0
    iteration = 0
    while solution == 0:
        lambdas = 0.1  # шаг
        solution = 1
        q = n - np.count_nonzero(p)

        M = np.zeros((6, 6))  # у двухфакторной квадратичной модели - 6
        for i in range(0, n):
            M += p[i] * (model(x[i][0], x[i][1]) * model(x[i][0], x[i][1]).T)  # информационная матрица M
        D = np.linalg.inv(M)  # дисперсионная матрица

        for i in range(0, n):  # TODO: выяснить, как считается производная моего функционала
            grad[i] = np.trace(
                np.linalg.matrix_power(D, 3) * (model(x[i][0], x[i][1])) * np.transpose(model(x[i][0], x[i][1])))

        grad /= np.linalg.norm(grad)  # нормируем градиент

        avg = 0.0  # среднее значение функций фи по ненулевым p
        for i in range(0, n):
            if p[i] != 0.0:
                avg += grad[i]
        avg /= n - q

        for i in range(0, n):  # условие решения задачи градиентного спуска (стр. 14)
            if p[i] != 0:
                if abs(grad[i] - avg) > 1e-10:
                    solution = 0

        for j in range(0, n):  # исключение точек из вектора р, чтобы они не менялись
            proj[j] = grad[j] - avg  # формула (7) на 15 стр.
            if p[j] == 0:
                if proj[j] > 0:
                    solution = 0
                else:
                    proj[j] = 0  # p[j] не изменится после рассмотрения новых точек (как бы исключаем)

        if iteration % 20 == 0:
            print("Iteration =", iteration, file=out)
            print("F2-criteria =", -1 * F2_optim(M), file=out)
            print("|proj| =", np.linalg.norm(proj), "\n", file=out)  # должно уменьшаться

        if solution == 0:
            for i in range(0, n):
                if proj[i] < 0 and lambdas > - p[i] / proj[i]:
                    lambdas = - p[i] / proj[i]

            for i in range(0, n):
                p[i] += lambdas * proj[i]  # рассматриваем новую точку
        iteration += 1

    print("Solution:", file=out)
    print("\nIteration =", iteration, file=out)
    print("\np:", file=out)
    for i in range(0, n):
        print(p[i], file=out)
    print("\nSolution F2-criteria =", -1 * F2_optim(M), file=out)
    print("|proj|=", np.linalg.norm(proj), file=out)

    print("\nChecking:", file=out)
    max_tr = check_optimal_plan(D)
    print("max_trace =", max_tr, file=out)


# Построим начальный план

l = 5  # количество точек на одном уровне начального плана
n = 25  # общее количество точек в начальном плане
p = np.ones(n)
weight = p * 0.04  # веса
spectr = np.zeros((n, 2))  # координаты точек

for i in range(0, l):
    for j in range(0, l):
        spectr[i * l + j][0] = -1 + 0.5 * i
        spectr[i * l + j][1] = -1 + 0.5 * j

gradient_descent(n, spectr, weight)
