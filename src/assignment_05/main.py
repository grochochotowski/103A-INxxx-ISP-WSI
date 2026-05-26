#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:51:50 2021

@author: Rafał Biedrzycki

Kodu tego mogą używać moi studenci na ćwiczeniach z przedmiotu
Wstęp do Sztucznej Inteligencji.

Kod ten powstał, aby przyspieszyć i ułatwić pracę studentów,
tak aby mogli skupić się na algorytmach sztucznej inteligencji.

Kod nie jest wzorem dobrej jakości programowania w Pythonie,
nie jest również wzorem programowania obiektowego
i może zawierać błędy.

Nie ma obowiązku używania tego kodu.
"""

import numpy as np
import matplotlib.pyplot as plt

# ToDo tu prosze podac najmłodsze cyfry numerow indeksow
# 336284 -> 4
# 348521 -> 1
p = [4, 1]

L_BOUND = -5
U_BOUND = 5

def q(x):
    return np.sin(x * np.sqrt(p[0] + 1)) + np.cos(x * np.sqrt(p[1] + 1))

x = np.linspace(L_BOUND, U_BOUND, 100)
y = q(x)

np.random.seed(1)

# f logistyczna jako przykład sigmoidalnej
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# pochodna fun. 'sigmoid'
def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

# f. straty
def nloss(y_out, y):
    return (y_out - y) ** 2

# pochodna f. straty
def d_nloss(y_out, y):
    return 2 * (y_out - y)

class DlNet:
    def __init__(self, x, y, hidden_l_size=9, lr=0.003):
        self.x = x
        self.y = y
        self.y_out = 0

        self.HIDDEN_L_SIZE = hidden_l_size
        self.LR = lr

        # ToDo inicjalizacja wag i biasów

        # Wagi i biasy między wejściem a warstwą ukrytą
        self.w1 = np.random.uniform(-1, 1, self.HIDDEN_L_SIZE)
        self.b1 = np.random.uniform(-1, 1, self.HIDDEN_L_SIZE)

        # Wagi i bias neuronu wyjściowego
        self.w2 = np.random.uniform(-1, 1, self.HIDDEN_L_SIZE)
        self.b2 = np.random.uniform(-1, 1)

    def forward(self, x):
        # ToDo propagacja w przód

        # Warstwa ukryta
        self.z1 = self.w1 * x + self.b1
        self.h1 = sigmoid(self.z1)

        # Warstwa wyjściowa liniowa
        self.y_out = np.dot(self.h1, self.w2) + self.b2

        return self.y_out

    def predict(self, x):
        # ToDo predykcja dla zbioru punktów
        result = []

        for xi in x:
            result.append(self.forward(xi))

        return np.array(result)

    def backward(self, x, y):
        # ToDo propagacja wsteczna

        # Najpierw przejście w przód
        y_out = self.forward(x)

        # Pochodna funkcji straty po wyjściu
        dL_dyout = d_nloss(y_out, y)

        # Gradienty dla warstwy wyjściowej
        dL_dw2 = dL_dyout * self.h1
        dL_db2 = dL_dyout

        # Gradienty dla warstwy ukrytej
        dL_dh1 = dL_dyout * self.w2
        dL_dz1 = dL_dh1 * d_sigmoid(self.z1)

        dL_dw1 = dL_dz1 * x
        dL_db1 = dL_dz1

        # Aktualizacja wag i biasów
        self.w2 -= self.LR * dL_dw2
        self.b2 -= self.LR * dL_db2

        self.w1 -= self.LR * dL_dw1
        self.b1 -= self.LR * dL_db1

    def train(self, x_set, y_set, iters):
        # ToDo uczenie sieci

        for i in range(0, iters):
            for j in range(len(x_set)):
                self.backward(x_set[j], y_set[j])


            # Kontrolne wypisanie błędu co 1000 iteracji
            if i % 1000 == 0:
                yh_tmp = self.predict(x_set)
                mse = np.mean(nloss(yh_tmp, y_set))
                print(f"Iteracja {i}, MSE = {mse:.6f}")




nn = DlNet(x, y, hidden_l_size=9, lr=0.003)
nn.train(x, y, 15000)


# ToDo tu umieścić wyniki (y) z sieci
yh = nn.predict(x)


# Wskaźniki jakości aproksymacji
mse = np.mean((yh - y) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(yh - y))

print()
print("Wskaźniki jakości aproksymacji:")
print(f"MSE  = {mse:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"MAE  = {mae:.6f}")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(x, y, 'r', label='Funkcja oryginalna')
plt.plot(x, yh, 'b', label='Aproksymacja sieci')

plt.legend()
plt.grid(True)
plt.show()

neuron_counts = [1, 2, 3, 5, 9, 15]

print("\nWpływ liczby neuronów:\n")

for neurons in neuron_counts:
    nn = DlNet(x, y, hidden_l_size=neurons, lr=0.003)
    nn.train(x, y, 15000)

    yh = nn.predict(x)

    mse = np.mean((yh - y) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(yh - y))

    print(
        f"Neurony: {neurons:2d} | "
        f"MSE={mse:.4f} | "
        f"RMSE={rmse:.4f} | "
        f"MAE={mae:.4f}"
    )