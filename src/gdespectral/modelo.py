"""Perda quadrática, gradiente e Hessiana.

J(theta) = 1/(2m) * ||X theta - y||^2
"""
import numpy as np


def perda(X, y, theta):
    r = X @ theta - y
    return 0.5 * np.mean(r ** 2)


def gradiente(X, y, theta):
    m = X.shape[0]
    return X.T @ (X @ theta - y) / m


def hessiana(X):
    return X.T @ X / X.shape[0]


def solucao_lstsq(X, y):
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta
