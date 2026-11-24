import numpy as np
from gdespectral.modelo import gradiente


def gradiente_descendente(X, y, theta0, eta, referencia, tolerancia=1e-6,
                         max_iter=5000, fator_divergencia=1e6):
    theta = theta0.copy().astype(float)
    erro_inicial = np.linalg.norm(theta - referencia)
    if erro_inicial == 0:
        return theta, 0, 'convergiu'
    for k in range(1, max_iter + 1):
        theta = theta - eta * gradiente(X, y, theta)
        erro = np.linalg.norm(theta - referencia)
        if erro <= tolerancia * erro_inicial:
            return theta, k, 'convergiu'
        if not np.isfinite(erro) or erro > fator_divergencia * erro_inicial:
            return theta, k, 'divergiu'
    return theta, max_iter, 'limite'
