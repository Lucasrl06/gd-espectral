import numpy as np


def padronizar(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def gerar_dados(fator, nivel, m, semente):
    rng = np.random.default_rng(semente)
    a = padronizar(rng.normal(size=m))
    b = padronizar(rng.normal(size=m))
    ruido = 0.1 * rng.normal(size=m)
    if fator == 'media':
        # Um preditor de desvio 1 e intercepto (Proposicao 7.4).
        X = (a + nivel).reshape(-1, 1)
        y = 1 + 2*a + ruido
    else:
        if fator == 'correlacao':
            # A correlacao da amostra pode diferir do valor alvo.
            b = padronizar(nivel*a + np.sqrt(1 - nivel**2)*b)
            X = np.column_stack((a, b))
        elif fator == 'escala':
            X = np.column_stack((a, nivel*b))
        else:
            raise ValueError('Fator desconhecido')
        y = 2*a - b + ruido
        y = y - y.mean()
    return X, y


def preparar(X, modo, intercepto=False):
    if modo == 'centrado':
        X = X - X.mean(axis=0)
    elif modo == 'padronizado':
        X = padronizar(X)
    elif modo != 'bruto':
        raise ValueError('Pre-processamento desconhecido')
    if intercepto:
        X = np.column_stack((np.ones(len(X)), X))
    return X
