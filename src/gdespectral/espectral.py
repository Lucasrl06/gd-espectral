import math


def calcular_passo(nome, mu, L):
    if nome == '1/L':
        return 1/L
    if nome == '2/(L+mu)':
        return 2/(L + mu)
    if nome == '2.1/L':
        return 2.1/L
    raise ValueError('Passo desconhecido')


def iteracoes_previstas(mu, L, eta, tolerancia):
    q = max(abs(1 - eta*mu), abs(1 - eta*L))
    if q >= 1:
        return None
    if q == 0:
        return 1
    return math.ceil(math.log(tolerancia) / math.log(q))
