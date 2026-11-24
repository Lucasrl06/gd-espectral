import numpy as np
from experiments import config as C
from gdespectral.dados import gerar_dados, preparar
from gdespectral.modelo import gradiente, hessiana, solucao_lstsq, perda
from gdespectral.gd import gradiente_descendente
from gdespectral.espectral import iteracoes_previstas


def test_gradiente_hessiana():
    X = np.array([[1., 2.], [3., 4.], [2., 1.]])
    y = np.array([1., 2., 3.])
    theta = np.array([.1, .2])
    np.testing.assert_allclose(gradiente(X, y, theta), hessiana(X) @ theta - X.T @ y/len(y))


def test_padronizacao():
    X, _ = gerar_dados('escala', 1000, 200, C.SEMENTE)
    Z = preparar(X, 'padronizado')
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1)


def test_solucao_e_perda():
    X, y = gerar_dados('correlacao', .5, 200, C.SEMENTE)
    ref = solucao_lstsq(X, y)
    theta0 = np.zeros(2)
    L = np.linalg.eigvalsh(hessiana(X))[-1]
    theta, _, estado = gradiente_descendente(X, y, theta0, 1/L, ref)
    assert estado == 'convergiu'
    np.testing.assert_allclose(theta, ref, atol=1e-5)
    assert perda(X, y, theta) < perda(X, y, theta0)


def test_divergencia_e_limite():
    X = np.eye(2)
    y = np.zeros(2)
    ref = np.zeros(2)
    inicio = np.ones(2)
    _, _, estado = gradiente_descendente(X, y, inicio, 4.2, ref)
    assert estado == 'divergiu'
    _, n, estado = gradiente_descendente(X, y, inicio, .01, ref, max_iter=1)
    assert estado == 'limite' and n == 1


def test_previsao():
    assert iteracoes_previstas(1, 1, 1, 1e-6) == 1
    assert iteracoes_previstas(1, 3, 2.1/3, 1e-6) is None
    assert iteracoes_previstas(1, 3, 1/3, 1e-6) == 35


def test_tamanho_do_plano():
    assert sum(map(len, C.NIVEIS.values())) * len(C.MODOS) * len(C.PASSOS) * C.INICIALIZACOES == 495
