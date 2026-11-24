import numpy as np
import pandas as pd
from experiments import config as C
from gdespectral.dados import gerar_dados, preparar
from gdespectral.modelo import hessiana, solucao_lstsq, perda
from gdespectral.gd import gradiente_descendente
from gdespectral.espectral import calcular_passo, iteracoes_previstas


def rodar():
    registros = []
    for fator, niveis in C.NIVEIS.items():
        for nivel in niveis:
            dados, y = gerar_dados(fator, nivel, C.OBSERVACOES, C.SEMENTE)
            rho = np.corrcoef(dados.T)[0, 1] if dados.shape[1] == 2 else None
            for modo in C.MODOS:
                X = preparar(dados, modo, intercepto=(fator == 'media'))
                H = hessiana(X)
                valores, vetores = np.linalg.eigh(H)
                mu, L = valores[0], valores[-1]
                if mu <= 0:
                    raise ValueError('Hessiana nao positiva definida')
                referencia = solucao_lstsq(X, y)
                for passo in C.PASSOS:
                    eta = calcular_passo(passo, mu, L)
                    prevista = iteracoes_previstas(mu, L, eta, C.TOLERANCIA)
                    for j in range(C.INICIALIZACOES):
                        semente = C.SEMENTE + j + 1
                        if j == C.INICIALIZACOES - 1:
                            indice = np.argmax(np.abs(1 - eta*valores))
                            theta0 = referencia + vetores[:, indice]
                            tipo = 'pior_caso'
                            semente = None
                        else:
                            theta0 = np.random.default_rng(semente).normal(size=X.shape[1])
                            tipo = 'aleatoria'
                        theta, k, estado = gradiente_descendente(
                            X, y, theta0, eta, referencia, C.TOLERANCIA,
                            C.MAX_ITER, C.FATOR_DIVERGENCIA)
                        observada = k if estado == 'convergiu' else None
                        registros.append({
                            'fator': fator, 'nivel': nivel, 'rho_amostral': rho,
                            'pre': modo, 'passo': passo, 'eta': eta,
                            'inicializacao': j+1, 'tipo_inicializacao': tipo,
                            'semente_dados': C.SEMENTE, 'semente_inicial': semente,
                            'mu': mu, 'L': L, 'kappa': L/mu,
                            'iter_prevista': prevista, 'iter_observada': observada,
                            'iter_executadas': k, 'estado': estado,
                            'razao_obs_prev': observada/prevista if observada is not None and prevista else None,
                            'erro_parametros': np.linalg.norm(theta - referencia),
                            'erro_relativo': np.linalg.norm(theta - referencia)/np.linalg.norm(theta0 - referencia),
                            'perda_final': perda(X, y, theta),
                            'perda_referencia': perda(X, y, referencia),
                        })
            print(f'{fator}: {nivel} concluido', flush=True)
    df = pd.DataFrame(registros)
    C.RESULTADOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RESULTADOS / 'execucoes.csv', index=False)
    return df


if __name__ == '__main__':
    rodar()
