import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import config as C
from gdespectral.dados import gerar_dados, preparar
from gdespectral.modelo import gradiente, hessiana, solucao_lstsq


def gerar(df):
    C.FIGURAS.mkdir(exist_ok=True)
    fig, eixos = plt.subplots(1, 3, figsize=(11, 4))
    titulos = ['Correlação alvo', 'Razão média/desvio', 'Razão de escalas']
    for ax, fator, titulo in zip(eixos, C.NIVEIS, titulos):
        dados = df[df.fator == fator].drop_duplicates(['nivel', 'pre'])
        for modo in C.MODOS:
            d = dados[dados.pre == modo].sort_values('nivel')
            ax.plot(range(len(d)), d.kappa, 'o-', label=modo)
        ax.set_yscale('log')
        ax.set_xticks(range(len(C.NIVEIS[fator])))
        ax.set_xticklabels([str(n) for n in C.NIVEIS[fator]])
        ax.set_xlabel(titulo)
        ax.set_ylabel('Número de condição')
        ax.grid(alpha=.25)
    eixos[0].legend()
    fig.tight_layout()
    fig.savefig(C.FIGURAS / 'condicionamento.png', dpi=150)
    plt.close(fig)

    d = df[(df.passo == '1/L') & (df.estado == 'convergiu')]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for tipo in ['aleatoria', 'pior_caso']:
        p = d[d.tipo_inicializacao == tipo]
        ax.scatter(p.iter_prevista, p.iter_observada, label=tipo, alpha=.65)
    maior = max(d.iter_prevista.max(), d.iter_observada.max())
    ax.plot([1, maior], [1, maior], '--', label='Igualdade')
    ax.set(xscale='log', yscale='log', xlabel='Iterações previstas (pior caso)',
           ylabel='Iterações observadas', title='Passo 1/L — casos convergentes')
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURAS / 'observado_vs_previsto.png', dpi=150)
    plt.close(fig)

    dados, y = gerar_dados('correlacao', .9, C.OBSERVACOES, C.SEMENTE)
    X = preparar(dados, 'padronizado')
    H = hessiana(X)
    L = np.linalg.eigvalsh(H)[-1]
    referencia = solucao_lstsq(X, y)
    theta = np.random.default_rng(C.SEMENTE + 1).normal(size=2)
    pontos = [theta.copy()]
    for _ in range(30):
        theta = theta - gradiente(X, y, theta)/L
        pontos.append(theta.copy())
    pontos = np.array(pontos)
    a = np.linspace(min(pontos[:,0].min(), referencia[0])-.5,
                    max(pontos[:,0].max(), referencia[0])+.5, 120)
    b = np.linspace(min(pontos[:,1].min(), referencia[1])-.5,
                    max(pontos[:,1].max(), referencia[1])+.5, 120)
    A, B = np.meshgrid(a, b)
    u, v = A-referencia[0], B-referencia[1]
    Z = (H[0,0]*u*u + 2*H[0,1]*u*v + H[1,1]*v*v)/2
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.contour(A, B, Z, levels=np.geomspace(.01, max(Z.max(), .02), 12))
    ax.plot(pontos[:,0], pontos[:,1], 'o-', ms=3, label='Primeiras 30 atualizações')
    ax.scatter(*referencia, marker='*', s=100, label='Mínimo de referência')
    ax.set(xlabel='Parâmetro 1', ylabel='Parâmetro 2', title='Correlação alvo 0,9; passo 1/L')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURAS / 'trajetorias.png', dpi=150)
    plt.close(fig)
