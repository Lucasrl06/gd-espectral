from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / 'experiments' / 'resultados'
FIGURAS = RAIZ / 'figures'
NIVEIS = {'correlacao': [0, 0.5, 0.9, 0.99],
          'media': [0, 1, 10, 100],
          'escala': [1, 10, 1000]}
MODOS = ['bruto', 'centrado', 'padronizado']
PASSOS = ['1/L', '2/(L+mu)', '2.1/L']
INICIALIZACOES = 5
OBSERVACOES = 200
SEMENTE = 2026
TOLERANCIA = 1e-6
MAX_ITER = 5000
FATOR_DIVERGENCIA = 1e6
