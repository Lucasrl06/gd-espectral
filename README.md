# Projeto 3 — Geometria Espectral da Regressão e Convergência do Gradiente Descendente

## Objetivo

Implementar o gradiente descendente, comparar os parâmetros obtidos com a solução de mínimos quadrados de NumPy e observar separadamente os efeitos da correlação, da média e da escala no condicionamento da Hessiana e na convergência.

## Instalação e reprodução

Na pasta que contém este README, com Python 3.10 ou superior:

```bash
python -m pip install -e .
```

Comando único (testes, experimentos e figuras):

```bash
python -m experiments.reproduzir
```

No Windows, pode-se usar `py -3.14` no lugar de `python` se essa for a versão instalada. Os resultados ficam em `experiments/resultados/` e os gráficos em `figures/`. O notebook em `notebooks/` apresenta uma leitura dos resultados gerados.

## Experimentos

Recorte parcial da seção 7.7. Os fatores são variados separadamente, não em um cruzamento completo.

| Fator | Níveis | Modelo |
| --- | --- | --- |
| Correlação alvo | 0, 0.5, 0.9, 0.99 | Dois preditores sem intercepto, resposta centrada |
| Razão média/desvio | 0, 1, 10, 100 | Um preditor e intercepto |
| Razão de escalas | 1, 10, 1000 | Dois preditores sem intercepto, resposta centrada |

Em cada cenário são considerados os dados brutos, centrados e padronizados e os passos 1/L, 2/(L+mu) e 2.1/L. São cinco inicializações: quatro aleatórias e uma na direção de maior fator de contração em módulo (ou de crescimento, para o passo instável). São 11 x 3 x 3 x 5 = 495 execuções. Os preditores dos casos de correlação já são centrados e padronizados por construção; os três modos podem, portanto, coincidir. Os modelos têm dois parâmetros em todos os casos.

São geradas 200 observações sintéticas por cenário com semente 2026. Para a correlação usa-se uma combinação linear de duas variáveis aleatórias normais; a correlação amostral, que pode diferir do alvo, é registrada. O mesmo sorteio de base é reutilizado para cada nível. As respostas são sintéticas, com ruído normal de desvio 0.1. Na escala, os desvios dos preditores têm razão igual ao nível escolhido; no deslocamento da média, o desvio do preditor é 1. Os dados e parâmetros são adimensionais. As sementes das inicializações aleatórias são 2027 a 2030; a direção de pior caso é determinística, dada a Hessiana.

## Critério de parada

A referência é a solução de `numpy.linalg.lstsq`. O erro é a norma da diferença entre os parâmetros obtidos e a referência. A tolerância relativa é 1e-6 e o teto operacional é de 5.000 atualizações. `convergiu`: erro relativo ao inicial dentro da tolerância. `divergiu`: erro superior a 1e6 vezes o inicial ou não finito. `limite`: o teto foi atingido sem satisfazer os dois critérios anteriores.

As iterações observadas são registradas apenas quando há convergência. A previsão é calculada pelo fator espectral de pior caso, separadamente das atualizações efetivamente executadas. Para passos sem garantia de convergência global, a previsão não é registrada. No CSV, uma razão observada/prevista vazia indica que não foi possível calculá-la nas condições acima. Os casos com limite não são tratados como divergência nem como confirmação de convergência.

## Delimitação

Não foram estudadas as interações entre os fatores, os passos 0.5/L e 1.9/L, e foram utilizadas apenas cinco das vinte inicializações por configuração previstas no plano original. Os experimentos utilizam uma amostra sintética por cenário, sem avaliar a variabilidade entre diferentes amostras.

Não foi realizada uma avaliação preditiva em novos dados, uma vez que a convergência numérica não garante a capacidade de generalização do modelo. A investigação das hipóteses é parcial, especialmente nos casos que atingiram o limite de iterações.

Esta versão foi simplificada para priorizar métodos básicos de cálculo e programação, mantendo o foco na compreensão dos procedimentos e na interpretação dos resultados. Métodos numéricos mais avançados não foram explorados nesta etapa do projeto.