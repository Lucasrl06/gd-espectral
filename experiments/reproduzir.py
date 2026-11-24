import subprocess
import sys
from experiments.rodar_plano import rodar
from experiments.gerar_figuras import gerar
from experiments import config as C


def main():
    subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=True)
    df = rodar()
    resumo = f'Total: {len(df)}\n' + df.estado.value_counts().to_string()
    resumo += '\n\n' + df.groupby(['fator', 'passo', 'estado']).size().to_string()
    (C.RESULTADOS / 'resumo.txt').write_text(resumo + '\n', encoding='utf-8')
    print(resumo)
    gerar(df)


if __name__ == '__main__':
    main()
