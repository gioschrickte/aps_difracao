"""Simulador de difração de Fraunhofer pela soma de fasores.

O programa lê o objeto (um PNG ou uma matriz) ou gera uma fenda dupla, calcula a
intensidade no anteparo somando os fasores das ondas secundárias de cada ponto do
objeto e mostra e salva uma figura com o padrão de difração.

Uso:
    python main.py              simula a fenda dupla padrão
    python main.py --help       mostra todas as opções
"""

import argparse
import sys
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

from aberturas import Abertura, carregar_abertura, criar_fenda_dupla
from difracao import (
    PadraoDeDifracao,
    avisos_de_validade,
    calcular_padrao_fraunhofer,
    encontrar_maximos_e_minimos,
    intensidade_teorica_fenda_dupla,
    numero_de_fresnel,
)
from visualizacao import INTENSIDADE_MINIMA_LOG, desenhar_resultado

# Fatores de conversão para metros
NANO, MICRO, MILI = 1e-9, 1e-6, 1e-3

EXEMPLOS = """\
exemplos:
  python main.py                                        fenda dupla padrão
  python main.py --separacao 500 --lambda 532           outra fenda dupla, laser verde
  python main.py --distancia 2 --log                    anteparo mais longe, escala log
  python main.py exemplos/abertura_circular.png --log   objeto lido de um PNG
  python main.py exemplos/matriz_fenda_dupla.txt --largura-objeto 0.2
"""


def _numero_positivo(converter: Callable[[str], float], nome: str) -> Callable[[str], float]:
    """Cria um tipo para o argparse que só aceita números maiores que zero."""

    def validar(texto: str) -> float:
        try:
            valor = converter(texto)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{texto}' não é um {nome} válido") from None
        if valor <= 0:
            raise argparse.ArgumentTypeError(f"deve ser maior que zero (recebido: {texto})")
        return valor

    return validar


real_positivo = _numero_positivo(float, "número")
inteiro_positivo = _numero_positivo(int, "número inteiro")


def ler_argumentos() -> argparse.Namespace:
    """Define e lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Simula a imagem de difração de Fraunhofer de um objeto (PNG ou matriz)\n"
        "pela soma de fasores. Sem arquivo, simula uma fenda dupla.",
        epilog=EXEMPLOS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    objeto = parser.add_argument_group("objeto")
    objeto.add_argument(
        "arquivo",
        nargs="?",
        help="imagem (PNG, JPG...) ou matriz (.txt/.csv) do objeto; branco ou 1 = a luz passa",
    )
    objeto.add_argument(
        "--largura-objeto",
        type=real_positivo,
        default=1.0,
        metavar="mm",
        help="largura real que a imagem inteira representa (padrão: %(default)s mm)",
    )
    objeto.add_argument(
        "--inverter",
        action="store_true",
        help="considera preto ou 0 como a região por onde a luz passa",
    )

    fenda = parser.add_argument_group("fenda dupla (usada quando nenhum arquivo é informado)")
    fenda.add_argument(
        "--largura-fenda",
        type=inteiro_positivo,
        default=40,
        metavar="µm",
        help="largura a de cada fenda (padrão: %(default)s µm)",
    )
    fenda.add_argument(
        "--separacao",
        type=inteiro_positivo,
        default=250,
        metavar="µm",
        help="distância d entre os centros das fendas (padrão: %(default)s µm)",
    )
    fenda.add_argument(
        "--altura-fenda",
        type=inteiro_positivo,
        default=100,
        metavar="µm",
        help="altura h das fendas (padrão: %(default)s µm)",
    )

    experimento = parser.add_argument_group("experimento")
    experimento.add_argument(
        "--lambda",
        "--comprimento-onda",
        dest="comprimento_onda",
        type=real_positivo,
        default=632.8,
        metavar="nm",
        help="comprimento de onda da luz (padrão: %(default)s nm, laser He-Ne)",
    )
    experimento.add_argument(
        "--distancia",
        type=real_positivo,
        default=1.0,
        metavar="m",
        help="distância L entre o objeto e o anteparo (padrão: %(default)s m)",
    )

    anteparo = parser.add_argument_group("anteparo")
    anteparo.add_argument(
        "--largura-anteparo",
        type=real_positivo,
        default=40.0,
        metavar="mm",
        help="largura da região calculada do anteparo (padrão: %(default)s mm)",
    )
    anteparo.add_argument(
        "--altura-anteparo",
        type=real_positivo,
        metavar="mm",
        help="altura da região calculada do anteparo (padrão: igual à largura)",
    )
    anteparo.add_argument(
        "--resolucao",
        type=inteiro_positivo,
        default=801,
        metavar="N",
        help="número de pontos calculados ao longo da largura (padrão: %(default)s)",
    )

    saida = parser.add_argument_group("saída")
    saida.add_argument(
        "--log",
        action="store_true",
        help="escala logarítmica de intensidade (realça máximos fracos)",
    )
    saida.add_argument(
        "--saida",
        default="resultado_difracao.png",
        metavar="arquivo",
        help="arquivo da figura gerada (padrão: %(default)s)",
    )
    saida.add_argument(
        "--sem-janela",
        action="store_true",
        help="só salva a figura, sem abrir a janela",
    )
    return parser.parse_args()


def criar_abertura(args: argparse.Namespace) -> Abertura:
    """Objeto escolhido pelo usuário: o arquivo informado ou a fenda dupla."""
    if args.arquivo:
        return carregar_abertura(args.arquivo, args.largura_objeto * MILI, args.inverter)
    return criar_fenda_dupla(
        largura_fenda=args.largura_fenda * MICRO,
        separacao=args.separacao * MICRO,
        altura_fenda=args.altura_fenda * MICRO,
    )


def imprimir_resumo(abertura: Abertura, padrao: PadraoDeDifracao) -> None:
    """Mostra os parâmetros da simulação e os avisos de validade do modelo."""
    n_linhas, n_colunas = abertura.transmitancia.shape
    largura_mm = (padrao.x[-1] - padrao.x[0]) / MILI
    altura_mm = (padrao.y[0] - padrao.y[-1]) / MILI
    fresnel = numero_de_fresnel(abertura, padrao.comprimento_onda, padrao.distancia)

    linhas = [
        ("Objeto", abertura.descricao),
        ("Matriz do objeto", f"{n_colunas} × {n_linhas} pixels de {abertura.tamanho_pixel / MICRO:.3g} µm"),
        ("Comprimento de onda", f"{padrao.comprimento_onda / NANO:.1f} nm"),
        ("Distância L", f"{padrao.distancia:.3g} m"),
        ("Anteparo", f"{largura_mm:.1f} × {altura_mm:.1f} mm ({len(padrao.x)} × {len(padrao.y)} pontos)"),
        ("Número de Fresnel", f"{fresnel:.3g} (Fraunhofer exige F << 1)"),
    ]
    print("Simulação de difração de Fraunhofer (soma de fasores)")
    for rotulo, valor in linhas:
        print(f"  {rotulo:<21}{valor}")

    for aviso in avisos_de_validade(abertura, padrao):
        print(f"\nAVISO: {aviso}")


def imprimir_maximos_e_minimos(
    padrao: PadraoDeDifracao,
    maximos: np.ndarray,
    minimos: np.ndarray,
    por_lado: int = 5,
) -> None:
    """Tabela com os máximos próximos ao centro do anteparo e os mínimos entre eles."""
    if len(maximos) == 0:
        print("\nNenhum máximo de intensidade encontrado na linha central.")
        return

    perfil = padrao.perfil_central()
    central = np.argmin(np.abs(padrao.x[maximos]))  # posição do máximo central na lista
    escolhidos = maximos[max(0, central - por_lado) : central + por_lado + 1]
    entre = minimos[(minimos > escolhidos[0]) & (minimos < escolhidos[-1])]
    pontos = sorted([(i, "máximo") for i in escolhidos] + [(i, "mínimo") for i in entre])

    print(f"\nMáximos e mínimos na linha central (Y = 0), até {por_lado} máximos de cada lado:")
    print(f"  {'tipo':<8}{'X (mm)':>10}{'I/I₀':>10}")
    for indice, tipo in pontos:
        print(f"  {tipo:<8}{padrao.x[indice] / MILI:>10.3f}{perfil[indice]:>10.4f}")


def imprimir_comparacao_com_teoria(
    padrao: PadraoDeDifracao,
    minimos: np.ndarray,
    curva_teorica: np.ndarray,
    separacao: float,
) -> None:
    """Compara a fenda dupla simulada com as previsões da teoria."""
    espacamento_teorico = padrao.comprimento_onda * padrao.distancia / separacao
    diferenca = np.max(np.abs(padrao.perfil_central() - curva_teorica))

    print("\nComparação com a teoria da fenda dupla:")
    print(f"  Espaçamento entre franjas (λL/d): {espacamento_teorico / MILI:.3f} mm")

    # Os dois mínimos vizinhos do máximo central ficam em ±λL/(2d), então a
    # distância entre eles é o próprio espaçamento entre franjas.
    x_minimos = padrao.x[minimos]
    if np.any(x_minimos < 0) and np.any(x_minimos > 0):
        medido = x_minimos[x_minimos > 0].min() - x_minimos[x_minimos < 0].max()
        precisao = padrao.x[1] - padrao.x[0]  # a posição de cada mínimo tem erro de até meio passo
        print(f"  Medido na simulação:              {medido / MILI:.3f} ± {precisao / MILI:.3f} mm")
    print(f"  Maior diferença entre a simulação e a curva teórica: {diferenca:.1e}")


def main() -> None:
    """Lê as entradas, simula a difração e mostra o resultado."""
    args = ler_argumentos()
    comprimento_onda = args.comprimento_onda * NANO
    largura_anteparo = args.largura_anteparo * MILI
    altura_anteparo = (args.altura_anteparo or args.largura_anteparo) * MILI

    try:
        abertura = criar_abertura(args)
        padrao = calcular_padrao_fraunhofer(
            abertura, comprimento_onda, args.distancia, largura_anteparo, altura_anteparo, args.resolucao
        )
    except (OSError, ValueError) as erro:
        sys.exit(f"Erro: {erro}")

    # Máximos com menos de 1% da intensidade central são ignorados. Na escala log,
    # que mostra máximos bem mais fracos, o limite desce até o piso da escala.
    limiar = INTENSIDADE_MINIMA_LOG if args.log else 0.01
    maximos, minimos = encontrar_maximos_e_minimos(padrao.perfil_central(), limiar)
    imprimir_resumo(abertura, padrao)
    imprimir_maximos_e_minimos(padrao, maximos, minimos)

    # Na fenda dupla gerada, a e d são conhecidos: dá para comparar com a teoria.
    curva_teorica = None
    if not args.arquivo:
        separacao = args.separacao * MICRO
        curva_teorica = intensidade_teorica_fenda_dupla(
            padrao.x, args.largura_fenda * MICRO, separacao, comprimento_onda, args.distancia
        )
        imprimir_comparacao_com_teoria(padrao, minimos, curva_teorica, separacao)

    figura = desenhar_resultado(abertura, padrao, maximos, minimos, curva_teorica, args.log)
    try:
        figura.savefig(args.saida, dpi=150)
    except (OSError, ValueError) as erro:
        sys.exit(f"Erro ao salvar a figura: {erro}")
    print(f"\nFigura salva em: {args.saida}")

    if not args.sem_janela:
        plt.show()


if __name__ == "__main__":
    main()
