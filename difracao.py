"""Difração de Fraunhofer calculada pela soma de fasores.

Princípio de Huygens: cada ponto aberto da abertura se comporta como uma fonte
de ondas secundárias. No anteparo, o campo elétrico é a soma dessas ondas, e
cada uma é representada por um fasor (um número complexo com módulo e fase).

Na aproximação de Fraunhofer (anteparo distante, L muito maior que a abertura),
a onda que sai do ponto (x, y) da abertura chega ao ponto (X, Y) do anteparo com
uma fase, em relação à onda que sai do centro da abertura, igual a

        φ = k · (x·X + y·Y) / L,        com k = 2π / λ.

O campo e a intensidade no anteparo são, então,

        E(X, Y) = Σ t(x, y) · e^(−iφ)       (soma sobre todos os pixels)
        I(X, Y) = |E(X, Y)|²
"""

from dataclasses import dataclass

import numpy as np

from aberturas import Abertura


@dataclass
class PadraoDeDifracao:
    """Intensidade no anteparo e as posições onde ela foi calculada.

    Atributos:
        intensidade: matriz I/I₀ (linhas = Y, de cima para baixo; colunas = X).
        x: posições horizontais no anteparo, em metros (da esquerda para a direita).
        y: posições verticais no anteparo, em metros (de cima para baixo).
        comprimento_onda: λ usado no cálculo, em metros.
        distancia: distância L entre a abertura e o anteparo, em metros.
    """

    intensidade: np.ndarray
    x: np.ndarray
    y: np.ndarray
    comprimento_onda: float
    distancia: float

    def perfil_central(self) -> np.ndarray:
        """Intensidade ao longo da linha horizontal que passa pelo centro (Y = 0)."""
        return self.intensidade[len(self.y) // 2]


def calcular_padrao_fraunhofer(
    abertura: Abertura,
    comprimento_onda: float,
    distancia: float,
    largura_anteparo: float,
    altura_anteparo: float,
    resolucao: int = 801,
) -> PadraoDeDifracao:
    """Calcula o padrão de difração no anteparo somando os fasores de cada pixel.

    Args:
        abertura: objeto/fenda iluminado por uma onda plana.
        comprimento_onda: λ da luz, em metros.
        distancia: distância L entre a abertura e o anteparo, em metros.
        largura_anteparo: largura da região calculada do anteparo, em metros.
        altura_anteparo: altura da região calculada do anteparo, em metros.
        resolucao: número de pontos calculados ao longo da largura do anteparo.
    """
    if resolucao < 3:
        raise ValueError("a resolução do anteparo deve ter pelo menos 3 pontos")

    k = 2 * np.pi / comprimento_onda  # número de onda
    x_abertura, y_abertura = abertura.coordenadas()  # fontes secundárias (pixels)

    passo = largura_anteparo / (resolucao - 1)
    x_anteparo = _eixo_simetrico(largura_anteparo / 2, passo)
    y_anteparo = _eixo_simetrico(altura_anteparo / 2, passo)[::-1]  # de cima para baixo

    # A fase se separa em uma parte horizontal e outra vertical:
    #     e^(−iφ) = e^(−ik·x·X/L) · e^(−ik·y·Y/L)
    # então basta uma matriz de fasores para cada direção.
    fasores_x = np.exp(-1j * k * np.outer(x_anteparo, x_abertura) / distancia)  # [X, coluna]
    fasores_y = np.exp(-1j * k * np.outer(y_anteparo, y_abertura) / distancia)  # [Y, linha]

    # Soma de fasores: E(X, Y) = Σ_linhas Σ_colunas e^(−ik·y·Y/L) · t · e^(−ik·x·X/L).
    # Escrita como produto de matrizes, a soma dupla sobre todos os pixels fica
    # em uma única linha:
    campo = fasores_y @ abertura.transmitancia @ fasores_x.T

    # Cada pixel não é um ponto, e sim um quadradinho aberto de lado Δ. Somar os
    # fasores dentro dele dá o mesmo fator de uma fenda simples, sinc(Δ·X / λL),
    # em cada direção. Lembrete: np.sinc(u) = sen(πu) / (πu).
    escala = abertura.tamanho_pixel / (comprimento_onda * distancia)
    campo *= np.outer(np.sinc(escala * y_anteparo), np.sinc(escala * x_anteparo))

    intensidade = np.abs(campo) ** 2
    return PadraoDeDifracao(
        intensidade=intensidade / intensidade.max(),
        x=x_anteparo,
        y=y_anteparo,
        comprimento_onda=comprimento_onda,
        distancia=distancia,
    )


def intensidade_teorica_fenda_dupla(
    x: np.ndarray,
    largura_fenda: float,
    separacao: float,
    comprimento_onda: float,
    distancia: float,
) -> np.ndarray:
    """Intensidade I/I₀ prevista pela teoria para a fenda dupla, na linha Y = 0.

        I/I₀ = cos²(π·d·senθ / λ) · sinc²(π·a·senθ / λ),   com senθ ≈ x / L

    O cos² vem da interferência entre as duas fendas (as franjas) e o sinc² vem
    da difração em cada fenda (o envelope que modula as franjas).
    """
    sen_theta = x / distancia
    interferencia = np.cos(np.pi * separacao * sen_theta / comprimento_onda) ** 2
    difracao = np.sinc(largura_fenda * sen_theta / comprimento_onda) ** 2
    return interferencia * difracao


def encontrar_maximos_e_minimos(perfil: np.ndarray, limiar: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Índices dos máximos e dos mínimos de intensidade de um perfil.

    Máximos: pontos mais claros que os vizinhos e com intensidade acima de
    `limiar` × intensidade máxima (ignora oscilações desprezíveis).
    Mínimos: o ponto mais escuro entre cada par de máximos vizinhos.
    """
    meio = perfil[1:-1]
    eh_maximo = (meio > perfil[:-2]) & (meio >= perfil[2:]) & (meio >= limiar * perfil.max())
    maximos = np.flatnonzero(eh_maximo) + 1

    minimos = np.array(
        [inicio + np.argmin(perfil[inicio:fim]) for inicio, fim in zip(maximos[:-1], maximos[1:])],
        dtype=int,
    )
    return maximos, minimos


def numero_de_fresnel(abertura: Abertura, comprimento_onda: float, distancia: float) -> float:
    """F = a² / (λ·L), com a = metade da maior dimensão da região aberta.

    A aproximação de Fraunhofer vale quando F << 1, ou seja, quando o anteparo
    está longe o bastante para os raios que chegam a um ponto serem paralelos.
    """
    a = abertura.tamanho_da_regiao_aberta() / 2
    return a**2 / (comprimento_onda * distancia)


def avisos_de_validade(abertura: Abertura, padrao: PadraoDeDifracao) -> list[str]:
    """Mensagens de alerta para quando o modelo ou a amostragem podem falhar."""
    avisos = []

    fresnel = numero_de_fresnel(abertura, padrao.comprimento_onda, padrao.distancia)
    if fresnel >= 1:
        distancia_minima = fresnel * padrao.distancia  # L para o qual F = 1
        avisos.append(
            f"número de Fresnel F = {fresnel:.2f} ≥ 1: o anteparo está perto demais para a "
            f"aproximação de Fraunhofer valer. Use L bem maior que {distancia_minima:.3g} m."
        )

    # O detalhe mais fino do padrão tem período λL/D, onde D é o tamanho da região
    # aberta. Para enxergá-lo são necessários pelo menos 2 pontos por período.
    menor_periodo = padrao.comprimento_onda * padrao.distancia / abertura.tamanho_da_regiao_aberta()
    passo = padrao.x[1] - padrao.x[0]
    if passo > menor_periodo / 2:
        avisos.append(
            "o padrão tem detalhes menores que o espaçamento entre os pontos do anteparo: "
            "aumente a resolução ou diminua a largura do anteparo."
        )
    return avisos


def _eixo_simetrico(meia_largura: float, passo: float) -> np.ndarray:
    """Pontos de −meia_largura a +meia_largura, espaçados de `passo`, incluindo o zero."""
    n = round(meia_largura / passo)
    return np.arange(-n, n + 1) * passo
