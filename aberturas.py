"""Objeto difrator (abertura): a matriz de transmitância pela qual a luz passa.

A abertura pode vir de três lugares:

1. Uma imagem (PNG, JPG...):
       carregar_abertura("fenda.png", largura_fisica=1e-3)
2. Um arquivo de matriz (.txt ou .csv):
       carregar_abertura("matriz.txt", largura_fisica=0.3e-3)
3. Uma matriz criada diretamente em Python:
       matriz = np.zeros((100, 100))
       matriz[40:60, 45:55] = 1                  # uma abertura retangular
       abertura = Abertura(matriz, tamanho_pixel=5e-6)

Convenção: 1 (branco) = a luz passa; 0 (preto) = opaco. Valores intermediários
representam uma transmissão parcial da amplitude da onda.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

EXTENSOES_DE_MATRIZ = {".txt", ".csv"}


@dataclass
class Abertura:
    """Objeto/fenda pelo qual a luz passa.

    Atributos:
        transmitancia: matriz 2D com a fração da amplitude transmitida por cada
            pixel, de 0 (opaco) a 1 (transparente). A linha 0 é o topo da imagem.
        tamanho_pixel: lado de cada pixel, em metros.
        descricao: texto curto usado no título da figura e no resumo.
    """

    transmitancia: np.ndarray
    tamanho_pixel: float
    descricao: str = "objeto"

    def __post_init__(self) -> None:
        self.transmitancia = np.asarray(self.transmitancia, dtype=float)
        if self.transmitancia.ndim != 2:
            raise ValueError("a transmitância deve ser uma matriz 2D")
        if self.transmitancia.min() < 0 or self.transmitancia.max() > 1:
            raise ValueError("a transmitância deve ter valores entre 0 (opaco) e 1 (transparente)")
        if not np.any(self.transmitancia > 0):
            raise ValueError("a abertura está totalmente opaca: nenhuma luz passa por ela")
        if self.tamanho_pixel <= 0:
            raise ValueError("o tamanho do pixel deve ser positivo")

    @property
    def largura(self) -> float:
        """Largura total da matriz, em metros."""
        return self.transmitancia.shape[1] * self.tamanho_pixel

    @property
    def altura(self) -> float:
        """Altura total da matriz, em metros."""
        return self.transmitancia.shape[0] * self.tamanho_pixel

    def coordenadas(self) -> tuple[np.ndarray, np.ndarray]:
        """Posições x (colunas) e y (linhas) dos centros dos pixels, em metros.

        A origem fica no centro da matriz. Como a linha 0 é o topo da imagem,
        y diminui de cima para baixo.
        """
        n_linhas, n_colunas = self.transmitancia.shape
        x = _posicoes_centradas(n_colunas, self.tamanho_pixel)
        y = _posicoes_centradas(n_linhas, self.tamanho_pixel)[::-1]
        return x, y

    def tamanho_da_regiao_aberta(self) -> float:
        """Maior dimensão (largura ou altura) da região por onde a luz passa, em metros."""
        linhas, colunas = np.nonzero(self.transmitancia)
        largura = (colunas.max() - colunas.min() + 1) * self.tamanho_pixel
        altura = (linhas.max() - linhas.min() + 1) * self.tamanho_pixel
        return max(largura, altura)


def carregar_abertura(caminho: str | Path, largura_fisica: float, inverter: bool = False) -> Abertura:
    """Lê uma imagem (PNG, JPG...) ou uma matriz de texto (.txt/.csv) como abertura.

    Na imagem, branco = a luz passa e preto = opaco. Pixels transparentes (canal
    alfa) também bloqueiam a luz.

    Args:
        caminho: arquivo de entrada.
        largura_fisica: largura real que a imagem inteira representa, em metros.
        inverter: troca claro por escuro (útil para fendas desenhadas em preto).
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"arquivo não encontrado: {caminho}")

    if caminho.suffix.lower() in EXTENSOES_DE_MATRIZ:
        separador = "," if caminho.suffix.lower() == ".csv" else None
        claridade = np.loadtxt(caminho, delimiter=separador, ndmin=2)
        alfa = 1.0
    else:
        try:
            with Image.open(caminho) as imagem:
                # "LA" = luminância (tons de cinza) + alfa (transparência), de 0 a 255
                pixels = np.asarray(imagem.convert("LA"), dtype=float) / 255
        except UnidentifiedImageError:
            raise ValueError(f"não foi possível ler '{caminho}' como imagem") from None
        claridade, alfa = pixels[..., 0], pixels[..., 1]

    if inverter:
        claridade = 1 - claridade
    transmitancia = claridade * alfa

    tamanho_pixel = largura_fisica / transmitancia.shape[1]
    return Abertura(transmitancia, tamanho_pixel, descricao=caminho.name)


def criar_fenda_dupla(
    largura_fenda: float,
    separacao: float,
    altura_fenda: float,
    tamanho_pixel: float = 1e-6,
) -> Abertura:
    """Cria a matriz de uma fenda dupla: duas fendas retangulares iguais, lado a lado.

    Args:
        largura_fenda: largura a de cada fenda, em metros.
        separacao: distância d entre os centros das duas fendas, em metros.
        altura_fenda: altura h das fendas, em metros.
        tamanho_pixel: lado de cada pixel da matriz, em metros. As medidas acima
            são arredondadas para um número inteiro de pixels.
    """
    a = _em_pixels(largura_fenda, tamanho_pixel)
    d = _em_pixels(separacao, tamanho_pixel)
    h = _em_pixels(altura_fenda, tamanho_pixel)
    if d <= a:
        raise ValueError("a separação entre os centros das fendas deve ser maior que a largura de cada fenda")

    # Matriz quadrada com as fendas no centro. A borda opaca é só para a figura:
    # pixels opacos não emitem ondas secundárias e não alteram o padrão.
    lado = max(d + a, h) + 2 * a
    matriz = np.zeros((lado, lado))
    topo = (lado - h) // 2
    esquerda = (lado - (d + a)) // 2

    linhas = slice(topo, topo + h)
    fenda_esquerda = slice(esquerda, esquerda + a)
    fenda_direita = slice(esquerda + d, esquerda + d + a)  # centro a centro = d
    matriz[linhas, fenda_esquerda] = 1
    matriz[linhas, fenda_direita] = 1

    em_micrometros = tamanho_pixel * 1e6
    descricao = f"fenda dupla (a = {a * em_micrometros:g} µm, d = {d * em_micrometros:g} µm)"
    return Abertura(matriz, tamanho_pixel, descricao)


def _em_pixels(comprimento: float, tamanho_pixel: float) -> int:
    """Converte um comprimento (m) em um número inteiro de pixels (pelo menos 1)."""
    return max(1, round(comprimento / tamanho_pixel))


def _posicoes_centradas(quantidade: int, passo: float) -> np.ndarray:
    """Posições de `quantidade` pontos espaçados de `passo`, centradas em zero."""
    return (np.arange(quantidade) - (quantidade - 1) / 2) * passo
