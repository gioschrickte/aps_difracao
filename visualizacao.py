"""Figura do resultado: objeto, padrão de difração no anteparo e perfil de intensidade."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.figure import Figure

from aberturas import Abertura
from difracao import PadraoDeDifracao

# Cores: superfície clara, tintas para textos e eixos, e uma cor para cada série.
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_SUAVE = "#898781"
EIXO = "#c3c2b7"
GRADE = "#e1e0d9"
SEM_LUZ = "#0d0d0d"
COR_SIMULACAO = "#2a78d6"  # azul
COR_MAXIMOS = "#eb6834"  # laranja
COR_MINIMOS = "#1baf7a"  # verde-água

# Mapa de intensidade com um único matiz: do quase preto (sem luz) ao azul-claro
# (máximo de luz). Cada cor fica numa posição proporcional à sua luminosidade,
# para que o brilho cresça por igual ao longo da escala.
MAPA_INTENSIDADE = LinearSegmentedColormap.from_list(
    "intensidade",
    [
        (0.00, SEM_LUZ),
        (0.24, "#0d366b"),
        (0.43, "#1c5cab"),
        (0.62, "#3987e5"),
        (0.81, "#86b6ef"),
        (1.00, "#cde2fb"),
    ],
).with_extremes(bad=SEM_LUZ, under=SEM_LUZ)  # intensidade zero na escala log

INTENSIDADE_MINIMA_LOG = 1e-4  # piso da escala logarítmica

ESTILO = {
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "axes.edgecolor": EIXO,
    "axes.linewidth": 0.8,
    "axes.labelcolor": TINTA_SECUNDARIA,
    "axes.titlecolor": TINTA,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": EIXO,
    "ytick.color": EIXO,
    "xtick.labelcolor": TINTA_SUAVE,
    "ytick.labelcolor": TINTA_SUAVE,
    "grid.color": GRADE,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "font.size": 10,
}


def desenhar_resultado(
    abertura: Abertura,
    padrao: PadraoDeDifracao,
    maximos: np.ndarray,
    minimos: np.ndarray,
    curva_teorica: np.ndarray | None = None,
    escala_log: bool = False,
) -> Figure:
    """Monta a figura com o objeto, o padrão no anteparo e o perfil na linha central.

    Args:
        abertura: objeto/fenda usado na simulação.
        padrao: resultado de calcular_padrao_fraunhofer.
        maximos: índices dos máximos do perfil central.
        minimos: índices dos mínimos do perfil central.
        curva_teorica: intensidade prevista pela teoria no perfil central (opcional).
        escala_log: usa escala logarítmica de intensidade (realça máximos fracos).
    """
    with plt.rc_context(ESTILO):
        figura = plt.figure(figsize=(12, 10), layout="constrained")
        paineis = figura.subplot_mosaic(
            [["objeto", "padrao"], ["perfil", "perfil"]],
            width_ratios=[1, 1.25],
            height_ratios=[1.6, 1],
        )
        _desenhar_objeto(paineis["objeto"], abertura)
        _desenhar_padrao(paineis["padrao"], padrao, escala_log)
        _desenhar_perfil(paineis["perfil"], padrao, maximos, minimos, curva_teorica, escala_log)

        figura.suptitle(
            "Difração de Fraunhofer pela soma de fasores  ·  "
            f"λ = {padrao.comprimento_onda * 1e9:.1f} nm  ·  L = {padrao.distancia:.2f} m",
            color=TINTA,
            fontsize=14,
            fontweight="bold",
        )
    return figura


def _desenhar_objeto(eixo: Axes, abertura: Abertura) -> None:
    """Objeto em tons de cinza: preto = opaco, branco = a luz passa."""
    meia_largura = abertura.largura / 2 * 1e6  # µm
    meia_altura = abertura.altura / 2 * 1e6
    eixo.imshow(
        abertura.transmitancia,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=(-meia_largura, meia_largura, -meia_altura, meia_altura),
    )
    eixo.set_title(f"Objeto: {abertura.descricao}")
    eixo.set_xlabel("x (µm)")
    eixo.set_ylabel("y (µm)")
    eixo.set_anchor("N")  # alinha o topo com o painel do padrão
    _sem_moldura(eixo)


def _desenhar_padrao(eixo: Axes, padrao: PadraoDeDifracao, escala_log: bool) -> None:
    """Mapa de intensidade no anteparo, com barra de cores."""
    passo = padrao.x[1] - padrao.x[0]
    bordas_mm = 1e3 * np.array(
        [
            padrao.x[0] - passo / 2,
            padrao.x[-1] + passo / 2,
            padrao.y[-1] - passo / 2,
            padrao.y[0] + passo / 2,
        ]
    )
    if escala_log:
        normalizacao = LogNorm(vmin=INTENSIDADE_MINIMA_LOG, vmax=1)
    else:
        normalizacao = Normalize(vmin=0, vmax=1)
    imagem = eixo.imshow(padrao.intensidade, cmap=MAPA_INTENSIDADE, norm=normalizacao, extent=bordas_mm)

    barra = eixo.figure.colorbar(imagem, ax=eixo, fraction=0.05, pad=0.03)
    barra.outline.set_visible(False)
    if escala_log:
        barra.set_label("Intensidade relativa I/I₀ (escala log)")
    else:
        barra.set_label("Intensidade relativa I/I₀")
        barra.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0  mínimo", "0.25", "0.5", "0.75", "1  máximo"])

    eixo.set_title("Padrão no anteparo")
    eixo.set_xlabel("X (mm)")
    eixo.set_ylabel("Y (mm)")
    _sem_moldura(eixo)


def _desenhar_perfil(
    eixo: Axes,
    padrao: PadraoDeDifracao,
    maximos: np.ndarray,
    minimos: np.ndarray,
    curva_teorica: np.ndarray | None,
    escala_log: bool,
) -> None:
    """Intensidade ao longo da linha central (Y = 0), com máximos e mínimos marcados."""
    x_mm = padrao.x * 1e3
    perfil = _acima_do_piso(padrao.perfil_central(), escala_log)

    eixo.plot(x_mm, perfil, color=COR_SIMULACAO, linewidth=2, label="Simulação (soma de fasores)")
    if curva_teorica is not None:
        eixo.plot(
            x_mm,
            _acima_do_piso(curva_teorica, escala_log),
            color=TINTA_SECUNDARIA,
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            label="Teoria: cos²(πd·senθ/λ) · sinc²(πa·senθ/λ)",
        )

    estilo_marcador = dict(
        linestyle="none",
        markersize=8,
        markeredgecolor=SUPERFICIE,
        markeredgewidth=1.5,
        clip_on=False,
        zorder=3,
    )
    eixo.plot(
        x_mm[maximos],
        perfil[maximos],
        marker="^",
        color=COR_MAXIMOS,
        label="Máximos (interferência construtiva)",
        **estilo_marcador,
    )
    eixo.plot(
        x_mm[minimos],
        perfil[minimos],
        marker="v",
        color=COR_MINIMOS,
        label="Mínimos (interferência destrutiva)",
        **estilo_marcador,
    )

    eixo.set_xlim(x_mm[0], x_mm[-1])
    if escala_log:
        eixo.set_yscale("log")
        eixo.set_ylim(INTENSIDADE_MINIMA_LOG, 1.5)
    else:
        eixo.set_ylim(0, 1.05)
    eixo.grid(True)
    eixo.set_title("Perfil de intensidade na linha central do anteparo (Y = 0)")
    eixo.set_xlabel("X (mm)")
    eixo.set_ylabel("I/I₀")
    # Legenda abaixo do gráfico, para não cobrir as franjas
    colunas = 2 if curva_teorica is not None else 3
    eixo.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=colunas)


def _acima_do_piso(intensidade: np.ndarray, escala_log: bool) -> np.ndarray:
    """Na escala log, desenha no piso os valores menores que ele (log de 0 não existe)."""
    return np.maximum(intensidade, INTENSIDADE_MINIMA_LOG) if escala_log else intensidade


def _sem_moldura(eixo: Axes) -> None:
    """Remove as bordas do painel (a própria imagem já delimita a área)."""
    for borda in eixo.spines.values():
        borda.set_visible(False)
