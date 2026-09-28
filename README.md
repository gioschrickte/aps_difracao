# Simulador de difração de Fraunhofer

Programa em Python que simula a imagem de difração de uma fenda (ou de qualquer
objeto desenhado como imagem) usando a soma de fasores. Você escolhe o objeto,
o comprimento de onda da luz e a distância até o anteparo, e o programa mostra
o padrão de luz e sombra que apareceria nele.

## 1. Instalar

Precisa do Python 3.10 ou mais novo. Na pasta do projeto, rode:

```bash
pip install -r requirements.txt
```

## 2. Rodar

O jeito mais simples, sem nenhuma opção, já simula uma fenda dupla:

```bash
python main.py
```

Isso abre uma janela com o resultado e também salva a imagem em
`resultado_difracao.png`.

## 3. Usando sua própria imagem

Você pode desenhar seu próprio objeto (fenda simples, fenda dupla, um furo
redondo, uma letra, o que quiser) em qualquer programa de imagem e salvar como
PNG. A regra é:

- **branco** = a luz passa por ali;
- **preto** = a luz é bloqueada.

Depois é só passar o arquivo para o programa e dizer qual é o tamanho real da
imagem (em milímetros), com `--largura-objeto`:

```bash
python main.py minha_fenda.png --largura-objeto 1.5
```

Se o seu desenho for o contrário (fenda em preto sobre fundo branco), use
`--inverter`.

Também tem alguns exemplos prontos na pasta `exemplos/` para testar:

```bash
python main.py exemplos/abertura_circular.png --log
python main.py exemplos/fendas_multiplas.png
```

## 4. Ajustando o experimento

As opções mais usadas:

| Opção | O que faz | Padrão |
|---|---|---|
| `--lambda` | comprimento de onda da luz, em nanômetros | 632.8 (laser vermelho He-Ne) |
| `--distancia` | distância entre o objeto e o anteparo, em metros | 1.0 |
| `--largura-objeto` | largura real da imagem usada como objeto, em mm | 1.0 |
| `--log` | mostra a imagem em escala logarítmica (realça os pontos fracos de luz) | desligado |
| `--sem-janela` | só salva a figura, sem abrir a janela (útil para rodar várias vezes seguidas) | desligado |

Exemplo com laser verde e o anteparo mais longe:

```bash
python main.py --lambda 532 --distancia 2
```

Para ver a lista completa de opções (inclusive as de criar a fenda dupla do
zero, sem precisar de imagem), rode:

```bash
python main.py --help
```

## 5. Lendo o resultado

A figura gerada tem três partes:

1. **Objeto** — a imagem que você deu de entrada, em preto e branco.
2. **Padrão no anteparo** — o mapa de intensidade da luz. Quanto mais claro,
   mais luz chega naquele ponto; o preto são as regiões de interferência
   destrutiva (sem luz nenhuma).
3. **Perfil de intensidade** — um corte horizontal passando pelo centro,
   com os máximos de luz marcados com ▲ e os mínimos com ▼.

No terminal, o programa também imprime um resumo dos parâmetros usados e uma
tabela com a posição (em mm) de cada máximo e mínimo perto do centro.

## Estrutura dos arquivos

- `main.py` — lê as opções de linha de comando e roda a simulação.
- `aberturas.py` — carrega o objeto (PNG, matriz de texto, ou fenda dupla gerada).
- `difracao.py` — o cálculo físico da difração (soma de fasores).
- `visualizacao.py` — monta a figura do resultado.
- `exemplos/` — algumas imagens de objetos prontas para testar.
