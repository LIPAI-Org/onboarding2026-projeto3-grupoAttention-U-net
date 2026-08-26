# 🧬 Projeto 3 — Segmentação Semântica em Imagens Histológicas

<p align="center">
  <strong>Segmentação semântica de imagens histológicas utilizando U-Net e Attention U-Net</strong>
</p>

<p align="center">
  <em>Projeto de Visão Computacional — LIPAI</em>
</p>

<p align="center">
  <a href="#sobre-o-projeto">Sobre</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#arquiteturas">Arquiteturas</a> •
  <a href="#metodologia">Metodologia</a> •
  <a href="#experimentos">Experimentos</a> •
  <a href="#resultados">Resultados</a> •
  <a href="#execução">Execução</a> •
  <a href="#estrutura-do-projeto">Estrutura</a>
</p>

---

## 📌 Sobre o projeto

Este projeto apresenta um pipeline completo de **segmentação semântica binária de imagens histológicas**, desenvolvido como parte do Projeto 3 do onboarding de 2026 do LIPAI.

O objetivo é investigar e comparar diferentes configurações de modelos de segmentação, considerando:

* arquitetura da rede;
* estratégia de inicialização e treinamento;
* função de perda;
* utilização de *data augmentation*;
* diferentes *seeds*;
* desempenho em diferentes conjuntos de dados;
* complexidade computacional das arquiteturas.

O pipeline contempla desde a organização dos datasets e geração dos *splits* até treinamento, validação, avaliação, geração de métricas, curvas de aprendizado, gráficos comparativos e resultados qualitativos.

De acordo com a especificação do projeto, o desenho experimental completo resulta em **72 execuções**, considerando todas as combinações de arquitetura/modo de treinamento, dataset, função de perda, *augmentation* e *seed*.

---

## 👨‍💻 Integrantes

| Integrante                       |
| -------------------------------- |
| **Gabriel dos Santos do Amaral** |
| **João Geiger Piza**             |

---

# 🎯 Objetivos

O projeto tem como principais objetivos:

* desenvolver um pipeline completo de segmentação semântica;
* comparar diferentes arquiteturas de segmentação;
* avaliar treinamento *From Scratch*;
* avaliar o uso de encoder pré-treinado com *fine-tuning* completo;
* comparar as funções de perda **BCE** e **Dice Loss**;
* investigar o impacto do *data augmentation*;
* executar múltiplas repetições utilizando diferentes *seeds*;
* avaliar os modelos no conjunto de teste;
* gerar curvas de aprendizado;
* gerar gráficos comparativos;
* produzir resultados qualitativos por meio de mosaicos;
* analisar a quantidade de parâmetros e os **GFLOPs** das arquiteturas.

---

# 🧪 Datasets

O projeto trabalha com dois conjuntos de dados histológicos, conforme especificado.

## 1. OralEpitheliumDB — Displasia

O primeiro dataset é baseado no **OralEpitheliumDB — Displasia**.

🔗 Repositório oficial:
https://github.com/LIPAI-Org/OralEpitheliumDB_Dataset

### Tarefa

Segmentação binária de **núcleos em imagens de epitélio oral**.

### Classes

| Classe | Descrição |
| ------ | --------- |
| `0`    | Fundo     |
| `1`    | Núcleo    |

Todas as categorias de núcleos presentes nas máscaras são agrupadas em uma única classe de primeiro plano.

Portanto, a tarefa final é:

> **Núcleo × Fundo**

---

## 2. Dataset de Tecido Tumoral

O segundo dataset é destinado à segmentação de regiões tumorais.

🔗 Dataset:
https://data.mendeley.com/datasets/9bsc36jyrt/1

São utilizadas as pastas:

* `ROI`
* `Mask`

Cada imagem da pasta `ROI` deve possuir sua máscara correspondente na pasta `Mask`.

### Classes

| Classe | Descrição |
| ------ | --------- |
| `0`    | Não tumor |
| `1`    | Tumor     |

A tarefa final é:

> **Tumor × Não tumor**

No código, esse dataset é identificado como **`HE`**.

---

# 📂 Organização dos dados

Os dados devem ser organizados dentro de `data/` da seguinte maneira:

```text
data/
├── mascaras/
│   ├── HE/
│   └── OEDB/
│
├── raw/
│   ├── HE/
│   └── OEDB/
│
└── splits/
    ├── HE/
    │   ├── treino/
    │   ├── val/
    │   └── teste/
    │
    └── OEDB/
        ├── treino/
        ├── val/
        └── teste/
```

Os arquivos de imagem e suas máscaras devem possuir nomes correspondentes para que o *dataset loader* consiga realizar o pareamento automaticamente.

---

# ✂️ Splits

Quando não existe uma divisão oficial do dataset, é utilizada a divisão:

| Conjunto  | Percentual |
| --------- | ---------: |
| Treino    |    **56%** |
| Validação |    **14%** |
| Teste     |    **30%** |

A divisão é gerada utilizando **seed 42** e deve ser criada uma única vez, sendo posteriormente reutilizada em todos os experimentos.

Isso garante que as diferentes configurações sejam comparadas utilizando exatamente os mesmos conjuntos de treino, validação e teste.

### Geração dos splits

```bash
python scripts/criar_splits.py
```

---

# 🧠 Arquiteturas

Foram utilizadas duas famílias de arquitetura:

1. **Attention U-Net**
2. **U-Net com encoder ResNet34**

---

## Attention U-Net

A **Attention U-Net** é a arquitetura específica escolhida pelo grupo.

Sua implementação foi realizada diretamente em PyTorch e utiliza mecanismos de atenção nos *skip connections* para filtrar e destacar regiões relevantes durante a reconstrução da máscara.

A implementação possui:

* blocos convolucionais;
* encoder;
* *bottleneck*;
* decoder;
* *skip connections*;
* portões de atenção (*Attention Gates*);
* convoluções transpostas;
* camada final de segmentação.

A arquitetura utiliza os seguintes canais:

```text
64 → 128 → 256 → 512 → 1024
```

A saída possui **1 canal**, adequado para a segmentação binária.

A arquitetura escolhida corresponde à Attention U-Net apresentada na especificação do projeto.

---

## U-Net com ResNet34

A segunda arquitetura utiliza uma U-Net implementada através da biblioteca **Segmentation Models PyTorch (SMP)**.

O encoder utilizado é:

```text
ResNet34
```

São avaliadas duas formas de inicialização:

### U-Net — From Scratch

```text
Encoder: ResNet34
Pesos: aleatórios
Modo: FS
```

### U-Net — Pré-treinada

```text
Encoder: ResNet34
Pesos: ImageNet
Modo: PT-ALL
```

No modo **PT-ALL**, todos os componentes permanecem treináveis durante o treinamento, incluindo o encoder, decoder e cabeça de segmentação.

---

# ⚙️ Configurações experimentais

O projeto avalia três combinações de arquitetura e modo de treinamento:

| Configuração | Arquitetura      | Modo   |
| ------------ | ---------------- | ------ |
| 1            | Attention U-Net  | FS     |
| 2            | U-Net + ResNet34 | FS     |
| 3            | U-Net + ResNet34 | PT-ALL |

O desenho experimental também considera:

* 2 datasets;
* 2 funções de perda;
* 2 condições de *augmentation*;
* 3 *seeds*.

Assim:

```text
3 arquiteturas/modos
× 2 losses
× 2 augmentations
× 2 datasets
× 3 seeds
= 72 execuções
```

As *seeds* utilizadas são:

```text
42
123
2025
```

---

# 📐 Pré-processamento

## Tamanho de entrada

O tamanho padrão utilizado é:

```text
256 × 256 pixels
```

Esse valor é mantido entre as diferentes configurações experimentais.

## Normalização

É utilizada a normalização padrão do ImageNet:

```python
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

A mesma normalização é aplicada aos diferentes modos de treinamento.

---

# 🔄 Data Augmentation

São avaliadas duas condições:

### Sem augmentation

São utilizadas apenas as transformações básicas:

* redimensionamento para `256 × 256`;
* normalização;
* conversão para tensor.

### Com augmentation

São utilizadas três operações geométricas:

* `HorizontalFlip`;
* `VerticalFlip`;
* `RandomRotate90`.

As transformações são aplicadas de forma sincronizada entre imagem e máscara, preservando o alinhamento espacial.

A implementação utiliza a biblioteca **Albumentations**.

A escolha dessas operações segue a recomendação do projeto de utilizar um número reduzido de transformações principais para facilitar a análise experimental.

### Verificação visual

O projeto também possui um script específico para verificar visualmente as transformações:

```bash
python scripts/testar_aumentos.py
```

---

# 🧩 Patching / Tiling

O pipeline utiliza entradas de tamanho fixo de:

```text
256 × 256
```

O processo de preparação dos dados redimensiona as imagens para esse tamanho antes de sua entrada na rede.

---

# 📉 Funções de perda

São avaliadas duas funções de perda separadamente.

## Binary Cross-Entropy

É utilizada:

```python
torch.nn.BCEWithLogitsLoss
```

Os *logits* são fornecidos diretamente à função de perda, sem aplicação prévia de `sigmoid`.

## Dice Loss

A Dice Loss é implementada a partir do coeficiente Dice.

A função é aplicada após a conversão dos *logits* em probabilidades utilizando `sigmoid`.

Não é utilizada uma combinação BCE + Dice nos experimentos.

---

# 🏋️ Treinamento

As configurações básicas utilizadas pelo projeto são:

| Parâmetro             | Valor                   |
| --------------------- | ----------------------- |
| Entrada               | `256 × 256`             |
| Batch size            | `8`                     |
| Épocas máximas        | `50`                    |
| Otimizador            | `Adam`                  |
| Learning rate         | `1e-3`                  |
| Scheduler             | `StepLR`                |
| Step size             | `10`                    |
| Gamma                 | `0.1`                   |
| Limiar de segmentação | `0.5`                   |
| Dispositivo           | CUDA, quando disponível |

O melhor modelo é selecionado com base no **maior mDice no conjunto de validação**, sem utilizar o conjunto de teste para escolha da época ou configuração.

---

# 📊 Métricas

Para cada execução são calculadas métricas no conjunto de teste.

### Métricas utilizadas

* Dice por classe;
* mDice;
* IoU por classe;
* mIoU;
* Precision da classe de interesse;
* Recall da classe de interesse.

Para o dataset de displasia, a classe de interesse é:

```text
Núcleo
```

Para o dataset de tecido tumoral:

```text
Tumor
```

Para cada configuração são realizadas três repetições e calculados:

* média;
* desvio padrão.

As principais métricas consolidadas são:

* mDice;
* mIoU;
* Dice da classe de interesse;
* IoU da classe de interesse;
* Precision da classe de interesse;
* Recall da classe de interesse.

---

# 📈 Curvas de aprendizado

Para cada execução são geradas curvas contendo:

* *training loss*;
* *validation loss*;
* mDice de validação.

Essas curvas permitem analisar:

* convergência;
* instabilidade;
* overfitting;
* underfitting;
* diferenças entre BCE e Dice;
* impacto dos pesos pré-treinados.

Os arquivos são organizados em:

```text
results/plots/curvas_aprendizado/
├── loss/
└── mdice/
```

---

# 🖼️ Resultados qualitativos

O projeto gera mosaicos para inspeção visual das segmentações.

Cada amostra apresenta:

1. imagem original;
2. Ground Truth;
3. predição da U-Net;
4. predição da Attention U-Net.

O script utiliza uma seleção aleatória reprodutível com **seed 42** e, por padrão, seleciona até **6 imagens por dataset**.

Para gerar os mosaicos:

```bash
python scripts/criar_mosaicos.py
```

Também é possível alterar a quantidade de imagens:

```bash
python scripts/criar_mosaicos.py --quantidade 8
```

Ou definir outra seed para a seleção:

```bash
python scripts/criar_mosaicos.py --seed 123
```

A especificação do projeto recomenda entre seis e oito amostras por dataset, incluindo casos bons, falhas, subsegmentação, sobresegmentação e regiões difíceis.

---

# 📊 Comparativos globais

O projeto gera gráficos comparativos para análise global dos experimentos.

São considerados:

* mDice;
* mIoU;
* número de parâmetros;
* GFLOPs.

Também é produzido um CSV consolidado com os resultados das configurações experimentais.

Para gerar os gráficos:

```bash
python scripts/criar_graficos_globais.py
```

Os resultados são armazenados em:

```text
results/plots/graficos_globais/
├── mdice/
├── miou/
└── gflops_e_parametros/
```

A especificação exige que os gráficos de mDice apresentem a média das três *seeds* e o desvio padrão correspondente.

---

# 📋 Resultados

Os resultados individuais das execuções são armazenados em:

```text
results/metrics/resultados_completos.csv
```

O arquivo contém **72 execuções**, com informações como:

* dataset;
* modelo;
* encoder;
* modo de treinamento;
* augmentation;
* loss;
* seed;
* tamanho de entrada;
* épocas;
* batch size;
* Dice;
* IoU;
* Precision;
* Recall;
* número de parâmetros;
* parâmetros treináveis;
* GFLOPs;
* melhor época;
* mDice de validação.

O resultado consolidado é armazenado em:

```text
results/metrics/resultados_consolidados.csv
```

Esse arquivo apresenta a média e o desvio padrão das três repetições.

A estrutura segue as exigências do CSV consolidado especificadas.

---

# 🏆 Resultados obtidos

Os resultados consolidados presentes no repositório permitem identificar o comportamento das diferentes configurações.

Entre as configurações avaliadas no CSV consolidado, as maiores médias de **mDice** observadas foram:

### Dataset HE

**U-Net + ResNet34 — PT-ALL — sem augmentation — BCE**

```text
mDice: 0.903157 ± 0.001806
mIoU:  0.835776 ± 0.001915
```

### Dataset OEDB

**U-Net + ResNet34 — PT-ALL — com augmentation — Dice**

```text
mDice: 0.852472 ± 0.000736
mIoU:  0.754039 ± 0.001268
```

Esses valores são apresentados aqui como **resultados presentes nos arquivos CSV do projeto**, não como uma conclusão geral sobre a superioridade de uma arquitetura em todos os cenários.

---

# 💻 Execução

## 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

## 2. Criar ambiente virtual

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Instalar dependências

O projeto utiliza principalmente:

```text
Python
PyTorch
NumPy
Pandas
Matplotlib
Pillow
Albumentations
Segmentation Models PyTorch
```

As versões específicas devem ser definidas de acordo com o ambiente utilizado para os experimentos.

## 4. Instalar os datasets

O link para ambos está na seção <a href="#datasets">Datasets</a>

## 5. "Limpar" os datasets

Os datasets são organizados em várias pastas. Visando a simplicidade, as imagens devem ser colocadas em:
```text
data/raw/HE
data/raw/OEDB
```

---

# 🚀 Pipeline completo

Após configurar os datasets, o pipeline pode ser executado seguindo as etapas abaixo.

### 1. Criar os splits

```bash
python scripts/criar_splits.py
```

### 2. Verificar as augmentations

```bash
python scripts/testar_aumentos.py
```

### 3. Executar todos os treinamentos

```bash
python scripts/treinar_todos.py
```

Esse script executa todas as combinações experimentais e, ao final, realiza a consolidação dos resultados.

### 4. Consolidar resultados

Caso seja necessário executar essa etapa separadamente:

```bash
python scripts/consolidar_resultados.py
```

### 5. Gerar gráficos globais

```bash
python scripts/criar_graficos_globais.py
```

### 6. Gerar mosaicos qualitativos

```bash
python scripts/criar_mosaicos.py
```

---

# 🗂️ Estrutura do projeto

```text
.
├── configs/
│   ├── basicas.py
│   └── experimentos.py
│
├── data/
│   ├── mascaras/
│   │   ├── HE/
│   │   └── OEDB/
│   │
│   ├── raw/
│   │   ├── HE/
│   │   └── OEDB/
│   │
│   └── splits/
│       ├── HE/
│       │   ├── teste/
│       │   ├── treino/
│       │   └── val/
│       │
│       └── OEDB/
│           ├── teste/
│           ├── treino/
│           └── val/
│
├── results/
│   ├── metrics/
│   │   ├── resultados_completos.csv
│   │   └── resultados_consolidados.csv
│   │
│   ├── modelos/
│   │   └── checkpoints/
│   │
│   ├── plots/
│   │   ├── curvas_aprendizado/
│   │   │   ├── loss/
│   │   │   └── mdice/
│   │   │
│   │   └── graficos_globais/
│   │       ├── mdice/
│   │       └── miou/
│   │
│   └── qualitative/
│
├── scripts/
│   ├── consolidar_resultados.py
│   ├── criar_graficos_globais.py
│   ├── criar_mosaicos.py
│   ├── criar_splits.py
│   ├── testar_aumentos.py
│   └── treinar_todos.py
│
├── src/
│   ├── avaliacao/
│   │   ├── avaliador.py
│   │   └── metricas.py
│   │
│   ├── data/
│   │   ├── dataloader.py
│   │   ├── dataset.py
│   │   └── transformadas.py
│   │
│   ├── losses/
│   │   ├── dice_loss.py
│   │   └── loss_factory.py
│   │
│   ├── modelos/
│   │   ├── attention_unet.py
│   │   ├── modelo_factory.py
│   │   └── unet.py
│   │
│   ├── treino/
│   │   ├── curvas_treino.py
│   │   ├── treinamento.py
│   │   └── validacao.py
│   │
│   └── utils/
│       ├── checkpoints.py
│       ├── experimentos.py
│       ├── graficos_globais.py
│       ├── paths.py
│       ├── rodar_experimentos.py
│       ├── seed.py
│       └── tabela_resultado.py
│
├── .gitignore
├── main.py
└── README.md
```

---

# 🧩 Organização do código

### `configs/`

Centraliza as configurações do projeto.

* `basicas.py` — hiperparâmetros gerais;
* `experimentos.py` — definição das combinações experimentais.

### `src/data/`

Responsável pelo carregamento e processamento dos datasets.

* `dataset.py` — implementação do dataset;
* `dataloader.py` — criação dos DataLoaders;
* `transformadas.py` — normalização e *data augmentation*.

### `src/modelos/`

Implementação das arquiteturas.

* `unet.py` — U-Net com ResNet34;
* `attention_unet.py` — Attention U-Net;
* `modelo_factory.py` — seleção das arquiteturas.

### `src/losses/`

Implementação e seleção das funções de perda.

### `src/treino/`

Contém a lógica de:

* treinamento;
* validação;
* curvas de aprendizado.

### `src/avaliacao/`

Responsável pelas métricas e avaliação dos modelos.

### `src/utils/`

Funções auxiliares para:

* checkpoints;
* seeds;
* caminhos;
* experimentos;
* geração de gráficos;
* consolidação de resultados.

### `scripts/`

Contém os pontos de entrada para executar as principais etapas do pipeline.

---

# 🔬 Reprodutibilidade

A reprodutibilidade dos experimentos é controlada por meio das *seeds*:

```text
42
123
2025
```

Os mesmos *splits* são utilizados em todas as configurações, evitando que diferenças na divisão dos dados interfiram na comparação dos modelos.

O projeto também registra as configurações e métricas das execuções no CSV de resultados.

Essa abordagem segue as boas práticas estabelecidas na especificação do projeto, incluindo controle da aleatoriedade, reutilização dos mesmos *splits* e separação entre validação e teste.

---

# 📚 Referências

### Attention U-Net

O projeto utiliza como referência a arquitetura **Attention U-Net**:

> Oktay et al. — *Attention U-Net: Learning Where to Look for the Pancreas*

https://arxiv.org/abs/1804.03999

### Segmentation Models PyTorch

A U-Net com encoder ResNet34 utiliza a biblioteca:

https://segmentation-modelspytorch.readthedocs.io/

### OralEpitheliumDB

https://github.com/LIPAI-Org/OralEpitheliumDB_Dataset

### Dataset de tecido tumoral

https://data.mendeley.com/datasets/9bsc36jyrt/1

---

# 📖 Especificação do projeto

O projeto foi desenvolvido seguindo as orientações de um documento passado:

**Projeto 3 — Segmentação Semântica em Imagens Histológicas**

Entre os requisitos definidos estão:

* utilização dos dois datasets;
* divisão treino/validação/teste;
* comparação entre arquiteturas;
* comparação entre BCE e Dice;
* avaliação de *data augmentation*;
* três *seeds*;
* 72 execuções;
* avaliação quantitativa;
* resultados qualitativos;
* análise de parâmetros e GFLOPs;
* consolidação dos resultados.

---

# 📌 Considerações finais

Este projeto implementa um pipeline experimental completo para segmentação semântica de imagens histológicas, permitindo avaliar sistematicamente diferentes estratégias de treinamento e suas influências sobre o desempenho dos modelos.

A organização modular do projeto facilita a reprodução dos experimentos, a comparação entre configurações e a extensão futura do pipeline para novas arquiteturas, funções de perda, estratégias de *augmentation* e *backbones*.

---

<p align="center">
  Desenvolvido por <strong>Gabriel dos Santos do Amaral</strong> e <strong>João Geiger Piza</strong>
</p>

<p align="center">
  <em>Projeto 3 — Segmentação Semântica em Imagens Histológicas</em>
</p>
