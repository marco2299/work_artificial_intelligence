# Projeto 2 - K-Means Clustering

Este projeto implementa o algoritmo de agrupamento **K-Means** (aprendizado não supervisionado) para análise e visualização de dados. O código compara uma implementação manual do K-Means (`KMeansHardcore`) com a implementação da biblioteca `scikit-learn`. O objetivo é compreender como os grupos são formados e como a escolha de **K** influencia o resultado.

**Autor:** Marco Antonio Maia  
**Disciplina:** GCC128 - Inteligência Artificial (UFLA)

## Estrutura do Projeto

- **`kmeans_hardcore.py`**: Implementação manual (hardcore) do algoritmo K-Means, com cálculo de inércia (WCSS).
- **`kmeans_sklearn.py`**: Implementação do K-Means utilizando a biblioteca `scikit-learn`.
- **`metrics.py`**: Funções para Silhouette Score, WCSS/Elbow, matrizes de confusão e curvas de avaliação.
- **`pca_plot.py`**: Visualização dos clusters com redução de dimensionalidade (PCA).
- **`iris.csv`**: Conjunto de dados Iris (opcional; o `main.py` carrega via `scikit-learn`).
- **`main.py`**: Arquivo principal que executa o projeto.
- **`gerar_relatorio.py`**: Script auxiliar para (re)gerar o relatório em PDF.
- **`saidas/`**: Figuras geradas na execução (Elbow, Silhouette, PCA, matrizes de confusão).
- **`relatorio_tecnico_kmeans.pdf`**: Relatório com resultados, análise e conclusão.
- **`link_da_apresentacao`**: Link do vídeo de apresentação.

## Funcionalidades

1. **Agrupamento com K-Means Manual (`kmeans_hardcore.py`)**:
   - Distância euclidiana e atualização iterativa dos centróides.
   - Suporte a diferentes valores de `K`.
   - Calcula **WCSS (inércia)** e **Silhouette Score**.

2. **Agrupamento com `scikit-learn` (`kmeans_sklearn.py`)**:
   - Usa `KMeans` com `n_init=10` e `random_state` fixo.
   - Retorna rótulos, centróides e inércia.

3. **Avaliação dos agrupamentos**:
   - **Inércia / WCSS** para vários valores de `K`.
   - **Silhouette Score**.
   - **Método do Cotovelo (Elbow Method)**.
   - Comparação com/sem **StandardScaler** (padronização).
   - Matriz de confusão apenas para **análise a posteriori** (rótulos reais não entram no algoritmo).

4. **Visualização**:
   - PCA em 2D e 1D para representar os clusters (Iris tem 4 atributos).

## Requisitos

```bash
pip install -r requirements.txt
```

## Como Executar

Na pasta do projeto:

```bash
python main.py
```

Em ambiente sem interface gráfica:

```bash
MPLBACKEND=Agg python main.py
```

Para regenerar o relatório PDF:

```bash
python gerar_relatorio.py
```

## Resultados

A execução imprime no console e salva em `saidas/`:

- Curvas do **Elbow Method** (WCSS vs K)
- Curvas de **Silhouette Score** vs K
- Matrizes de confusão (comparação com espécies reais)
- Gráficos **PCA** dos clusters

### Exemplo de saída no console

```plaintext
===== Efeito da normalizacao (StandardScaler) =====
  [sem_normalizacao] WCSS=... | Silhouette=...
  [com_normalizacao] WCSS=... | Silhouette=...

===== Metodo do Cotovelo e Silhouette (padronizado) =====
  K | WCSS-SK | Sil-SK | WCSS-HC | Sil-HC
  ...
```

## Conjunto de Dados

O conjunto Iris contém 150 amostras e 3 espécies:

- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

Atributos: comprimento/largura da sépala e da pétala (cm).

## Observações

- Os rótulos das espécies **não** são usados como entrada do K-Means.
- A padronização (`StandardScaler`) é aplicada porque a distância euclidiana é sensível à escala.
- O intervalo de `K` usado no Elbow/Silhouette é `2..8`; a avaliação detalhada usa `K = {2,3,4,5}`.

## 🔗 Link para o vídeo de explicação

[https://youtu.be/qumcng2L9kY](https://youtu.be/qumcng2L9kY)

---

Projeto desenvolvido por **Marco Antonio Maia** para a disciplina Inteligência Artificial, Universidade Federal de Lavras (UFLA).