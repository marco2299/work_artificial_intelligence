# Projeto 3 - Perceptron (MLPClassifier)

Este projeto utiliza o **MLPClassifier** do Scikit-learn para classificação supervisionada nas bases **Iris** e **Wine**, e compara o desempenho com o **KNN** do Trabalho #01 (implementação manual).

**Autor:** Marco Antonio Maia  
**Disciplina:** GCC128 - Inteligência Artificial (UFLA)

## Estrutura do Projeto

- **`mlp.py`**: Classificação com `MLPClassifier` (Scikit-learn).
- **`knn.py`**: Implementação manual do KNN (referência ao TP01).
- **`metrics.py`**: Métricas (acurácia, precisão, revocação) e plot da matriz de confusão.
- **`main.py`**: Executa KNN e MLP em Iris e Wine.
- **`saidas/`**: PNGs das matrizes de confusão geradas na execução.
- **`classificacao_iris.txt` / `classificacao_wine.txt`**: Resultados detalhados (gerados).
- **`relatorio_tecnico_perceptron.pdf`**: Relatório (até 1 página de análise + conclusão).
- **`link_da_apresentacao`**: Link do vídeo de apresentação.

## Funcionalidades

1. **MLPClassifier** em Iris e Wine (com `StandardScaler`).
2. **Matriz de confusão** plotada/salva e métricas: precisão, revocação e acurácia.
3. **Comparação** com o KNN do Trabalho #01 nos mesmos conjuntos e divisão treino/teste.

## Requisitos

```bash
pip install -r requirements.txt
```

## Como Executar

```bash
python main.py
```

Sem interface gráfica:

```bash
MPLBACKEND=Agg python main.py
```

Para regenerar o relatório PDF:

```bash
python gerar_relatorio.py
```

## Resultados

A execução imprime no console a comparação KNN vs MLP e gera:

- `classificacao_iris.txt`, `classificacao_wine.txt`
- `saidas/confusion_*.png`

## Conjuntos de Dados

- **Iris:** 150 amostras, 3 classes, 4 atributos.
- **Wine:** 178 amostras, 3 cultivares, 13 atributos químicos.

## Link do vídeo de apresentação

Veja [`link_da_apresentacao`](./link_da_apresentacao) ou:

[https://youtu.be/Q4XuWCt_vhE](https://youtu.be/Q4XuWCt_vhE)

---

Projeto desenvolvido por **Marco Antonio Maia** para a disciplina Inteligência Artificial, Universidade Federal de Lavras (UFLA).
