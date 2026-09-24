import os

import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler

from knn import run_knn
from metrics import calculate_metrics, confusion_matrix, plot_confusion_matrix
from mlp import run_mlp

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "saidas")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_classification_file(filepath, results):
    """Salva os resultados de classificacao em um arquivo de texto."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            for algorithm_name, ids, y_true, y_pred, target_names in results:
                f.write(f"###--- {algorithm_name} ---###\n\n")
                f.write(" Id".ljust(8) + "Species".ljust(18) + "Predicted_Species\n")
                for idx, real, pred in zip(ids, y_true, y_pred):
                    f.write(
                        f"{str(idx).rjust(3)}  "
                        f"{target_names[real].ljust(16)} "
                        f"{target_names[pred]}\n"
                    )
                f.write("\n" + "-" * 50 + "\n\n")
                acc, prec, rec = calculate_metrics(y_true, y_pred)
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"Precision: {prec:.4f}\n")
                f.write(f"Recall: {rec:.4f}\n\n")
                f.write("-" * 50 + "\n\n")
                cm = confusion_matrix(y_true, y_pred)
                f.write("".ljust(20))
                for name in target_names:
                    f.write(str(name).ljust(18))
                f.write("\n")
                for i, row in enumerate(cm):
                    f.write(str(target_names[i]).ljust(20))
                    for val in row:
                        f.write(str(val).ljust(18))
                    f.write("\n")
                f.write("\n")
    except Exception as e:
        print(f"Erro ao salvar o arquivo {filepath}: {e}")


def print_and_plot(name, dataset_name, y_true, y_pred, target_names):
    acc, prec, rec = calculate_metrics(y_true, y_pred)
    print(f"[{name} / {dataset_name}] Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")
    plot_confusion_matrix(
        y_true,
        y_pred,
        labels=list(target_names),
        title=f"{name} - {dataset_name}",
        save_path=os.path.join(
            OUTPUT_DIR, f"confusion_{name.lower()}_{dataset_name.lower()}.png"
        ),
    )
    return acc, prec, rec


if __name__ == "__main__":
    summary = []

    # Dataset Iris
    iris = load_iris()
    X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    y_iris = pd.Series(iris.target)
    iris_target_names = iris.target_names

    print("\n=== Executando KNN no Dataset Iris (referencia TP01) ===")
    idx_iris_knn, y_test_iris_knn, y_pred_iris_knn, _ = run_knn(
        X_iris, y_iris, k=5, dataset_name="Iris"
    )
    summary.append(
        ("KNN", "Iris", *print_and_plot("KNN", "Iris", y_test_iris_knn, y_pred_iris_knn, iris_target_names))
    )

    print("\n=== Executando MLPClassifier no Dataset Iris ===")
    scaler_iris = StandardScaler()
    X_iris_scaled = scaler_iris.fit_transform(X_iris)
    idx_iris_mlp, y_test_iris_mlp, y_pred_iris_mlp, _ = run_mlp(
        X_iris_scaled,
        y_iris,
        hidden_layer_sizes=(100,),
        max_iter=500,
        dataset_name="Iris",
    )
    summary.append(
        ("MLP", "Iris", *print_and_plot("MLP", "Iris", y_test_iris_mlp, y_pred_iris_mlp, iris_target_names))
    )

    # Dataset Wine
    wine = load_wine()
    X_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
    y_wine = pd.Series(wine.target)
    wine_target_names = wine.target_names

    print("\n=== Executando KNN no Dataset Wine (referencia TP01) ===")
    idx_wine_knn, y_test_wine_knn, y_pred_wine_knn, _ = run_knn(
        X_wine, y_wine, k=5, dataset_name="Wine"
    )
    summary.append(
        ("KNN", "Wine", *print_and_plot("KNN", "Wine", y_test_wine_knn, y_pred_wine_knn, wine_target_names))
    )

    print("\n=== Executando MLPClassifier no Dataset Wine ===")
    scaler_wine = StandardScaler()
    X_wine_scaled = scaler_wine.fit_transform(X_wine)
    idx_wine_mlp, y_test_wine_mlp, y_pred_wine_mlp, _ = run_mlp(
        X_wine_scaled,
        y_wine,
        hidden_layer_sizes=(100,),
        max_iter=500,
        dataset_name="Wine",
    )
    summary.append(
        ("MLP", "Wine", *print_and_plot("MLP", "Wine", y_test_wine_mlp, y_pred_wine_mlp, wine_target_names))
    )

    project_dir = os.path.dirname(__file__)

    iris_results = [
        ("KNN", idx_iris_knn, y_test_iris_knn, y_pred_iris_knn, iris_target_names),
        ("MLP", idx_iris_mlp, y_test_iris_mlp, y_pred_iris_mlp, iris_target_names),
    ]
    write_classification_file(os.path.join(project_dir, "classificacao_iris.txt"), iris_results)

    wine_results = [
        ("KNN", idx_wine_knn, y_test_wine_knn, y_pred_wine_knn, wine_target_names),
        ("MLP", idx_wine_mlp, y_test_wine_mlp, y_pred_wine_mlp, wine_target_names),
    ]
    write_classification_file(os.path.join(project_dir, "classificacao_wine.txt"), wine_results)

    print("\n===== Comparacao KNN (TP01) vs MLPClassifier =====")
    print(f"{'Alg':<6} {'Dataset':<8} {'Acc':>8} {'Prec':>8} {'Rec':>8}")
    for alg, ds, acc, prec, rec in summary:
        print(f"{alg:<6} {ds:<8} {acc:8.4f} {prec:8.4f} {rec:8.4f}")

    print(f"\nArquivos de classificacao salvos em: {project_dir}/")
    print(f"Matrizes de confusao (PNG) salvas em: {OUTPUT_DIR}/")
