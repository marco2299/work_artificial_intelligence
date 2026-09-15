"""Gera o relatorio tecnico PDF do Projeto 2 (K-Means)."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(
    TTFont("DejaVuBold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
)


def build_report(path="relatorio_tecnico_kmeans.pdf"):
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleBR",
        parent=styles["Heading1"],
        fontName="DejaVuBold",
        fontSize=14,
        spaceAfter=8,
        alignment=1,
    )
    heading = ParagraphStyle(
        "HeadingBR",
        parent=styles["Heading2"],
        fontName="DejaVuBold",
        fontSize=12,
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "BodyBR",
        parent=styles["Normal"],
        fontName="DejaVu",
        fontSize=10,
        leading=14,
        alignment=4,
        spaceAfter=6,
    )
    meta = ParagraphStyle(
        "MetaBR",
        parent=styles["Normal"],
        fontName="DejaVu",
        fontSize=10,
        leading=13,
        spaceAfter=3,
    )

    story = []
    story.append(Paragraph("UNIVERSIDADE FEDERAL DE LAVRAS", title))
    story.append(Paragraph("DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO", meta))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("<b>Aluno:</b> Marco Antonio Maia", meta))
    story.append(Paragraph("<b>Matrícula:</b> 202220256", meta))
    story.append(Paragraph("<b>Disciplina:</b> GCC128 – Inteligência Artificial", meta))
    story.append(Paragraph("<b>Professor:</b> Ahmed Ali Abdalla Esmin", meta))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Relatório Técnico – K-Means", title))

    story.append(Paragraph("1. Base de dados e algoritmo", heading))
    story.append(
        Paragraph(
            "O conjunto Iris (Fisher, 1936) possui 150 amostras e quatro atributos "
            "contínuos (comprimento e largura da sépala e da pétala, em cm), "
            "distribuídos em três espécies: setosa, versicolor e virginica. "
            "A base é balanceada (50 amostras por classe) e não apresenta valores "
            "faltantes. Embora seja classicamente usada em classificação, neste "
            "trabalho os rótulos reais <b>não entram</b> no K-Means: servem apenas "
            "para análise/comparação a posteriori dos agrupamentos.",
            body,
        )
    )
    story.append(
        Paragraph(
            "O K-Means é um método de aprendizado não supervisionado que particiona "
            "os dados em K grupos. Inicializam-se K centróides; cada amostra é "
            "atribuída ao centróide mais próximo (distância euclidiana); os "
            "centróides são recalculados como a média do grupo. O processo se "
            "repete até convergência ou limite de iterações. Como a distância "
            "euclidiana é sensível à escala, os atributos foram padronizados com "
            "<b>StandardScaler</b>. Também foi executada uma comparação sem "
            "padronização para evidenciar o efeito da normalização.",
            body,
        )
    )

    story.append(Paragraph("2. Implementação e avaliação", heading))
    story.append(
        Paragraph(
            "Foram desenvolvidas duas abordagens: (i) <b>KMeansHardcore</b>, "
            "implementação manual com inicialização aleatória, distância "
            "euclidiana e atualização iterativa dos centróides, expondo também a "
            "inércia (WCSS); (ii) <b>scikit-learn KMeans</b>, com k-means++, "
            "n_init=10 e random_state fixo. Em ambas, os rótulos Iris não são "
            "entradas do algoritmo.",
            body,
        )
    )
    story.append(
        Paragraph(
            "A avaliação segue o enunciado de clustering: <b>Inércia/WCSS</b>, "
            "<b>Silhouette Score</b> e o <b>Método do Cotovelo (Elbow)</b> para "
            "K em {2,...,8}. Para K em {2,3,4,5} geram-se matrizes de confusão "
            "(apenas comparação com espécies reais) e visualização por <b>PCA</b> "
            "(1 e 2 componentes), adequada porque Iris tem quatro dimensões.",
            body,
        )
    )

    story.append(Paragraph("3. Resultados e análise", heading))
    story.append(
        Paragraph(
            "<b>Influência de K:</b> valores baixos de K produzem grupos mais "
            "grosseiros; valores altos reduzem o WCSS, mas podem fragmentar "
            "grupos naturais. No Iris, K=3 tende a refletir as três espécies, "
            "embora versicolor e virginica se sobreponham em parte do espaço. "
            "O Silhouette frequentemente favorece K=2 (setosa vs. demais), o que "
            "mostra que a métrica e o conhecimento do domínio devem ser lidos juntos.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Elbow e Silhouette:</b> a curva de WCSS decresce com K; o "
            "“cotovelo” indica o ponto em que o ganho marginal diminui (em geral "
            "próximo de K=3 no Iris). O Silhouette mede coesão e separação. "
            "Os dois critérios auxiliam a escolha de K sem usar rótulos como entrada.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Efeito da normalização:</b> sem padronização, atributos com maior "
            "variância (ex.: comprimento da pétala) dominam a distância. Com "
            "<b>StandardScaler</b>, cada atributo contribui de forma comparável, "
            "melhorando a interpretabilidade dos agrupamentos. Os valores absolutos "
            "de WCSS mudam com a escala; por isso a comparação deve ser feita "
            "dentro do mesmo pré-processamento.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Características dos grupos e limitações:</b> o K-Means assume "
            "clusters aproximadamente esféricos/convexos, é sensível à "
            "inicialização e exige escolher K. A versão hardcore é didática, "
            "porém menos otimizada; a do scikit-learn é mais rápida e "
            "consistente (k-means++). PCA facilita a visualização, mas comprime "
            "informação das quatro dimensões originais.",
            body,
        )
    )

    story.append(Paragraph("4. Conclusão", heading))
    story.append(
        Paragraph(
            "O projeto cumpre o uso de K-Means com múltiplos valores de K, "
            "padronização justificada, métricas de agrupamento (WCSS e "
            "Silhouette), Método do Cotovelo, visualização via PCA e análise "
            "dos resultados. A implementação manual ajuda a compreender o "
            "algoritmo; a biblioteca oferece desempenho e praticidade para "
            "experimentos. Link do vídeo de apresentação: "
            "https://youtu.be/",
            body,
        )
    )

    doc.build(story)
    print(f"Relatório gerado: {path}")


if __name__ == "__main__":
    build_report()
