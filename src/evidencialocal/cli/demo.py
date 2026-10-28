"""Generador de datos de demostración sintéticos."""

import json
from pathlib import Path

from faker import Faker
from fpdf import FPDF

fake = Faker("es_ES")


DEMO_DOCUMENTS = [
    {
        "filename": "01_aprendizaje_supervisado.pdf",
        "title": "Fundamentos del Aprendizaje Supervisado",
        "content": """El aprendizaje supervisado es un paradigma de aprendizaje automático donde el algoritmo aprende
a partir de datos etiquetados. Cada ejemplo de entrenamiento consiste en una entrada (características)
y una salida esperada (etiqueta). El objetivo es aprender una función de mapeo de entradas a salidas
que generalice bien a datos no vistos.

Los principales tipos de aprendizaje supervisado son:
1. Clasificación: la salida es una categoría discreta (ej. spam/no spam, dígito 0-9).
2. Regresión: la salida es un valor continuo (ej. precio de casa, temperatura).

Algoritmos comunes incluyen: regresión lineal, regresión logística, árboles de decisión,
random forests, SVM, y redes neuronales. La evaluación se hace con métricas como accuracy,
precision, recall, F1-score para clasificación, y MSE, MAE, R² para regresión.

El overfitting ocurre cuando el modelo memoriza los datos de entrenamiento en lugar de generalizar.
Técnicas para prevenirlo: validación cruzada, regularización (L1/L2), early stopping, dropout.""",
    },
    {
        "filename": "02_redes_neuronales.pdf",
        "title": "Introducción a las Redes Neuronales",
        "content": """Las redes neuronales artificiales son modelos computacionales inspirados en el cerebro biológico.
Están compuestas por neuronas artificiales organizadas en capas: capa de entrada, capas ocultas,
y capa de salida. Cada neurona aplica una transformación lineal seguida de una función de activación
no lineal (ReLU, sigmoid, tanh).

El entrenamiento usa retropropagación (backpropagation) con gradient descent para minimizar
una función de pérdida. Hiperparámetros clave: learning rate, batch size, número de capas/neuronas,
función de activación, optimizador (SGD, Adam, RMSprop).

Arquitecturas importantes: MLP (perceptrón multicapa), CNN (redes convolucionales) para imágenes,
RNN/LSTM/Transformer para secuencias. El deep learning ha revolucionado visión por computador,
NLP, reconocimiento de voz, y más.""",
    },
    {
        "filename": "03_aprendizaje_no_supervisado.pdf",
        "title": "Aprendizaje No Supervisado y Reducción de Dimensionalidad",
        "content": """El aprendizaje no supervisado encuentra patrones en datos SIN etiquetas.
Tareas principales: clustering (agrupamiento), reducción de dimensionalidad, detección de anomalías.

Clustering: K-means, clustering jerárquico, DBSCAN. Métricas: inercia, silueta, índice de Davies-Bouldin.
Reducción de dimensionalidad: PCA (análisis de componentes principales), t-SNE, UMAP, autoencoders.
PCA proyecta datos a subespacio de menor dimensión maximizando varianza explicada.

Aplicaciones: segmentación de clientes, compresión de imágenes, visualización de datos de alta dimensión,
preprocesamiento para aprendizaje supervisado.""",
    },
    {
        "filename": "04_evaluacion_modelos.pdf",
        "title": "Evaluación y Validación de Modelos",
        "content": """La evaluación rigurosa es crucial para estimar el rendimiento real de un modelo.
División de datos: train/validation/test (ej. 70/15/15 o 80/10/10). Validación cruzada (k-fold)
usa múltiples particiones para estimación más robusta.

Métricas de clasificación: accuracy, precision, recall, F1-score, AUC-ROC, matriz de confusión.
Métricas de regresión: MSE, RMSE, MAE, R², MAPE.
Problemas comunes: data leakage (fuga de datos), desbalance de clases, overfitting al validation set.

Selección de modelo: comparar múltiples algoritmos, ajuste de hiperparámetros (grid search, random search,
Bayesian optimization). Interpretabilidad: feature importance, SHAP values, LIME.""",
    },
]

DEMO_QUESTIONS = [
    {
        "id": "q1",
        "question": "¿Qué es el aprendizaje supervisado?",
        "gold_answer": "Es un paradigma de aprendizaje automático donde el algoritmo aprende a partir de datos etiquetados, mapeando entradas a salidas esperadas.",
    },
    {
        "id": "q2",
        "question": "¿Cuáles son los principales tipos de aprendizaje supervisado?",
        "gold_answer": "Clasificación (salida categórica) y regresión (salida continua).",
    },
    {
        "id": "q3",
        "question": "¿Qué es el overfitting y cómo se previene?",
        "gold_answer": "Overfitting es cuando el modelo memoriza datos de entrenamiento en lugar de generalizar. Se previene con validación cruzada, regularización, early stopping, dropout.",
    },
    {
        "id": "q4",
        "question": "¿Cómo funcionan las redes neuronales?",
        "gold_answer": "Son modelos con neuronas artificiales en capas (entrada, ocultas, salida) que aplican transformación lineal + activación no lineal. Se entrenan con backpropagation y gradient descent.",
    },
    {
        "id": "q5",
        "question": "¿Qué es el aprendizaje no supervisado?",
        "gold_answer": "Encuentra patrones en datos sin etiquetas. Tareas: clustering, reducción de dimensionalidad, detección de anomalías.",
    },
    {
        "id": "q6",
        "question": "¿Para qué sirve PCA?",
        "gold_answer": "Reducción de dimensionalidad proyectando datos a subespacio de menor dimensión maximizando varianza explicada.",
    },
    {
        "id": "q7",
        "question": "¿Qué métricas se usan para evaluar clasificación?",
        "gold_answer": "Accuracy, precision, recall, F1-score, AUC-ROC, matriz de confusión.",
    },
    {
        "id": "q8",
        "question": "¿Qué es data leakage?",
        "gold_answer": "Fuga de información del conjunto de test/validation al entrenamiento, inflando métricas artificialmente.",
    },
]


def generate_demo_data(output_dir: Path, force: bool = False) -> None:
    """Genera PDFs sintéticos y preguntas de demo."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verificar si ya existe
    pdf_files = list(output_dir.glob("*.pdf"))
    questions_file = output_dir.parent / "demo_questions.jsonl"

    if pdf_files and not force:
        print(f"Ya existen {len(pdf_files)} PDFs en {output_dir}. Use --force para sobrescribir.")
        return

    # Generar PDFs con fpdf2
    for doc in DEMO_DOCUMENTS:
        pdf_path = output_dir / doc["filename"]
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        # Título
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.multi_cell(0, 8, doc["title"])
        pdf.ln(5)
        # Contenido
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, doc["content"])
        pdf.output(str(pdf_path))

    # Generar questions.jsonl
    questions_file.parent.mkdir(parents=True, exist_ok=True)
    with open(questions_file, "w", encoding="utf-8") as f:
        for q in DEMO_QUESTIONS:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Generados {len(DEMO_DOCUMENTS)} documentos en {output_dir}")
    print(f"Generadas {len(DEMO_QUESTIONS)} preguntas en {questions_file}")
