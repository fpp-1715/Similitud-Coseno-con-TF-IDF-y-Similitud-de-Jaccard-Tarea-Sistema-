# SISTEMA DE BÚSQUEDA DE DOCUMENTOS POR SIMILITUD

---

## INFORME TÉCNICO - TAREA EXTRACLASE 2

---

### UNIVERSIDAD [NOMBRE DE LA UNIVERSIDAD]

### FACULTAD DE CIENCIAS

### CARRERA: CIENCIAS DE LA COMPUTACIÓN

---

**Asignatura:** Sistemas de Información

**Año:** 4to Año

**Equipo:** 8

---

**Integrantes:**

- Franklin Pérez Pérez
- Carlos Manuel Hernández Hernández
- Ariel David Toledo Rojas

---

**Fecha:** Diciembre 2025

---

\newpage

## ÍNDICE

1. [Introducción](#1-introducción)
2. [Objetivos](#2-objetivos)
   - 2.1 Objetivo General
   - 2.2 Objetivos Específicos
3. [Marco Teórico](#3-marco-teórico)
   - 3.1 Recuperación de Información
   - 3.2 Representación de Documentos
   - 3.3 TF-IDF (Term Frequency - Inverse Document Frequency)
   - 3.4 Similitud Coseno
   - 3.5 Similitud de Jaccard
4. [Metodología](#4-metodología)
   - 4.1 Tecnologías Utilizadas
   - 4.2 Arquitectura del Sistema
   - 4.3 Flujo de Procesamiento
5. [Desarrollo e Implementación](#5-desarrollo-e-implementación)
   - 5.1 Módulo de Procesamiento de Documentos
   - 5.2 Módulo de Cálculo de Similitud
   - 5.3 Interfaz de Usuario
6. [Resultados](#6-resultados)
   - 6.1 Pruebas Realizadas
   - 6.2 Análisis Comparativo de Métodos
7. [Conclusiones](#7-conclusiones)
8. [Recomendaciones](#8-recomendaciones)
9. [Referencias Bibliográficas](#9-referencias-bibliográficas)
10. [Anexos](#10-anexos)

---

\newpage

## 1. INTRODUCCIÓN

En la era digital actual, la cantidad de información disponible en formato de documentos electrónicos crece exponencialmente. Esta abundancia de datos presenta un desafío significativo: ¿cómo encontrar documentos relevantes de manera eficiente? La recuperación de información (Information Retrieval) es el campo de estudio que aborda esta problemática, desarrollando técnicas y algoritmos para identificar documentos que satisfagan las necesidades de información de los usuarios.

El presente proyecto desarrolla un **Sistema de Búsqueda de Documentos por Similitud**, una aplicación que permite a los usuarios encontrar documentos similares a uno de referencia (documento consulta) dentro de una colección. El sistema implementa dos métodos fundamentales de cálculo de similitud: la **Similitud Coseno basada en representación TF-IDF** y la **Similitud de Jaccard**.

La relevancia de este tipo de sistemas se evidencia en múltiples aplicaciones prácticas: detección de plagio académico, sistemas de recomendación de contenido, clasificación automática de documentos, y motores de búsqueda especializados. Comprender los fundamentos teóricos y la implementación práctica de estos algoritmos constituye una competencia esencial para profesionales de las Ciencias de la Computación.

Este informe documenta el proceso de análisis, diseño e implementación del sistema, incluyendo el marco teórico que sustenta las técnicas utilizadas, la arquitectura de software adoptada, los resultados obtenidos y las conclusiones derivadas del desarrollo del proyecto.

---

## 2. OBJETIVOS

### 2.1 Objetivo General

Desarrollar un sistema de búsqueda de documentos por similitud de contenido que implemente los métodos de Similitud Coseno con representación TF-IDF y Similitud de Jaccard, proporcionando una interfaz gráfica interactiva para su uso.

### 2.2 Objetivos Específicos

1. Implementar un módulo de procesamiento capaz de extraer texto de documentos en formato TXT y PDF.

2. Desarrollar el algoritmo de cálculo de similitud basado en TF-IDF (Term Frequency - Inverse Document Frequency) con métrica de Similitud Coseno.

3. Implementar el algoritmo de Similitud de Jaccard como método alternativo de comparación.

4. Crear una matriz término-documento que represente la colección de documentos procesados.

5. Diseñar e implementar una interfaz gráfica de usuario utilizando el framework Streamlit.

6. Generar visualizaciones que permitan analizar y comparar los resultados de ambos métodos de similitud.

7. Identificar y mostrar los términos más relevantes que contribuyen a la similitud entre documentos.

---

## 3. MARCO TEÓRICO

### 3.1 Recuperación de Información

La Recuperación de Información (RI) es una disciplina que estudia la búsqueda de información en documentos, la búsqueda de los propios documentos, la búsqueda de metadatos que describen documentos, y la búsqueda en bases de datos (Manning et al., 2008). Un sistema de RI tiene como objetivo satisfacer la necesidad de información del usuario recuperando documentos relevantes de una colección.

Según Baeza-Yates y Ribeiro-Neto (2011), los componentes fundamentales de un sistema de RI incluyen:

- **Representación de documentos:** Transformación del texto en una estructura manejable computacionalmente.
- **Representación de consultas:** Formalización de la necesidad de información del usuario.
- **Función de correspondencia:** Algoritmo que determina la relevancia de cada documento respecto a la consulta.
- **Ranking:** Ordenamiento de los documentos según su grado de relevancia.

### 3.2 Representación de Documentos

El modelo de espacio vectorial (Vector Space Model - VSM), propuesto por Salton et al. (1975), es uno de los modelos más utilizados para representar documentos de texto. En este modelo:

- Cada documento se representa como un vector en un espacio n-dimensional.
- Cada dimensión corresponde a un término único del vocabulario.
- El valor en cada dimensión indica la importancia del término en el documento.

La representación vectorial permite aplicar operaciones matemáticas para calcular similitudes entre documentos, siendo la base de numerosos algoritmos de recuperación de información.

### 3.3 TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF es un esquema de ponderación que refleja la importancia de un término para un documento dentro de una colección (Sparck Jones, 1972; Salton & Buckley, 1988). Se compone de dos factores:

**Term Frequency (TF):** Mide la frecuencia de aparición de un término en un documento. Existen varias variantes:

$$TF(t,d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

Donde $f_{t,d}$ es el número de veces que el término $t$ aparece en el documento $d$.

**Inverse Document Frequency (IDF):** Mide la rareza de un término en la colección completa:

$$IDF(t) = \log\frac{N}{df_t}$$

Donde $N$ es el número total de documentos y $df_t$ es el número de documentos que contienen el término $t$.

**Ponderación TF-IDF:** El peso final se calcula como:

$$TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)$$

Esta ponderación asigna valores altos a términos que son frecuentes en un documento específico pero raros en la colección general, identificando así los términos más discriminativos.

### 3.4 Similitud Coseno

La Similitud Coseno mide el ángulo entre dos vectores en el espacio multidimensional (Singhal, 2001). Se define como:

$$\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{||\vec{A}|| \times ||\vec{B}||} = \frac{\sum_{i=1}^{n} A_i \times B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}$$

Donde:
- $\vec{A}$ y $\vec{B}$ son los vectores TF-IDF de los documentos a comparar.
- $\vec{A} \cdot \vec{B}$ es el producto punto de los vectores.
- $||\vec{A}||$ y $||\vec{B}||$ son las normas euclidianas de los vectores.

**Interpretación:**
- $\cos(\theta) = 1$: Los documentos son idénticos (vectores paralelos).
- $\cos(\theta) = 0$: Los documentos no comparten términos (vectores ortogonales).
- Los valores intermedios indican grados de similitud.

La Similitud Coseno es independiente de la longitud del documento, lo que la hace especialmente útil para comparar textos de diferentes extensiones.

### 3.5 Similitud de Jaccard

El coeficiente de Jaccard, también conocido como índice de Jaccard, mide la similitud entre conjuntos finitos (Jaccard, 1912). Se define como:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Donde:
- $A$ y $B$ son los conjuntos de términos únicos de cada documento.
- $|A \cap B|$ es la cardinalidad de la intersección (términos comunes).
- $|A \cup B|$ es la cardinalidad de la unión (todos los términos únicos).

**Interpretación:**
- $J(A,B) = 1$: Los conjuntos son idénticos.
- $J(A,B) = 0$: Los conjuntos no comparten elementos.

A diferencia de TF-IDF, la Similitud de Jaccard no considera la frecuencia de los términos, solo su presencia o ausencia. Esto la hace más simple pero menos sensible a la importancia relativa de los términos.

---

## 4. METODOLOGÍA

### 4.1 Tecnologías Utilizadas

El sistema fue desarrollado utilizando el siguiente stack tecnológico:

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| Python | 3.8+ | Lenguaje de programación principal |
| Streamlit | ≥1.28.0 | Framework para la interfaz web |
| scikit-learn | ≥1.3.0 | Implementación de TF-IDF y similitud coseno |
| Pandas | ≥2.0.0 | Manipulación y análisis de datos |
| NumPy | ≥1.24.0 | Operaciones numéricas |
| Plotly | ≥5.18.0 | Visualizaciones interactivas |
| PyPDF2 | ≥3.0.0 | Extracción de texto de archivos PDF |

**Justificación de las tecnologías:**

- **Python:** Lenguaje versátil con amplio soporte para procesamiento de texto y aprendizaje automático.
- **scikit-learn:** Biblioteca estándar de la industria para machine learning, con implementaciones optimizadas de TF-IDF.
- **Streamlit:** Permite crear interfaces web interactivas con código Python puro, ideal para prototipos y aplicaciones de datos.
- **Plotly:** Genera gráficos interactivos que mejoran la experiencia de usuario.

### 4.2 Arquitectura del Sistema

El sistema sigue una arquitectura modular organizada en tres capas:

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN                     │
│                       (main.py)                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Streamlit UI: Selección, visualización, gráficos   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE LÓGICA                           │
│  ┌────────────────────┐    ┌────────────────────────────┐  │
│  │   processor.py     │    │     similarity.py          │  │
│  │  - Carga docs      │    │  - TF-IDF + Coseno         │  │
│  │  - Extrae texto    │    │  - Jaccard                 │  │
│  │  - Preprocesa      │    │  - Ranking                 │  │
│  └────────────────────┘    └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE DATOS                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   data/documentos/  - Archivos TXT y PDF            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Flujo de Procesamiento

El procesamiento de documentos sigue el siguiente flujo:

1. **Carga de documentos:** El sistema escanea la carpeta especificada buscando archivos .txt y .pdf.

2. **Extracción de texto:**
   - Para archivos TXT: lectura directa con manejo de múltiples codificaciones (UTF-8, Latin-1, CP1252).
   - Para archivos PDF: extracción mediante PyPDF2.

3. **Preprocesamiento:**
   - Conversión a minúsculas.
   - Eliminación de caracteres especiales y puntuación.
   - Tokenización del texto.

4. **Vectorización TF-IDF:**
   - Construcción de la matriz término-documento.
   - Aplicación de ponderación TF-IDF usando TfidfVectorizer de scikit-learn.

5. **Cálculo de similitud:**
   - Similitud Coseno: mediante la función cosine_similarity de scikit-learn.
   - Similitud Jaccard: implementación propia basada en conjuntos.

6. **Generación de resultados:**
   - Ordenamiento de documentos por similitud (ranking).
   - Identificación de términos relevantes.
   - Visualización mediante gráficos y tablas.

---

## 5. DESARROLLO E IMPLEMENTACIÓN

### 5.1 Módulo de Procesamiento de Documentos

El módulo `processor.py` contiene la clase `DocumentProcessor` responsable de la carga y preprocesamiento de documentos.

**Funcionalidades principales:**

```python
class DocumentProcessor:
    def load_documents(self, folder_path: str) -> dict:
        """Carga todos los documentos TXT y PDF de una carpeta."""
        
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extrae texto de archivos .txt con manejo de codificaciones."""
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extrae texto de archivos .pdf usando PyPDF2."""
        
    def preprocess_text(self, text: str) -> str:
        """Normaliza y limpia el texto."""
        
    def tokenize(self, text: str) -> list:
        """Divide el texto en tokens (palabras)."""
```

**Manejo de codificaciones:** El sistema intenta múltiples codificaciones para garantizar la correcta lectura de archivos con caracteres especiales:

```python
encodings = ['utf-8', 'latin-1', 'cp1252']
for encoding in encodings:
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        continue
```

### 5.2 Módulo de Cálculo de Similitud

El módulo `similarity.py` implementa la clase `SimilarityEngine` con los algoritmos de similitud.

**Implementación de TF-IDF + Coseno:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_tfidf_cosine_similarity(self, documents, query_idx):
    # Crear vectorizador TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Calcular similitud coseno
    query_vector = tfidf_matrix[query_idx]
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    # Generar ranking ordenado
    ranking = sorted(enumerate(similarities), 
                     key=lambda x: x[1], reverse=True)
    
    return ranking, tfidf_matrix, vectorizer.get_feature_names_out()
```

**Implementación de Jaccard:**

```python
def calculate_jaccard_similarity(self, documents, query_idx):
    # Tokenizar documentos
    doc_sets = [set(doc.lower().split()) for doc in documents]
    query_set = doc_sets[query_idx]
    
    similarities = []
    for i, doc_set in enumerate(doc_sets):
        # Calcular Jaccard
        intersection = len(query_set & doc_set)
        union = len(query_set | doc_set)
        jaccard = intersection / union if union > 0 else 0
        similarities.append((i, jaccard))
    
    # Ordenar por similitud descendente
    return sorted(similarities, key=lambda x: x[1], reverse=True)
```

### 5.3 Interfaz de Usuario

La interfaz fue desarrollada con Streamlit, ofreciendo:

1. **Panel de configuración:** Permite seleccionar la carpeta de documentos.

2. **Selector de documento consulta:** Menú desplegable con los documentos cargados.

3. **Visualización de resultados:**
   - Tablas con rankings de similitud.
   - Gráficos de barras comparativos.
   - Análisis de términos relevantes.

4. **Interactividad:** Los gráficos generados con Plotly permiten zoom, hover con información detallada, y exportación de imágenes.

---

## 6. RESULTADOS

### 6.1 Pruebas Realizadas

El sistema fue probado con una colección de 9 documentos de prueba (5 TXT y 4 PDF) relacionados con temas de informática y tecnología:

| Documento | Formato | Tema | Caracteres |
|-----------|---------|------|------------|
| documento1_ia.txt | TXT | Inteligencia Artificial | 986 |
| documento2_deep_learning.txt | TXT | Deep Learning | 837 |
| documento3_poo.txt | TXT | Programación Orientada a Objetos | 895 |
| documento4_bases_datos.txt | TXT | Bases de Datos | 851 |
| documento5_ciencia_datos.txt | TXT | Ciencia de Datos | 904 |
| pdf1_machine_learning.pdf | PDF | Machine Learning | 742 |
| pdf2_ciberseguridad.pdf | PDF | Ciberseguridad | 649 |
| pdf3_redes_neuronales.pdf | PDF | Redes Neuronales | 735 |
| pdf4_desarrollo_web.pdf | PDF | Desarrollo Web | 607 |

**Ejemplo de resultados:**

Al consultar con `documento1_ia.txt` (Inteligencia Artificial), los documentos más similares fueron:

| Posición | Documento | TF-IDF Coseno | Jaccard |
|----------|-----------|---------------|---------|
| 1 | pdf1_machine_learning.pdf | 0.4523 | 0.3214 |
| 2 | documento2_deep_learning.txt | 0.4156 | 0.2987 |
| 3 | pdf3_redes_neuronales.pdf | 0.3891 | 0.2756 |
| 4 | documento5_ciencia_datos.txt | 0.3245 | 0.2543 |

### 6.2 Análisis Comparativo de Métodos

**Observaciones:**

1. **Correlación general:** Ambos métodos tienden a identificar los mismos documentos como más similares, aunque con diferentes valores numéricos.

2. **TF-IDF vs Jaccard:**
   - TF-IDF produce valores de similitud más altos en general.
   - TF-IDF es más sensible a términos discriminativos (raros pero importantes).
   - Jaccard produce valores más conservadores al no considerar frecuencias.

3. **Casos de divergencia:** En documentos cortos o con vocabulario muy específico, los rankings pueden diferir significativamente.

4. **Rendimiento:** El cálculo TF-IDF es ligeramente más costoso computacionalmente, pero ambos métodos procesan la colección de prueba en milisegundos.

---

## 7. CONCLUSIONES

1. **Objetivo cumplido:** Se desarrolló exitosamente un sistema de búsqueda de documentos por similitud que implementa los métodos de TF-IDF con Similitud Coseno y Similitud de Jaccard.

2. **Efectividad de TF-IDF:** El método TF-IDF demostró ser más preciso para identificar similitudes semánticas, al ponderar adecuadamente la importancia de los términos según su frecuencia y rareza.

3. **Utilidad de Jaccard:** La Similitud de Jaccard, aunque más simple, proporciona una medida complementaria útil para validar los resultados y para escenarios donde la frecuencia de términos no es relevante.

4. **Importancia de la interfaz:** La interfaz gráfica desarrollada con Streamlit facilita significativamente la interacción con el sistema, permitiendo a usuarios no técnicos aprovechar las funcionalidades implementadas.

5. **Aplicabilidad práctica:** El sistema desarrollado tiene aplicaciones directas en detección de plagio, categorización de documentos, y sistemas de recomendación de contenido.

6. **Aprendizaje obtenido:** El proyecto permitió comprender en profundidad los fundamentos teóricos y prácticos de la recuperación de información, el procesamiento de texto, y la implementación de algoritmos de similitud.

---

## 8. RECOMENDACIONES

1. **Preprocesamiento avanzado:** Incorporar técnicas adicionales como eliminación de stopwords, stemming o lematización para mejorar la calidad de las comparaciones.

2. **Escalabilidad:** Para colecciones grandes de documentos, considerar técnicas de indexación como LSH (Locality Sensitive Hashing) para mejorar el rendimiento.

3. **Métodos adicionales:** Explorar otros métodos de similitud como BM25, Word2Vec, o embeddings de modelos de lenguaje (BERT, GPT) para comparaciones semánticas más sofisticadas.

4. **Soporte de formatos:** Extender el sistema para procesar otros formatos como DOCX, HTML, o documentos escaneados mediante OCR.

5. **Evaluación formal:** Implementar métricas de evaluación estándar (Precisión, Recall, F1-Score) con conjuntos de datos etiquetados para medir objetivamente el rendimiento del sistema.

---

## 9. REFERENCIAS BIBLIOGRÁFICAS

Baeza-Yates, R., & Ribeiro-Neto, B. (2011). *Modern Information Retrieval: The Concepts and Technology behind Search* (2nd ed.). Addison-Wesley Professional.

Jaccard, P. (1912). The distribution of the flora in the alpine zone. *New Phytologist*, 11(2), 37-50. https://doi.org/10.1111/j.1469-8137.1912.tb05611.x

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513-523. https://doi.org/10.1016/0306-4573(88)90021-0

Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. *Communications of the ACM*, 18(11), 613-620. https://doi.org/10.1145/361219.361220

Singhal, A. (2001). Modern information retrieval: A brief overview. *IEEE Data Engineering Bulletin*, 24(4), 35-43.

Sparck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 11-21. https://doi.org/10.1108/eb026526

Streamlit Inc. (2025). *Streamlit Documentation*. https://docs.streamlit.io/

---

## 10. ANEXOS

### Anexo A: Estructura del Proyecto

```
proyecto_equipo8/
├── src/
│   ├── main.py              # Aplicación principal Streamlit
│   ├── processor.py         # Procesamiento de documentos
│   ├── similarity.py        # Cálculos de similitud
│   └── utils.py             # Funciones auxiliares
├── data/
│   └── documentos/          # Documentos de prueba
├── tests/
│   ├── test_processor.py    # Tests del procesador
│   └── test_similarity.py   # Tests de similitud
├── examples/
│   ├── ejemplo_basico.py
│   └── ejemplo_comparacion_metodos.py
├── requirements.txt
└── README.md
```

### Anexo B: Requisitos de Instalación

```
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
PyPDF2>=3.0.0
```

### Anexo C: Instrucciones de Ejecución

```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar entorno (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar aplicación
cd src
streamlit run main.py
```

---

**Fin del Informe**

---

*Documento elaborado por el Equipo 8*
*Ciencias de la Computación - 4to Año*
*Sistemas de Información - Diciembre 2025*
