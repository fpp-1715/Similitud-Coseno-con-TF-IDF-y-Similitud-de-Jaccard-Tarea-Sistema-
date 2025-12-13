# 📄 Sistema de Búsqueda de Documentos por Similitud

## Tarea Extraclase 2 - Equipo 8

---

### 📚 Información Académica

| Campo | Información |
|-------|-------------|
| **Asignatura** | Sistemas de Información |
| **Carrera** | Ciencias de la Computación - 4to Año |
| **Fecha** | Diciembre 2025 |

### 👥 Integrantes del Equipo

| # | Nombre Completo |
|---|-----------------|
| 1 | Franklin Pérez Pérez |
| 2 | Carlos Manuel Hernández Hernández |
| 3 | Ariel David Toledo Rojas |

---

## 📋 Descripción del Proyecto

Sistema de búsqueda de documentos por similitud de contenido que permite encontrar documentos similares a uno dado (documento consulta). El sistema implementa dos métodos de cálculo de similitud:

1. **Similitud Coseno usando representación TF-IDF** - Basada en la frecuencia de términos ponderada
2. **Similitud de Jaccard** - Basada en la intersección y unión de conjuntos de palabras

El sistema cuenta con una **interfaz web interactiva** desarrollada con Streamlit que permite:
- Seleccionar documentos de consulta
- Visualizar rankings de similitud
- Comparar ambos métodos de similitud
- Analizar términos relevantes mediante gráficos

---

## 🎯 Funcionalidades Implementadas

### Requisitos del Proyecto ✅

| Requisito | Estado | Implementación |
|-----------|:------:|----------------|
| Similitud Coseno TF-IDF | ✅ | `scikit-learn` (TfidfVectorizer + cosine_similarity) |
| Similitud Jaccard | ✅ | Función personalizada basada en conjuntos |
| Procesar archivos TXT | ✅ | Múltiples codificaciones (UTF-8, Latin-1, CP1252) |
| Procesar archivos PDF | ✅ | Biblioteca `PyPDF2` |
| Matriz término-documento | ✅ | Generada automáticamente con TF-IDF |
| Ranking por similitud | ✅ | Ordenado de mayor a menor similitud |
| Términos relevantes | ✅ | Visualización de términos que contribuyen a similitud |
| Representación visual | ✅ | Gráficos de barras interactivos con `Plotly` |
| Interfaz interactiva | ✅ | Aplicación web con `Streamlit` |
| Carpeta configurable | ✅ | Usuario puede seleccionar carpeta de documentos |

---

## 📁 Estructura del Proyecto

```
proyecto_equipo8/
│
├── src/                              # Código fuente principal
│   ├── main.py                       # Aplicación Streamlit (punto de entrada)
│   ├── processor.py                  # Módulo de procesamiento de documentos
│   ├── similarity.py                 # Módulo de cálculo de similitud
│   └── utils.py                      # Funciones auxiliares
│
├── data/                             # Datos de prueba
│   └── documentos/                   # Documentos de ejemplo (TXT y PDF)
│       ├── documento1_ia.txt
│       ├── documento2_deep_learning.txt
│       ├── documento3_poo.txt
│       ├── documento4_bases_datos.txt
│       ├── documento5_ciencia_datos.txt
│       ├── pdf1_machine_learning.pdf
│       ├── pdf2_ciberseguridad.pdf
│       ├── pdf3_redes_neuronales.pdf
│       └── pdf4_desarrollo_web.pdf
│
├── tests/                            # Pruebas unitarias
│   ├── __init__.py
│   ├── test_processor.py             # Tests del procesador
│   └── test_similarity.py            # Tests de similitud
│
├── examples/                         # Ejemplos de uso programático
│   ├── ejemplo_basico.py             # Uso básico del sistema
│   └── ejemplo_comparacion_metodos.py # Comparación TF-IDF vs Jaccard
│
├── requirements.txt                  # Dependencias del proyecto
└── README.md                         # Este archivo
```

---

## 🚀 Instalación y Configuración

### Requisitos Previos

- **Python 3.8** o superior
- **pip** (gestor de paquetes de Python)

### Paso 1: Descargar el Proyecto

Extraer el archivo ZIP o clonar el repositorio en una carpeta local.

### Paso 2: Crear Entorno Virtual (Recomendado)

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv .venv
.venv\Scripts\activate.bat

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Verificar Instalación

```bash
python -c "import streamlit; import sklearn; import PyPDF2; print('✅ Instalación correcta')"
```

---

## 📖 Uso del Sistema

### Opción 1: Interfaz Web (Recomendado) 🌐

```bash
cd src
streamlit run main.py
```

Abrir en el navegador: **http://localhost:8501**

#### Pasos en la interfaz:
1. **Configurar carpeta** de documentos en el panel lateral (opcional)
2. **Seleccionar** un documento de consulta del menú desplegable
3. **Hacer clic** en "🔍 Buscar Documentos Similares"
4. **Explorar** los resultados:
   - Rankings de similitud TF-IDF y Jaccard
   - Gráficos comparativos
   - Análisis de términos relevantes

### Opción 2: Uso Programático 💻

```python
from processor import DocumentProcessor
from similarity import SimilarityEngine

# Cargar documentos desde una carpeta
processor = DocumentProcessor()
documents = processor.load_documents("./data/documentos")

# Crear motor de similitud
engine = SimilarityEngine()
doc_list = list(documents.values())
doc_names = list(documents.keys())

# Calcular similitud TF-IDF Coseno (documento índice 0 como consulta)
results_tfidf, matrix, terms = engine.calculate_tfidf_cosine_similarity(doc_list, 0)

# Calcular similitud de Jaccard
results_jaccard = engine.calculate_jaccard_similarity(doc_list, 0)

# Mostrar resultados
print("Similitud TF-IDF Coseno:")
for idx, score in results_tfidf:
    print(f"  {doc_names[idx]}: {score:.4f}")

print("\nSimilitud Jaccard:")
for idx, score in results_jaccard:
    print(f"  {doc_names[idx]}: {score:.4f}")
```

### Opción 3: Ejecutar Ejemplos 📝

```bash
cd examples
python ejemplo_basico.py
python ejemplo_comparacion_metodos.py
```

---

## 🔧 Descripción de Módulos

### `processor.py` - Procesamiento de Documentos

**Clase:** `DocumentProcessor`

| Método | Descripción |
|--------|-------------|
| `load_documents(folder_path)` | Carga todos los documentos TXT/PDF de una carpeta |
| `extract_text_from_txt(file_path)` | Extrae texto de archivos .txt |
| `extract_text_from_pdf(file_path)` | Extrae texto de archivos .pdf usando PyPDF2 |
| `preprocess_text(text)` | Normaliza texto (minúsculas, elimina puntuación) |
| `tokenize(text)` | Divide el texto en tokens (palabras) |

### `similarity.py` - Cálculo de Similitud

**Clase:** `SimilarityEngine`

| Método | Descripción |
|--------|-------------|
| `calculate_tfidf_cosine_similarity(docs, query_idx)` | Calcula similitud usando TF-IDF + Coseno |
| `calculate_jaccard_similarity(docs, query_idx)` | Calcula similitud de Jaccard |
| `get_top_terms_tfidf(matrix, terms, doc_idx, n)` | Obtiene los N términos más relevantes |

### `main.py` - Aplicación Web

Aplicación Streamlit que integra todos los módulos y proporciona:
- Interfaz gráfica intuitiva
- Visualización de resultados con tablas y gráficos
- Configuración dinámica de parámetros

---

## 📊 Fundamento Teórico

### TF-IDF (Term Frequency - Inverse Document Frequency)

El método TF-IDF pondera la importancia de cada término considerando:

**Fórmula TF-IDF:**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

Donde:
- **TF(t,d)** = Frecuencia del término t en documento d / Total de términos en d
- **IDF(t)** = log(N / df(t))
- **N** = Número total de documentos
- **df(t)** = Número de documentos que contienen el término t

**Similitud Coseno:**
```
cos(θ) = (A · B) / (||A|| × ||B||)
```

La similitud coseno mide el ángulo entre dos vectores TF-IDF, donde:
- **1.0** = Documentos idénticos
- **0.0** = Documentos sin términos en común

### Similitud de Jaccard

Mide la similitud entre conjuntos de palabras:

**Fórmula:**
```
J(A, B) = |A ∩ B| / |A ∪ B|
```

Donde:
- **A ∩ B** = Palabras comunes entre ambos documentos
- **A ∪ B** = Todas las palabras únicas de ambos documentos

Interpretación:
- **1.0** = Conjuntos idénticos
- **0.0** = Sin palabras en común

### Comparación de Métodos

| Aspecto | TF-IDF Coseno | Jaccard |
|---------|---------------|---------|
| Considera frecuencia | ✅ Sí | ❌ No |
| Considera rareza del término | ✅ Sí (IDF) | ❌ No |
| Complejidad | Mayor | Menor |
| Mejor para | Textos largos, análisis semántico | Textos cortos, comparación rápida |

---

## 🧪 Pruebas

### Ejecutar Tests Unitarios

```bash
# Desde la carpeta raíz del proyecto
python -m pytest tests/ -v

# O ejecutar tests individuales
python tests/test_processor.py
python tests/test_similarity.py
```

### Documentos de Prueba Incluidos

El proyecto incluye **9 documentos de ejemplo** en `data/documentos/`:

**Archivos TXT (5):**
- `documento1_ia.txt` - Inteligencia Artificial
- `documento2_deep_learning.txt` - Deep Learning
- `documento3_poo.txt` - Programación Orientada a Objetos
- `documento4_bases_datos.txt` - Bases de Datos
- `documento5_ciencia_datos.txt` - Ciencia de Datos

**Archivos PDF (4):**
- `pdf1_machine_learning.pdf` - Machine Learning
- `pdf2_ciberseguridad.pdf` - Ciberseguridad
- `pdf3_redes_neuronales.pdf` - Redes Neuronales
- `pdf4_desarrollo_web.pdf` - Desarrollo Web

---

## 📦 Dependencias

```
streamlit>=1.28.0      # Framework web interactivo
scikit-learn>=1.3.0    # TF-IDF y similitud coseno
pandas>=2.0.0          # Manipulación de datos
numpy>=1.24.0          # Cálculos numéricos
plotly>=5.18.0         # Gráficos interactivos
PyPDF2>=3.0.0          # Lectura de archivos PDF
```

---

## 🖥️ Capturas de Pantalla

### Interfaz Principal
La aplicación muestra:
- Panel lateral con configuración
- Selector de documento de consulta
- Resultados en tablas ordenadas por similitud
- Gráficos de barras comparativos
- Análisis de términos relevantes

---

## 📝 Notas Adicionales

1. **Codificación de archivos:** El sistema intenta múltiples codificaciones (UTF-8, Latin-1, CP1252) para archivos TXT.

2. **PDFs escaneados:** El sistema no puede extraer texto de PDFs que sean imágenes escaneadas (requieren OCR).

3. **Rendimiento:** Para colecciones grandes de documentos, el cálculo TF-IDF puede tomar algunos segundos.

4. **Carpeta personalizada:** Puede configurar cualquier carpeta de documentos desde la interfaz web.

---

## 📄 Licencia

Proyecto académico desarrollado para la asignatura de Sistemas de Información.
Ciencias de la Computación - 4to Año - Diciembre 2025

---

**Equipo 8** | Franklin Pérez Pérez • Carlos Manuel Hernández Hernández • Ariel David Toledo Rojas



RUN SERVER
"E:\Ciber\sistema\Tarea Extraclase 2\proyecto_equipo8\src"; & "E:\Ciber\sistema\Tarea Extraclase 2\.venv\Scripts\streamlit.exe" run main.py --server.headless true