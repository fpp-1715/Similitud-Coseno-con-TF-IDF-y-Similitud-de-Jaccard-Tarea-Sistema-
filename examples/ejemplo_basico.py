"""
================================================================================
EJEMPLO DE USO: ejemplo_basico.py
DESCRIPCIÓN: Demostración básica del sistema de búsqueda por similitud
AUTOR: Equipo 8
================================================================================

Este script muestra cómo usar el sistema de forma programática,
sin necesidad de la interfaz web.
"""

import sys
import os

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from processor import DocumentProcessor
from similarity import SimilarityEngine


def main():
    """Función principal del ejemplo"""
    
    print("=" * 70)
    print("EJEMPLO BÁSICO: Sistema de Búsqueda por Similitud")
    print("=" * 70)
    
    # =========================================================================
    # PASO 1: Crear documentos de ejemplo
    # =========================================================================
    print("\n📄 PASO 1: Creando documentos de ejemplo...")
    
    documentos = {
        "doc1_ia.txt": """
            La inteligencia artificial es un campo de la informática que se enfoca 
            en crear sistemas capaces de realizar tareas que normalmente requieren 
            inteligencia humana. Esto incluye el aprendizaje automático y el 
            procesamiento del lenguaje natural.
        """,
        "doc2_ml.txt": """
            El aprendizaje automático es una rama de la inteligencia artificial 
            que permite a las máquinas aprender de los datos. Los algoritmos de 
            aprendizaje automático pueden identificar patrones y hacer predicciones.
        """,
        "doc3_web.txt": """
            El desarrollo web involucra la creación de sitios y aplicaciones para 
            internet. Se utilizan tecnologías como HTML, CSS y JavaScript para 
            construir interfaces de usuario interactivas.
        """,
        "doc4_db.txt": """
            Las bases de datos son sistemas organizados para almacenar información.
            SQL es el lenguaje estándar para consultar bases de datos relacionales.
            MongoDB es una base de datos NoSQL popular.
        """
    }
    
    for nombre in documentos:
        print(f"  ✓ {nombre}")
    
    # =========================================================================
    # PASO 2: Preprocesar documentos
    # =========================================================================
    print("\n🔧 PASO 2: Preprocesando documentos...")
    
    processor = DocumentProcessor()
    docs_procesados = {}
    
    for nombre, texto in documentos.items():
        texto_limpio = processor.preprocess_text(texto)
        docs_procesados[nombre] = texto_limpio
        print(f"  ✓ {nombre}: {len(texto_limpio)} caracteres")
    
    # =========================================================================
    # PASO 3: Calcular similitudes
    # =========================================================================
    print("\n🔍 PASO 3: Calculando similitudes...")
    
    engine = SimilarityEngine()
    doc_list = list(docs_procesados.values())
    doc_names = list(docs_procesados.keys())
    
    # Documento consulta: el primero (sobre IA)
    query_idx = 0
    print(f"\n  📌 Documento consulta: {doc_names[query_idx]}")
    
    # Calcular TF-IDF Coseno
    print("\n  --- SIMILITUD TF-IDF COSENO ---")
    results_tfidf, matrix, terms = engine.calculate_tfidf_cosine_similarity(doc_list, query_idx)
    
    for idx, sim in results_tfidf:
        if idx != query_idx:
            print(f"    {doc_names[idx]}: {sim*100:.2f}%")
    
    # Calcular Jaccard
    print("\n  --- SIMILITUD JACCARD ---")
    results_jaccard = engine.calculate_jaccard_similarity(doc_list, query_idx)
    
    for idx, sim in results_jaccard:
        if idx != query_idx:
            print(f"    {doc_names[idx]}: {sim*100:.2f}%")
    
    # =========================================================================
    # PASO 4: Mostrar términos importantes
    # =========================================================================
    print("\n📊 PASO 4: Términos más relevantes del documento consulta...")
    
    top_terms = engine.get_top_terms_tfidf(matrix, terms, query_idx, top_n=10)
    
    print("\n  Top 10 términos (TF-IDF):")
    for i, (term, weight) in enumerate(top_terms, 1):
        print(f"    {i}. {term}: {weight:.4f}")
    
    # =========================================================================
    # CONCLUSIÓN
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ Ejemplo completado exitosamente")
    print("=" * 70)
    
    print("\n💡 Observaciones:")
    print("   - doc2_ml.txt es el más similar a doc1_ia.txt (ambos sobre IA/ML)")
    print("   - doc3_web.txt y doc4_db.txt son menos similares (temas diferentes)")
    print("   - TF-IDF considera la importancia de términos únicos")
    print("   - Jaccard solo considera presencia/ausencia de términos")


if __name__ == "__main__":
    main()
