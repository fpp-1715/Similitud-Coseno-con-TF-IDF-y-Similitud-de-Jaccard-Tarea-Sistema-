"""
================================================================================
EJEMPLO DE USO: ejemplo_comparacion_metodos.py
DESCRIPCIÓN: Comparación detallada de los métodos TF-IDF Coseno vs Jaccard
AUTOR: Equipo 8
================================================================================
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from processor import DocumentProcessor
from similarity import SimilarityEngine


def main():
    print("=" * 70)
    print("COMPARACIÓN: TF-IDF Coseno vs Jaccard")
    print("=" * 70)
    
    # Documentos diseñados para mostrar diferencias entre métodos
    documentos = [
        # Doc 0: Muchas repeticiones de "gato"
        "gato gato gato gato gato perro animal mascota",
        
        # Doc 1: Comparte términos pero sin repeticiones
        "gato perro animal mascota casa hogar",
        
        # Doc 2: Pocos términos en común
        "carro moto vehiculo transporte ciudad",
        
        # Doc 3: Términos completamente diferentes
        "computadora software programa codigo python"
    ]
    
    processor = DocumentProcessor()
    engine = SimilarityEngine()
    
    # Preprocesar
    docs_clean = [processor.preprocess_text(d) for d in documentos]
    
    print("\n📄 Documentos:")
    for i, doc in enumerate(docs_clean):
        print(f"  Doc {i}: '{doc}'")
    
    print("\n" + "-" * 70)
    print("Comparando desde Doc 0 (muchas repeticiones de 'gato')")
    print("-" * 70)
    
    # TF-IDF
    results_tfidf, _, _ = engine.calculate_tfidf_cosine_similarity(docs_clean, 0)
    
    # Jaccard
    results_jaccard = engine.calculate_jaccard_similarity(docs_clean, 0)
    
    print("\n{:<10} {:<20} {:<20}".format("Doc", "TF-IDF Coseno", "Jaccard"))
    print("-" * 50)
    
    for i in range(len(documentos)):
        tfidf_sim = next(sim for idx, sim in results_tfidf if idx == i)
        jaccard_sim = next(sim for idx, sim in results_jaccard if idx == i)
        print(f"Doc {i:<5} {tfidf_sim*100:>15.2f}%    {jaccard_sim*100:>15.2f}%")
    
    print("\n" + "=" * 70)
    print("📊 ANÁLISIS DE DIFERENCIAS")
    print("=" * 70)
    
    print("""
    TF-IDF COSENO:
    - Considera la FRECUENCIA de términos (TF)
    - Penaliza términos muy comunes (IDF)
    - La repetición de "gato" en Doc 0 aumenta su peso
    - Mejor para detectar documentos con vocabulario similar y frecuencias parecidas
    
    JACCARD:
    - Solo considera PRESENCIA o AUSENCIA de términos
    - No importa cuántas veces aparece un término
    - |A ∩ B| / |A ∪ B| - proporción de términos compartidos
    - Mejor para comparar conjuntos de vocabulario únicos
    
    CUÁNDO USAR CADA UNO:
    - TF-IDF: Cuando la frecuencia de términos es importante
    - Jaccard: Cuando solo importa qué términos están presentes
    """)


if __name__ == "__main__":
    main()
