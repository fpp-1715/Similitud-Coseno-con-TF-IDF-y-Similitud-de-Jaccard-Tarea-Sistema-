"""
================================================================================
TESTS UNITARIOS: test_similarity.py
DESCRIPCIÓN: Pruebas para el módulo de cálculo de similitud
AUTOR: Equipo 8
================================================================================
"""

import unittest
import sys
import os

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from similarity import SimilarityEngine


class TestSimilarityEngine(unittest.TestCase):
    """Tests para la clase SimilarityEngine"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.engine = SimilarityEngine()
        self.sample_docs = [
            "el gato negro duerme",
            "el perro negro corre",
            "el carro rojo brilla"
        ]
    
    def test_tfidf_returns_correct_structure(self):
        """Test: estructura de retorno de TF-IDF"""
        results, matrix, terms = self.engine.calculate_tfidf_cosine_similarity(
            self.sample_docs, 0
        )
        
        # Verificar que retorna lista de tuplas
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)  # 3 documentos
        
        # Verificar estructura de tuplas
        for idx, sim in results:
            self.assertIsInstance(idx, int)
            self.assertIsInstance(sim, float)
    
    def test_tfidf_self_similarity_is_one(self):
        """Test: similitud consigo mismo debe ser 1.0"""
        results, _, _ = self.engine.calculate_tfidf_cosine_similarity(
            self.sample_docs, 0
        )
        
        # El documento 0 consigo mismo debe tener similitud 1.0
        self_sim = next(sim for idx, sim in results if idx == 0)
        self.assertAlmostEqual(self_sim, 1.0, places=5)
    
    def test_tfidf_results_sorted(self):
        """Test: resultados ordenados de mayor a menor"""
        results, _, _ = self.engine.calculate_tfidf_cosine_similarity(
            self.sample_docs, 0
        )
        
        similarities = [sim for _, sim in results]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
    
    def test_jaccard_returns_correct_structure(self):
        """Test: estructura de retorno de Jaccard"""
        results = self.engine.calculate_jaccard_similarity(self.sample_docs, 0)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
    
    def test_jaccard_self_similarity_is_one(self):
        """Test: Jaccard consigo mismo debe ser 1.0"""
        results = self.engine.calculate_jaccard_similarity(self.sample_docs, 0)
        
        self_sim = next(sim for idx, sim in results if idx == 0)
        self.assertAlmostEqual(self_sim, 1.0, places=5)
    
    def test_jaccard_identical_docs(self):
        """Test: documentos idénticos tienen Jaccard = 1.0"""
        docs = ["hola mundo", "hola mundo"]
        results = self.engine.calculate_jaccard_similarity(docs, 0)
        
        other_sim = next(sim for idx, sim in results if idx == 1)
        self.assertAlmostEqual(other_sim, 1.0, places=5)
    
    def test_jaccard_no_common_terms(self):
        """Test: sin términos comunes Jaccard = 0"""
        docs = ["gato perro", "carro moto"]
        results = self.engine.calculate_jaccard_similarity(docs, 0)
        
        other_sim = next(sim for idx, sim in results if idx == 1)
        self.assertAlmostEqual(other_sim, 0.0, places=5)
    
    def test_jaccard_range(self):
        """Test: Jaccard siempre entre 0 y 1"""
        results = self.engine.calculate_jaccard_similarity(self.sample_docs, 0)
        
        for _, sim in results:
            self.assertGreaterEqual(sim, 0.0)
            self.assertLessEqual(sim, 1.0)


class TestTopTerms(unittest.TestCase):
    """Tests para funciones de análisis de términos"""
    
    def setUp(self):
        self.engine = SimilarityEngine()
        self.docs = ["gato gato perro", "perro pez", "carro moto"]
    
    def test_get_top_terms(self):
        """Test: obtener términos principales"""
        _, matrix, terms = self.engine.calculate_tfidf_cosine_similarity(
            self.docs, 0
        )
        
        top_terms = self.engine.get_top_terms_tfidf(matrix, terms, 0, top_n=5)
        
        self.assertIsInstance(top_terms, list)
        self.assertLessEqual(len(top_terms), 5)
        
        # Verificar estructura
        for term, weight in top_terms:
            self.assertIsInstance(term, str)
            self.assertIsInstance(weight, float)


if __name__ == '__main__':
    print("=" * 60)
    print("EJECUTANDO TESTS: test_similarity.py")
    print("=" * 60)
    unittest.main(verbosity=2)
