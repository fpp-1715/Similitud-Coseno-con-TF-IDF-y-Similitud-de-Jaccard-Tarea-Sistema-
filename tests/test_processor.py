"""
================================================================================
TESTS UNITARIOS: test_processor.py
DESCRIPCIÓN: Pruebas para el módulo de procesamiento de documentos
AUTOR: Equipo 8
================================================================================
"""

import unittest
import sys
import os

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from processor import DocumentProcessor


class TestDocumentProcessor(unittest.TestCase):
    """Tests para la clase DocumentProcessor"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.processor = DocumentProcessor()
    
    def test_preprocess_lowercase(self):
        """Test: conversión a minúsculas"""
        texto = "HOLA MUNDO"
        resultado = self.processor.preprocess_text(texto)
        self.assertEqual(resultado, "hola mundo")
    
    def test_preprocess_punctuation(self):
        """Test: eliminación de puntuación"""
        texto = "¡Hola, mundo!"
        resultado = self.processor.preprocess_text(texto)
        self.assertEqual(resultado, "hola mundo")
    
    def test_preprocess_multiple_spaces(self):
        """Test: normalización de espacios"""
        texto = "hola    mundo   test"
        resultado = self.processor.preprocess_text(texto)
        self.assertEqual(resultado, "hola mundo test")
    
    def test_preprocess_empty_string(self):
        """Test: manejo de cadena vacía"""
        resultado = self.processor.preprocess_text("")
        self.assertEqual(resultado, "")
    
    def test_tokenize(self):
        """Test: tokenización de texto"""
        texto = "hola mundo test"
        tokens = self.processor.tokenize(texto)
        self.assertEqual(tokens, ["hola", "mundo", "test"])
    
    def test_tokenize_empty(self):
        """Test: tokenización de cadena vacía"""
        tokens = self.processor.tokenize("")
        self.assertEqual(tokens, [""])
    
    def test_supported_extensions(self):
        """Test: extensiones soportadas"""
        self.assertIn('.txt', self.processor.supported_extensions)
        self.assertIn('.pdf', self.processor.supported_extensions)
    
    def test_get_vocabulary(self):
        """Test: obtención de vocabulario"""
        documents = {
            "doc1": "gato perro",
            "doc2": "perro pez"
        }
        vocab = self.processor.get_vocabulary(documents)
        self.assertEqual(vocab, {"gato", "perro", "pez"})


class TestPreprocessingOptions(unittest.TestCase):
    """Tests para opciones de preprocesamiento"""
    
    def setUp(self):
        self.processor = DocumentProcessor()
    
    def test_keep_uppercase(self):
        """Test: mantener mayúsculas"""
        texto = "HOLA Mundo"
        resultado = self.processor.preprocess_text(texto, lowercase=False)
        self.assertIn("HOLA", resultado)
    
    def test_remove_numbers(self):
        """Test: eliminar números"""
        texto = "hola 123 mundo"
        resultado = self.processor.preprocess_text(texto, remove_numbers=True)
        self.assertNotIn("123", resultado)
    
    def test_keep_numbers(self):
        """Test: mantener números"""
        texto = "hola 123 mundo"
        resultado = self.processor.preprocess_text(texto, remove_numbers=False)
        self.assertIn("123", resultado)


if __name__ == '__main__':
    print("=" * 60)
    print("EJECUTANDO TESTS: test_processor.py")
    print("=" * 60)
    unittest.main(verbosity=2)
