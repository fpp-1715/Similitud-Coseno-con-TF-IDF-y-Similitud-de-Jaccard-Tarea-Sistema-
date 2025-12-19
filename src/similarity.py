"""
Módulo para cálculo de similitud entre documentos
Implementa: TF-IDF con similitud coseno y Jaccard
Usa scikit-learn según recomendación del profesor
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Stopwords en español - palabras sin significado relevante
SPANISH_STOPWORDS = {
    # Artículos
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    # Preposiciones
    'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante',
    'en', 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según',
    'sin', 'so', 'sobre', 'tras', 'versus', 'vía',
    # Conjunciones
    'y', 'e', 'ni', 'que', 'o', 'u', 'pero', 'mas', 'sino', 'aunque',
    'si', 'como', 'porque', 'pues', 'luego', 'conque',
    # Pronombres
    'yo', 'tú', 'él', 'ella', 'nosotros', 'nosotras', 'vosotros', 'vosotras',
    'ellos', 'ellas', 'usted', 'ustedes', 'me', 'te', 'se', 'lo', 'le',
    'nos', 'os', 'les', 'mi', 'tu', 'su', 'nuestro', 'vuestro', 'suyo',
    'mío', 'tuyo', 'este', 'ese', 'aquel', 'esta', 'esa', 'aquella',
    'esto', 'eso', 'aquello', 'estos', 'esos', 'aquellos', 'estas', 'esas',
    'aquellas', 'quien', 'quienes', 'cual', 'cuales', 'donde', 'cuando',
    # Adverbios comunes
    'muy', 'más', 'menos', 'poco', 'mucho', 'demasiado', 'bastante',
    'además', 'también', 'tampoco', 'sí', 'no', 'nunca', 'siempre',
    'jamás', 'aquí', 'ahí', 'allí', 'acá', 'allá', 'cerca', 'lejos',
    'hoy', 'ayer', 'mañana', 'ahora', 'antes', 'después', 'luego',
    'todavía', 'aún', 'ya', 'bien', 'mal', 'así', 'tal', 'tanto',
    # Verbos auxiliares y comunes (infinitivo y conjugaciones frecuentes)
    'ser', 'estar', 'haber', 'tener', 'hacer', 'poder', 'deber',
    'ir', 'dar', 'saber', 'decir', 'ver', 'poner', 'querer',
    'es', 'son', 'está', 'están', 'fue', 'fueron', 'sea', 'sean',
    'ha', 'han', 'había', 'habían', 'hay', 'he', 'hemos',
    'tiene', 'tienen', 'tenía', 'tenían', 'tuvo', 'tuvieron',
    'hace', 'hacen', 'hizo', 'hicieron', 'haga', 'hagan',
    'puede', 'pueden', 'pudo', 'pudieron', 'pueda', 'puedan',
    'debe', 'deben', 'debía', 'debían', 'debió', 'debieron',
    'va', 'van', 'fue', 'fueron', 'iba', 'iban', 'vaya', 'vayan',
    'da', 'dan', 'dio', 'dieron', 'daba', 'daban',
    'sabe', 'saben', 'sabía', 'sabían', 'supo', 'supieron',
    'dice', 'dicen', 'dijo', 'dijeron', 'decía', 'decían',
    've', 'ven', 'vio', 'vieron', 'veía', 'veían',
    'pone', 'ponen', 'puso', 'pusieron', 'ponía', 'ponían',
    'quiere', 'quieren', 'quiso', 'quisieron', 'quería', 'querían',
    # Otros
    'del', 'al', 'otro', 'otra', 'otros', 'otras', 'mismo', 'misma',
    'todo', 'toda', 'todos', 'todas', 'cada', 'algo', 'nada', 'alguien',
    'nadie', 'alguno', 'ninguno', 'alguna', 'ninguna', 'varios', 'varias',
    'cualquier', 'cualquiera', 'cualesquiera', 'tal', 'tales'
}


class SimilarityEngine:
    """
    Motor de cálculo de similitud entre documentos
    """
    
    def __init__(self):
        """Inicializa el motor de similitud"""
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None
    
    def calculate_tfidf_cosine_similarity(
        self, 
        documents: List[str], 
        query_idx: int
    ) -> Tuple[List[Tuple[int, float]], np.ndarray, List[str]]:
        """
        Calcula la similitud coseno usando representación TF-IDF
        
        Args:
            documents: Lista de textos de documentos
            query_idx: Índice del documento consulta
            
        Returns:
            Tupla con:
            - Lista de (índice, similitud) ordenada de mayor a menor
            - Matriz TF-IDF
            - Lista de nombres de características (términos)
        """
        # Crear vectorizador TF-IDF con stopwords en español
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(SPANISH_STOPWORDS),  # Eliminar palabras vacías
            max_features=5000,  # Limitar características para eficiencia
            ngram_range=(1, 1),  # Solo unigramas
            min_df=1  # Mínimo 1 documento debe contener el término
        )
        
        # Construir matriz término-documento TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        
        # Obtener vector del documento consulta
        query_vector = self.tfidf_matrix[query_idx]
        
        # Calcular similitud coseno con todos los documentos
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Crear lista de (índice, similitud) ordenada
        results = [(i, sim) for i, sim in enumerate(similarities)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results, self.tfidf_matrix, self.feature_names
    
    def calculate_jaccard_similarity(
        self, 
        documents: List[str], 
        query_idx: int
    ) -> List[Tuple[int, float]]:
        """
        Calcula la similitud de Jaccard entre el documento consulta y el resto
        Filtra stopwords para considerar solo términos significativos
        
        Args:
            documents: Lista de textos de documentos
            query_idx: Índice del documento consulta
            
        Returns:
            Lista de (índice, similitud) ordenada de mayor a menor
        """
        # Obtener conjunto de términos del documento consulta (sin stopwords)
        query_terms = set(documents[query_idx].lower().split())
        query_terms = query_terms - SPANISH_STOPWORDS  # Eliminar palabras vacías
        
        results = []
        
        for i, doc in enumerate(documents):
            doc_terms = set(doc.lower().split())
            doc_terms = doc_terms - SPANISH_STOPWORDS  # Eliminar palabras vacías
            
            # Calcular Jaccard: |A ∩ B| / |A ∪ B|
            intersection = len(query_terms & doc_terms)
            union = len(query_terms | doc_terms)
            
            similarity = intersection / union if union > 0 else 0.0
            results.append((i, similarity))
        
        # Ordenar por similitud descendente
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_top_terms_tfidf(
        self, 
        tfidf_matrix: np.ndarray, 
        feature_names: List[str], 
        doc_idx: int, 
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Obtiene los términos más relevantes (mayor peso TF-IDF) de un documento
        
        Args:
            tfidf_matrix: Matriz TF-IDF
            feature_names: Lista de nombres de términos
            doc_idx: Índice del documento
            top_n: Número de términos a retornar
            
        Returns:
            Lista de (término, peso_tfidf)
        """
        # Obtener vector del documento
        doc_vector = tfidf_matrix[doc_idx].toarray().flatten()
        
        # Obtener índices de los términos ordenados por peso
        top_indices = doc_vector.argsort()[::-1][:top_n]
        
        # Crear lista de (término, peso)
        top_terms = [
            (feature_names[idx], doc_vector[idx]) 
            for idx in top_indices 
            if doc_vector[idx] > 0
        ]
        
        return top_terms
    
    def get_shared_important_terms(
        self, 
        tfidf_matrix: np.ndarray, 
        feature_names: List[str], 
        doc1_idx: int, 
        doc2_idx: int, 
        top_n: int = 10
    ) -> List[Tuple[str, float, float]]:
        """
        Obtiene los términos importantes compartidos entre dos documentos
        
        Args:
            tfidf_matrix: Matriz TF-IDF
            feature_names: Lista de nombres de términos
            doc1_idx: Índice del primer documento
            doc2_idx: Índice del segundo documento
            top_n: Número de términos a retornar
            
        Returns:
            Lista de (término, peso_doc1, peso_doc2)
        """
        # Obtener vectores de ambos documentos
        vec1 = tfidf_matrix[doc1_idx].toarray().flatten()
        vec2 = tfidf_matrix[doc2_idx].toarray().flatten()
        
        # Encontrar términos presentes en ambos documentos
        shared_terms = []
        for i, term in enumerate(feature_names):
            if vec1[i] > 0 and vec2[i] > 0:
                # Usar el mínimo peso como indicador de relevancia compartida
                combined_weight = min(vec1[i], vec2[i])
                shared_terms.append((term, vec1[i], vec2[i], combined_weight))
        
        # Ordenar por peso combinado
        shared_terms.sort(key=lambda x: x[3], reverse=True)
        
        # Retornar solo término y pesos individuales
        return [(t[0], t[1], t[2]) for t in shared_terms[:top_n]]
    
    def get_term_document_matrix_info(
        self, 
        tfidf_matrix: np.ndarray, 
        feature_names: List[str], 
        doc_names: List[str]
    ) -> Dict:
        """
        Obtiene información sobre la matriz término-documento
        
        Args:
            tfidf_matrix: Matriz TF-IDF
            feature_names: Lista de nombres de términos
            doc_names: Lista de nombres de documentos
            
        Returns:
            Diccionario con información de la matriz
        """
        return {
            "num_documents": tfidf_matrix.shape[0],
            "num_terms": tfidf_matrix.shape[1],
            "sparsity": 1 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])),
            "terms_sample": feature_names[:20],
            "documents": doc_names
        }
    
    def calculate_pairwise_similarities(
        self, 
        documents: List[str], 
        method: str = "tfidf"
    ) -> np.ndarray:
        """
        Calcula la matriz de similitud entre todos los pares de documentos
        Filtra stopwords para ambos métodos
        
        Args:
            documents: Lista de textos de documentos
            method: Método a usar ("tfidf" o "jaccard")
            
        Returns:
            Matriz de similitud (n x n)
        """
        n = len(documents)
        similarity_matrix = np.zeros((n, n))
        
        if method == "tfidf":
            vectorizer = TfidfVectorizer(stop_words=list(SPANISH_STOPWORDS))
            tfidf_matrix = vectorizer.fit_transform(documents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        
        elif method == "jaccard":
            # Convertir documentos a conjuntos de términos (sin stopwords)
            term_sets = [set(doc.lower().split()) - SPANISH_STOPWORDS for doc in documents]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        similarity_matrix[i][j] = 1.0
                    else:
                        intersection = len(term_sets[i] & term_sets[j])
                        union = len(term_sets[i] | term_sets[j])
                        similarity_matrix[i][j] = intersection / union if union > 0 else 0.0
        
        return similarity_matrix
