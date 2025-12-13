"""
Funciones auxiliares para la aplicación
"""

import os
from typing import Dict, List, Tuple
import pandas as pd


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Formatea un valor como porcentaje
    
    Args:
        value: Valor entre 0 y 1
        decimals: Número de decimales
        
    Returns:
        String formateado como porcentaje
    """
    return f"{value * 100:.{decimals}f}%"


def create_results_dataframe(
    results: List[Tuple[int, float]], 
    doc_names: List[str],
    exclude_idx: int = None
) -> pd.DataFrame:
    """
    Crea un DataFrame con los resultados de similitud
    
    Args:
        results: Lista de (índice, similitud)
        doc_names: Lista de nombres de documentos
        exclude_idx: Índice a excluir (documento consulta)
        
    Returns:
        DataFrame con columnas Documento, Similitud, Porcentaje
    """
    data = []
    for idx, sim in results:
        if exclude_idx is not None and idx == exclude_idx:
            continue
        data.append({
            "Documento": doc_names[idx],
            "Similitud": sim,
            "Porcentaje": format_percentage(sim)
        })
    
    return pd.DataFrame(data)


def get_similarity_interpretation(similarity: float) -> Tuple[str, str]:
    """
    Obtiene una interpretación textual de la similitud
    
    Args:
        similarity: Valor de similitud entre 0 y 1
        
    Returns:
        Tupla (interpretación, emoji)
    """
    if similarity >= 0.8:
        return "Muy alta similitud", "🟢"
    elif similarity >= 0.6:
        return "Alta similitud", "🟡"
    elif similarity >= 0.4:
        return "Similitud moderada", "🟠"
    elif similarity >= 0.2:
        return "Baja similitud", "🔴"
    else:
        return "Muy baja similitud", "⚫"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Trunca un texto si excede la longitud máxima
    
    Args:
        text: Texto a truncar
        max_length: Longitud máxima
        
    Returns:
        Texto truncado con "..." si fue necesario
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def validate_folder(folder_path: str) -> Tuple[bool, str]:
    """
    Valida que una carpeta existe y contiene documentos válidos
    
    Args:
        folder_path: Ruta de la carpeta
        
    Returns:
        Tupla (es_válida, mensaje)
    """
    if not os.path.exists(folder_path):
        return False, f"La carpeta '{folder_path}' no existe"
    
    if not os.path.isdir(folder_path):
        return False, f"'{folder_path}' no es una carpeta"
    
    valid_extensions = ['.txt', '.pdf']
    files = [f for f in os.listdir(folder_path) 
             if os.path.isfile(os.path.join(folder_path, f)) 
             and os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not files:
        return False, "No se encontraron archivos .txt o .pdf"
    
    return True, f"Se encontraron {len(files)} documento(s)"


def calculate_stats(documents: Dict[str, str]) -> Dict:
    """
    Calcula estadísticas básicas de los documentos
    
    Args:
        documents: Diccionario de documentos
        
    Returns:
        Diccionario con estadísticas
    """
    if not documents:
        return {}
    
    lengths = [len(text) for text in documents.values()]
    word_counts = [len(text.split()) for text in documents.values()]
    
    return {
        "total_documents": len(documents),
        "total_characters": sum(lengths),
        "total_words": sum(word_counts),
        "avg_characters": sum(lengths) / len(lengths),
        "avg_words": sum(word_counts) / len(word_counts),
        "min_characters": min(lengths),
        "max_characters": max(lengths),
        "min_words": min(word_counts),
        "max_words": max(word_counts)
    }
