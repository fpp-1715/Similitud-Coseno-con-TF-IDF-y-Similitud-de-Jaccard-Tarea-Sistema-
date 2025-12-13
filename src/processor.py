"""
Módulo para extracción de texto y preprocesamiento de documentos
Soporta formatos TXT y PDF
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

# Intentar importar textract (recomendado por el profesor)
try:
    import textract
    TEXTRACT_AVAILABLE = True
except ImportError:
    TEXTRACT_AVAILABLE = False

# Fallback a PyPDF2 si textract no está disponible
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


class DocumentProcessor:
    """
    Procesador de documentos para extracción y preprocesamiento de texto
    """
    
    def __init__(self):
        """Inicializa el procesador de documentos"""
        self.supported_extensions = ['.txt', '.pdf']
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """
        Extrae texto de un archivo .txt
        
        Args:
            file_path: Ruta del archivo .txt
            
        Returns:
            Contenido del archivo como string
        """
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                print(f"Error leyendo {file_path}: {e}")
                return ""
        
        return ""
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extrae texto de un archivo .pdf usando textract o PyPDF2
        
        Args:
            file_path: Ruta del archivo .pdf
            
        Returns:
            Contenido del archivo como string
        """
        # Intentar con textract primero (mejor para PDFs complejos)
        if TEXTRACT_AVAILABLE:
            try:
                text = textract.process(file_path).decode('utf-8')
                return text
            except Exception as e:
                print(f"Error con textract en {file_path}: {e}")
        
        # Fallback a PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text
            except Exception as e:
                print(f"Error con PyPDF2 en {file_path}: {e}")
        
        print(f"No se pudo extraer texto de {file_path}. Instale textract o PyPDF2.")
        return ""
    
    def extract_text(self, file_path: str) -> str:
        """
        Extrae texto de un archivo según su extensión
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Texto extraído
        """
        ext = Path(file_path).suffix.lower()
        
        if ext == '.txt':
            return self.extract_text_from_txt(file_path)
        elif ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        else:
            print(f"Formato no soportado: {ext}")
            return ""
    
    def preprocess_text(self, text: str, lowercase: bool = True, 
                       remove_punctuation: bool = True,
                       remove_numbers: bool = False) -> str:
        """
        Preprocesa el texto aplicando normalización
        
        Args:
            text: Texto a preprocesar
            lowercase: Convertir a minúsculas
            remove_punctuation: Eliminar puntuación
            remove_numbers: Eliminar números
            
        Returns:
            Texto preprocesado
        """
        if not text:
            return ""
        
        # Convertir a minúsculas
        if lowercase:
            text = text.lower()
        
        # Eliminar números si se especifica
        if remove_numbers:
            text = re.sub(r'\d+', ' ', text)
        
        # Eliminar caracteres especiales y puntuación
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Eliminar espacios múltiples y saltos de línea extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_documents(self, folder_path: str, preprocess: bool = True) -> Dict[str, str]:
        """
        Carga todos los documentos de una carpeta
        
        Args:
            folder_path: Ruta de la carpeta con documentos
            preprocess: Aplicar preprocesamiento al texto
            
        Returns:
            Diccionario {nombre_archivo: texto}
        """
        documents = {}
        
        if not os.path.exists(folder_path):
            print(f"La carpeta {folder_path} no existe")
            return documents
        
        for file_name in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_name)
            
            if not os.path.isfile(file_path):
                continue
            
            ext = Path(file_name).suffix.lower()
            
            if ext not in self.supported_extensions:
                continue
            
            text = self.extract_text(file_path)
            
            if text:
                if preprocess:
                    text = self.preprocess_text(text)
                documents[file_name] = text
                print(f"✓ Cargado: {file_name} ({len(text)} caracteres)")
        
        return documents
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokeniza el texto en palabras
        
        Args:
            text: Texto a tokenizar
            
        Returns:
            Lista de tokens
        """
        return text.split()
    
    def get_vocabulary(self, documents: Dict[str, str]) -> set:
        """
        Obtiene el vocabulario de todos los documentos
        
        Args:
            documents: Diccionario de documentos
            
        Returns:
            Conjunto de términos únicos
        """
        vocabulary = set()
        for text in documents.values():
            vocabulary.update(self.tokenize(text))
        return vocabulary
