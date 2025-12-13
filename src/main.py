import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from processor import DocumentProcessor
from similarity import SimilarityEngine

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Búsqueda por Similitud - Equipo 8",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Sistema de Búsqueda de Documentos por Similitud")
st.markdown("**Tarea 8** - Búsqueda de documentos similares usando TF-IDF Coseno y Jaccard")

# Inicializar el procesador y motor de similitud
@st.cache_resource
def init_processor():
    return DocumentProcessor()

@st.cache_resource
def init_similarity_engine():
    return SimilarityEngine()

processor = init_processor()
similarity_engine = init_similarity_engine()

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

# Ruta de documentos - CONFIGURABLE
st.sidebar.subheader("📁 Ubicación de Documentos")

# Ruta por defecto
default_path = r"E:\Ciber\sistema\Tarea Extraclase 2\proyecto_equipo8\data\documentos"

# Opción para elegir método de selección
folder_method = st.sidebar.radio(
    "Método de selección:",
    ["Ruta predeterminada", "Escribir ruta manualmente"],
    help="Elige cómo quieres seleccionar la carpeta de documentos"
)

if folder_method == "Ruta predeterminada":
    documents_folder = default_path
    st.sidebar.info(f"📂 Usando: `{documents_folder}`")
else:
    documents_folder = st.sidebar.text_input(
        "✏️ Escribe la ruta completa de la carpeta:",
        value=default_path,
        help="Ejemplo: C:\\Users\\MiUsuario\\Documentos\\MisArchivos"
    )

# Mostrar ruta actual
st.sidebar.markdown(f"**Ruta actual:** `{documents_folder}`")

# Verificar si la carpeta existe
if os.path.exists(documents_folder):
    st.sidebar.success("✅ Carpeta encontrada")
else:
    st.sidebar.error("❌ La carpeta no existe")

# Cargar documentos
if st.sidebar.button("🔄 Cargar/Recargar Documentos", type="primary"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data
def load_docs(folder):
    return processor.load_documents(folder)

if os.path.exists(documents_folder):
    documents = load_docs(documents_folder)
    
    if documents and len(documents) > 0:
        st.sidebar.success(f"✅ {len(documents)} documento(s) cargado(s)")
        
        # Mostrar lista de documentos cargados
        with st.sidebar.expander("📋 Documentos cargados"):
            for doc_name in documents.keys():
                st.write(f"• {doc_name}")
        
        # ===== SECCIÓN PRINCIPAL =====
        st.header("🔍 Búsqueda por Similitud")
        
        doc_names = list(documents.keys())
        
        # Seleccionar documento consulta
        query_doc = st.selectbox(
            "📌 Selecciona el documento consulta",
            doc_names,
            help="Este documento se comparará con todos los demás"
        )
        
        # Número de resultados a mostrar
        num_results = st.slider(
            "Número de resultados a mostrar",
            min_value=1,
            max_value=max(1, len(doc_names) - 1),
            value=min(5, len(doc_names) - 1)
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_tfidf = st.checkbox("📊 Similitud TF-IDF Coseno", value=True)
        with col2:
            show_jaccard = st.checkbox("📊 Similitud Jaccard", value=True)
        
        if st.button("🚀 Buscar Documentos Similares", type="primary"):
            if len(documents) < 2:
                st.warning("⚠️ Se necesitan al menos 2 documentos para comparar")
            else:
                # Preparar documentos
                doc_list = list(documents.values())
                doc_names_list = list(documents.keys())
                query_idx = doc_names_list.index(query_doc)
                
                st.markdown("---")
                
                # ===== RESULTADOS TF-IDF COSENO =====
                if show_tfidf:
                    st.subheader("📈 Resultados: Similitud TF-IDF Coseno")
                    
                    # Calcular similitud TF-IDF
                    tfidf_results, tfidf_matrix, feature_names = similarity_engine.calculate_tfidf_cosine_similarity(
                        doc_list, query_idx
                    )
                    
                    # Crear DataFrame de resultados
                    tfidf_df = pd.DataFrame([
                        {"Documento": doc_names_list[idx], "Similitud": sim, "Porcentaje": f"{sim*100:.2f}%"}
                        for idx, sim in tfidf_results if idx != query_idx
                    ][:num_results])
                    
                    col_table, col_chart = st.columns([1, 1])
                    
                    with col_table:
                        st.markdown("**📋 Ranking de Similitud**")
                        st.dataframe(
                            tfidf_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col_chart:
                        st.markdown("**📊 Gráfico de Similitudes**")
                        if not tfidf_df.empty:
                            fig_tfidf = px.bar(
                                tfidf_df,
                                x="Similitud",
                                y="Documento",
                                orientation='h',
                                color="Similitud",
                                color_continuous_scale="Blues",
                                title="Similitud TF-IDF Coseno"
                            )
                            fig_tfidf.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig_tfidf, use_container_width=True)
                    
                    # Mostrar términos más relevantes
                    st.markdown("**🔤 Términos más relevantes del documento consulta**")
                    top_terms = similarity_engine.get_top_terms_tfidf(
                        tfidf_matrix, feature_names, query_idx, top_n=15
                    )
                    
                    terms_df = pd.DataFrame(top_terms, columns=["Término", "Peso TF-IDF"])
                    
                    col_terms, col_terms_chart = st.columns([1, 1])
                    
                    with col_terms:
                        st.dataframe(terms_df, use_container_width=True, hide_index=True)
                    
                    with col_terms_chart:
                        fig_terms = px.bar(
                            terms_df,
                            x="Peso TF-IDF",
                            y="Término",
                            orientation='h',
                            color="Peso TF-IDF",
                            color_continuous_scale="Greens",
                            title="Términos más importantes"
                        )
                        fig_terms.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_terms, use_container_width=True)
                    
                    # Términos compartidos con documentos similares
                    if tfidf_results and len(tfidf_results) > 1:
                        most_similar_idx = tfidf_results[0][0] if tfidf_results[0][0] != query_idx else tfidf_results[1][0]
                        shared_terms = similarity_engine.get_shared_important_terms(
                            tfidf_matrix, feature_names, query_idx, most_similar_idx, top_n=10
                        )
                        
                        if shared_terms:
                            st.markdown(f"**🔗 Términos compartidos con '{doc_names_list[most_similar_idx]}'**")
                            shared_df = pd.DataFrame(shared_terms, columns=["Término", "Peso Doc. Consulta", "Peso Doc. Similar"])
                            st.dataframe(shared_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # ===== RESULTADOS JACCARD =====
                if show_jaccard:
                    st.subheader("📈 Resultados: Similitud Jaccard")
                    
                    # Calcular similitud Jaccard
                    jaccard_results = similarity_engine.calculate_jaccard_similarity(
                        doc_list, query_idx
                    )
                    
                    # Crear DataFrame de resultados
                    jaccard_df = pd.DataFrame([
                        {"Documento": doc_names_list[idx], "Similitud": sim, "Porcentaje": f"{sim*100:.2f}%"}
                        for idx, sim in jaccard_results if idx != query_idx
                    ][:num_results])
                    
                    col_table_j, col_chart_j = st.columns([1, 1])
                    
                    with col_table_j:
                        st.markdown("**📋 Ranking de Similitud**")
                        st.dataframe(
                            jaccard_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col_chart_j:
                        st.markdown("**📊 Gráfico de Similitudes**")
                        if not jaccard_df.empty:
                            fig_jaccard = px.bar(
                                jaccard_df,
                                x="Similitud",
                                y="Documento",
                                orientation='h',
                                color="Similitud",
                                color_continuous_scale="Oranges",
                                title="Similitud Jaccard"
                            )
                            fig_jaccard.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig_jaccard, use_container_width=True)
                    
                    # Mostrar estadísticas de conjuntos
                    st.markdown("**📊 Análisis de conjuntos de términos**")
                    
                    query_terms = set(doc_list[query_idx].lower().split())
                    
                    if jaccard_results:
                        most_similar_j_idx = jaccard_results[0][0] if jaccard_results[0][0] != query_idx else jaccard_results[1][0]
                        similar_terms = set(doc_list[most_similar_j_idx].lower().split())
                        
                        intersection = query_terms & similar_terms
                        union = query_terms | similar_terms
                        only_query = query_terms - similar_terms
                        only_similar = similar_terms - query_terms
                        
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        
                        with col_stats1:
                            st.metric("Términos en consulta", len(query_terms))
                        with col_stats2:
                            st.metric("Términos en común", len(intersection))
                        with col_stats3:
                            st.metric("Términos en unión", len(union))
                        
                        with st.expander(f"🔍 Ver términos en común con '{doc_names_list[most_similar_j_idx]}'"):
                            if intersection:
                                st.write(", ".join(sorted(list(intersection))[:50]))
                            else:
                                st.write("No hay términos en común")
                
                # ===== COMPARACIÓN DE MÉTODOS =====
                if show_tfidf and show_jaccard:
                    st.markdown("---")
                    st.subheader("📊 Comparación de Métodos")
                    
                    # Crear DataFrame comparativo
                    comparison_data = []
                    for i, doc_name in enumerate(doc_names_list):
                        if i != query_idx:
                            tfidf_sim = next((sim for idx, sim in tfidf_results if idx == i), 0)
                            jaccard_sim = next((sim for idx, sim in jaccard_results if idx == i), 0)
                            comparison_data.append({
                                "Documento": doc_name,
                                "TF-IDF Coseno": tfidf_sim,
                                "Jaccard": jaccard_sim
                            })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Gráfico comparativo
                    fig_comparison = go.Figure()
                    fig_comparison.add_trace(go.Bar(
                        name='TF-IDF Coseno',
                        x=comparison_df['Documento'],
                        y=comparison_df['TF-IDF Coseno'],
                        marker_color='steelblue'
                    ))
                    fig_comparison.add_trace(go.Bar(
                        name='Jaccard',
                        x=comparison_df['Documento'],
                        y=comparison_df['Jaccard'],
                        marker_color='darkorange'
                    ))
                    fig_comparison.update_layout(
                        barmode='group',
                        title='Comparación de Similitudes por Método',
                        xaxis_title='Documento',
                        yaxis_title='Similitud',
                        height=400
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Tabla comparativa
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # ===== SECCIÓN DE EXPLORACIÓN =====
        st.markdown("---")
        st.header("🔎 Explorar Documento")
        
        explore_doc = st.selectbox(
            "Selecciona un documento para explorar",
            doc_names,
            key="explore"
        )
        
        if st.button("📖 Ver contenido"):
            with st.expander(f"Contenido de '{explore_doc}'", expanded=True):
                content = documents[explore_doc]
                st.text_area("", content[:5000] + ("..." if len(content) > 5000 else ""), height=300)
                st.caption(f"Total de caracteres: {len(content)}")
    
    else:
        st.warning("⚠️ No se encontraron documentos .txt o .pdf en la carpeta especificada")
        st.info("💡 Agrega archivos .txt o .pdf en la carpeta `data/documentos/`")
else:
    st.error(f"❌ La carpeta '{documents_folder}' no existe")
    st.info("💡 Crea la carpeta y agrega documentos de prueba")

# Footer
st.markdown("---")
st.markdown("*Desarrollado por **Equipo 8** - Tarea Extraclase 2*")
