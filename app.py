import streamlit as st
from rag.chroma_azure_rag import ChromaAzureRAG
import tempfile
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_rag_systems():
    collection_name = "chroma_docs"
    persist_directory = os.getenv("PERSIST_DIRECTORY")  # Remove fallback
    return {
        "Chroma Azure RAG": ChromaAzureRAG(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    }

def main():
    st.set_page_config(layout="wide")
    st.title("Azure-Powered RAG System")
    
    rag_systems = get_rag_systems()
    
    with st.sidebar:
        st.header("Configuration")
        selected_rag = st.selectbox(
            "Select RAG System",
            options=list(rag_systems.keys())
        )
        
        k_value = st.slider("Number of relevant chunks", 1, 10, 4)
        
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf", "txt", "md"]
        )
        
        if uploaded_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                chunks = rag_systems[selected_rag].process_document(
                    tmp_file_path,
                    metadata={"filename": uploaded_file.name}
                )
                os.unlink(tmp_file_path)
                st.success(f"Successfully processed {chunks} chunks from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        if st.button("Show Collection Stats"):
            try:
                stats = rag_systems[selected_rag].get_collection_stats()
                st.json(stats)
            except Exception as e:
                st.error(f"Error getting stats: {str(e)}")
            
        if st.button("Clear Collection"):
            try:
                rag_systems[selected_rag].delete_collection()
                st.success("Collection cleared successfully!")
            except Exception as e:
                st.error(f"Error clearing collection: {str(e)}")

    # Main content area
    st.header("Query Documents")
    query = st.text_input("Enter your question:")
    
    if query:
        try:
            with st.spinner("Searching..."):
                response = rag_systems[selected_rag].query(query, k=k_value)
                
                st.markdown("### Answer")
                st.write(response["answer"])
                
                st.markdown("### Sources")
                for i, doc in enumerate(response["sources"], 1):
                    st.write(f"{i}. {doc.metadata.get('filename', doc.metadata.get('source', 'Unknown source'))}")
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()