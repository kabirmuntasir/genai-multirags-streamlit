import streamlit as st
from rag.chroma_azure_rag import ChromaAzureRAG
import tempfile
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from rag.azure_search_rag import AzureSearchRAG
from rag.light_rag import LightRAG

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_rag_systems():
    collection_name = "docs-index"
    return {
        "Chroma Azure RAG": ChromaAzureRAG(
            persist_directory=os.getenv("PERSIST_DIRECTORY"),
            collection_name="chroma_docs"
        ),
        "Azure Search RAG": AzureSearchRAG(
            collection_name=collection_name
        ),
        "Light RAG": LightRAG(
            persist_directory="./document_store/light_db",
            collection_name="light_docs"
        )
    }

def main():
    st.set_page_config(layout="wide")
    st.title("Azure-Powered RAG System")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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

    # Chat interface
    st.header("Chat with Documents")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for i, doc in enumerate(message["sources"], 1):
                        source_file = doc.metadata.get('source')
                        page = doc.metadata.get('page', 'N/A')
                        filename = doc.metadata.get('filename', Path(source_file).name if source_file else 'Unknown')
                        
                        st.markdown(f"**Source {i}:** {filename}")
                        st.markdown(f"**Page:** {page}")
                        if source_file:
                            st.markdown(f"[Open Source File]({source_file})")
                        st.markdown("---")
                        st.markdown("**Relevant Content:**")
                        st.markdown(doc.page_content)

    # Query input
    query = st.chat_input("Ask a question about your documents:")
    
    if query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                try:
                    response = rag_systems[selected_rag].query(query, k=k_value)
                    st.markdown(response["answer"])
                    
                    with st.expander("View Sources"):
                        for i, doc in enumerate(response["sources"], 1):
                            source_file = doc.metadata.get('source')
                            page = doc.metadata.get('page', 'N/A')
                            filename = doc.metadata.get('filename', Path(source_file).name if source_file else 'Unknown')
                            
                            st.markdown(f"**Source {i}:** {filename}")
                            st.markdown(f"**Page:** {page}")
                            if source_file:
                                st.markdown(f"[Open Source File]({source_file})")
                            st.markdown("---")
                            st.markdown("**Relevant Content:**")
                            st.markdown(doc.page_content)
                    
                    # Save assistant response with sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()