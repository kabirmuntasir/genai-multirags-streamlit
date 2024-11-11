# Azure-Powered RAG System

This repository contains an implementation of a Retrieval-Augmented Generation (RAG) system powered by Azure services. The system supports multiple RAG implementations, including Chroma Azure RAG, Azure Search RAG, and Light RAG. It provides a Streamlit-based interface for document upload, querying, and performance evaluation.

## Features

- **Document Upload**: Upload documents in PDF, TXT, or MD format.
- **Query Interface**: Ask questions about the uploaded documents and get answers with relevant sources.
- **Multiple RAG Implementations**: Choose between Chroma Azure RAG, Azure Search RAG, and Light RAG.
- **Performance Evaluation**: Evaluate the performance of different RAG systems using metrics like precision, recall, F1-score, and response time.
- **Visualization**: Display performance metrics in a table and bar chart.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/azure-powered-rag-system.git
    cd azure-powered-rag-system
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    Create a `.env` file in the root directory with the following content:
    ```env
    AZURE_OPENAI_ENDPOINT="https://your-openai-endpoint.openai.azure.com"
    AZURE_OPENAI_API_KEY="your-openai-api-key"
    AZURE_DEPLOYMENT_NAME="your-deployment-name"
    AZURE_EMBEDDING_DEPLOYMENT_NAME="text-embedding-ada-002"
    OPENAI_API_VERSION="2023-05-15"
    PERSIST_DIRECTORY="./document_store/chroma_db"
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=200
    AZURE_SEARCH_SERVICE_ENDPOINT="https://your-azure-search-endpoint.search.windows.net"
    AZURE_SEARCH_ADMIN_KEY="your-azure-search-admin-key"
    AZURE_SEARCH_INDEX_NAME="docs-index"
    ```

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Upload Documents**:
    - Navigate to the "Document Upload" section in the sidebar.
    - Upload documents in PDF, TXT, or MD format.

3. **Query Documents**:
    - Use the chat interface to ask questions about the uploaded documents.
    - View the answers along with relevant sources.

4. **Preprocess Documents**:
    - Before querying, you need to preprocess the documents for the selected RAG system.
    - Run the following command to preprocess documents for different RAG systems:

    **Chroma Azure RAG**:
    ```bash
    python utils/preprocess.py --rag_type chroma --delete yes --collection_name chroma_docs
    ```

    **Azure Search RAG**:
    ```bash
    python utils/preprocess.py --rag_type azure_search --delete yes --collection_name docs-index
    ```

    **Light RAG**:
    ```bash
    python utils/preprocess.py --rag_type light --delete yes --collection_name light_docs
    ```

5. **Evaluate Performance**:
    - Navigate to the "Performance Evaluation" section in the sidebar.
    - Click the "Evaluate Performance" button to evaluate the performance of different RAG systems.
    - View the performance metrics in a table and bar chart.

## File Structure

- [app.py](http://_vscodecontentref_/0): Main Streamlit app file.
- [evaluate_rags.py](http://_vscodecontentref_/1): Script to evaluate the performance of different RAG systems.
- [rag](http://_vscodecontentref_/2): Directory containing RAG implementations.
  - `chroma_azure_rag.py`: Chroma Azure RAG implementation.
  - `azure_search_rag.py`: Azure Search RAG implementation.
  - `light_rag.py`: Light RAG implementation.
- [utils](http://_vscodecontentref_/3): Directory containing utility scripts.
  - `preprocess.py`: Script to preprocess documents for RAG systems.
- [.env](http://_vscodecontentref_/4): Environment variables file (not included in the repository).
- [.gitignore](http://_vscodecontentref_/5): Git ignore file.
- [requirements.txt](http://_vscodecontentref_/6): Python dependencies file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.