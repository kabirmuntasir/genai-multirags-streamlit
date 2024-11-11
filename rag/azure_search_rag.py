# rag/azure_search_rag.py
import logging
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import AzureSearch
from pathlib import Path
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureSearchRAG:
    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=200,
        collection_name=None  # Used as index name
    ):
        self.collection_name = collection_name or os.getenv("AZURE_SEARCH_INDEX_NAME")
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize vector store
        self.vector_store = AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_SEARCH_ADMIN_KEY"),
            index_name=self.collection_name,
            embedding_function=self.embeddings.embed_query
        )

    def process_document(self, file_path, metadata=None):
        file_path = Path(file_path)
        try:
            logger.info(f"Processing file: {file_path}")
            
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() in ['.txt', '.md']:
                loader = UnstructuredFileLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents")
            
            for doc in documents:
                doc.metadata = metadata or {}
                doc.metadata['source'] = str(file_path)
                
            texts = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(texts)} chunks")
            
            self.vector_store.add_documents(texts)
            return len(texts)
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def query(self, query_text, k=4):
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"size": k}
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=AzureChatOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("OPENAI_API_VERSION")
                ),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            result = qa_chain.invoke({"query": query_text})
            return {
                "query": query_text,
                "answer": result.get("result", "No answer found") + "\n\n*Powered by Regular RAG (Azure AI Search)*",
                "sources": result.get("source_documents", [])
            }
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            raise

    def get_collection_stats(self):
        stats = self.vector_store.client.get_document_count(self.collection_name)
        return {
            "total_documents": stats,
            "collection_name": self.collection_name
        }

    def delete_collection(self):
        logger.info(f"Deleting index: {self.collection_name}")
        try:
            # Initialize admin client for index management
            admin_client = SearchIndexClient(
                endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
                credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
            )
            
            # Delete the index if it exists
            try:
                admin_client.delete_index(self.collection_name)
                logger.info(f"Index {self.collection_name} deleted successfully")
            except Exception as e:
                if "ResourceNotFound" not in str(e):
                    raise
                logger.info(f"Index {self.collection_name} does not exist")
            
            # Reinitialize vector store with fresh index
            self.vector_store = AzureSearch(
                azure_search_endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
                azure_search_key=os.getenv("AZURE_SEARCH_ADMIN_KEY"),
                index_name=self.collection_name,
                embedding_function=self.embeddings.embed_query
            )
            logger.info(f"Vector store reinitialized with fresh index: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            raise