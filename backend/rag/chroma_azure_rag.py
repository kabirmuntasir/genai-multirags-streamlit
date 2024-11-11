import logging
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import chromadb
from pathlib import Path
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaAzureRAG:
    def __init__(
        self,
        persist_directory="./document_store/chroma_db",
        chunk_size=1000,
        chunk_overlap=200,
        collection_name=None
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name or f"chroma_docs_{str(uuid.uuid4())[:8]}"
        
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
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def process_document(self, file_path, metadata=None):
        file_path = Path(file_path)
        try:
            # Add more detailed logging
            logger.info(f"Processing file: {file_path}")
            
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() in ['.txt', '.md']:
                loader = UnstructuredFileLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents")
            
            # Ensure metadata is properly set
            for doc in documents:
                doc.metadata = metadata or {}
                doc.metadata['source'] = str(file_path)
                
            texts = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(texts)} chunks")
            
            # Add and persist documents
            self.vector_store.add_documents(texts)
            self.vector_store.persist()
            
            return len(texts)
        
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def query(self, query_text, k=4):
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
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
            
            result = qa_chain({"query": query_text})
            return {
                "query": query_text,
                "answer": result["result"]+ "\n\n*Powered by Regular RAG (Chroma DB)*",
                "sources": result["source_documents"]
            }
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            raise

    def get_collection_stats(self):
        return {
            "total_documents": len(self.vector_store.get()),
            "collection_name": self.collection_name
        }

    def delete_collection(self):
        logger.info(f"Deleting collection: {self.collection_name}")
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )