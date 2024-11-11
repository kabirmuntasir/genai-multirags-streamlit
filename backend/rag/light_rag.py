# rag/light_rag.py
import logging
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from langchain.schema import BaseRetriever

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightRAG:
    def __init__(
        self,
        persist_directory="./document_store/light_db",
        chunk_size=1000,
        chunk_overlap=200,
        collection_name=None
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name or "light_docs"
        
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
        
        # Initialize vector stores
        self.dense_store = Chroma(
            collection_name=f"{self.collection_name}_dense",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Initialize sparse indexing
        self.tfidf = TfidfVectorizer()
        self.bm25 = None
        self.documents = []
        
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
            
            # Store documents in dense index
            self.dense_store.add_documents(texts)
            
            # Store for sparse index
            self.documents.extend(texts)
            
            # Get all document content for BM25
            all_docs = [doc.page_content for doc in self.documents]
            
            # Tokenize corpus for BM25
            tokenized_corpus = [doc.lower().split() for doc in all_docs]
            logger.info(f"Tokenized corpus size: {len(tokenized_corpus)}")
            
            # Initialize BM25 with full corpus
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index initialized")
            
            return len(texts)
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def query(self, query_text, k=4):
        try:
            # Dense retrieval
            dense_results = self.dense_store.similarity_search(query_text, k=k)
            
            # Ensure we have BM25 index
            if not self.bm25 or not self.documents:
                logger.warning("BM25 index not initialized, using only dense retrieval")
                combined_docs = dense_results
            else:
                # Sparse retrieval (BM25)
                tokenized_query = query_text.lower().split()
                bm25_scores = self.bm25.get_scores(tokenized_query)
                top_sparse_idx = np.argsort(bm25_scores)[-k:][::-1]
                sparse_results = [self.documents[i] for i in top_sparse_idx]
                
                # Combine results
                combined_docs = self._adaptive_selection(
                    query_text,
                    dense_results,
                    sparse_results,
                    k=k
                )

            # Create proper retriever
            class CombinedRetriever(BaseRetriever):
                def __init__(self, documents):
                    super().__init__()
                    self._documents = documents
                    
                def get_relevant_documents(self, query):
                    return self._documents
                    
                async def aget_relevant_documents(self, query):
                    return self._documents

            retriever = CombinedRetriever(combined_docs)
            
            # Create QA chain
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
                "answer": result.get("result", "No answer found") + "\n\n*Powered by Light RAG (Nano DB)*",
                "sources": result.get("source_documents", [])
            }
            
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            raise

    def _adaptive_selection(self, query, dense_docs, sparse_docs, k=4):
        """Adaptive passage selection combining dense and sparse results"""
        # Calculate relevance scores
        dense_scores = [self._calculate_relevance(query, doc.page_content) for doc in dense_docs]
        sparse_scores = [self._calculate_relevance(query, doc.page_content) for doc in sparse_docs]
        
        # Combine and sort results
        all_docs = list(zip(dense_docs + sparse_docs, dense_scores + sparse_scores))
        all_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in all_docs[:k]]
    
    def _calculate_relevance(self, query, document):
        """Calculate relevance score between query and document"""
        query_embedding = self.embeddings.embed_query(query)
        doc_embedding = self.embeddings.embed_query(document)
        return np.dot(query_embedding, doc_embedding)

    def get_collection_stats(self):
        return {
            "total_documents": len(self.documents),
            "collection_name": self.collection_name
        }

    def delete_collection(self):
        logger.info(f"Deleting collection: {self.collection_name}")
        self.dense_store.delete_collection()
        self.documents = []
        self.bm25 = None
        self.dense_store = Chroma(
            collection_name=f"{self.collection_name}_dense",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )