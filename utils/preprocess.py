import sys
import os
from pathlib import Path

# Add the project directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from dotenv import load_dotenv
from rag.chroma_azure_rag import ChromaAzureRAG
from rag.light_rag import LightRAG

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from rag.azure_search_rag import AzureSearchRAG  # Add import

def process_documents(rag_type="chroma", delete="no", collection_name=None):
    try:
        if rag_type == "chroma":
            persist_directory = os.getenv("PERSIST_DIRECTORY")
            rag = ChromaAzureRAG(
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        elif rag_type == "azure_search":
            rag = AzureSearchRAG(collection_name=collection_name)
            
        elif rag_type == "light":  # Add this case
            rag = LightRAG(
                persist_directory="./document_store/light_db",
                collection_name=collection_name
            )
        else:
            raise ValueError(f"Unsupported RAG type: {rag_type}")
        
        if delete.lower() == "yes":
            logger.info(f"Deleting collection: {collection_name}")
            rag.delete_collection()
            
        input_dir = Path("./document_store/input")
        input_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        for file_path in input_dir.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.md']:
                try:
                    chunks = rag.process_document(
                        file_path,
                        metadata={
                            "filename": file_path.name,
                            "source": str(file_path)
                        }
                    )
                    processed_count += chunks
                    logger.info(f"Processed {chunks} chunks from {file_path}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    
        return processed_count
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process documents with RAG system")
    parser.add_argument("--rag_type", type=str, default="chroma", help="Type of RAG system to use")
    parser.add_argument("--delete", type=str, default="no", help="Whether to delete the collection before processing")
    parser.add_argument("--collection_name", type=str, default=None, help="Name of the collection")

    args = parser.parse_args()
    process_documents(rag_type=args.rag_type, delete=args.delete, collection_name=args.collection_name)