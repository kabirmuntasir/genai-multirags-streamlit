# evaluate_rags.py
import time
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from rag.chroma_azure_rag import ChromaAzureRAG
from rag.azure_search_rag import AzureSearchRAG
from rag.light_rag import LightRAG
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_rag_system(rag_system, queries, true_answers):
    response_times = []
    predictions = []
    
    for query, true_answer in zip(queries, true_answers):
        start_time = time.time()
        response = rag_system.query(query)
        response_time = time.time() - start_time
        response_times.append(response_time)
        
        predicted_answer = response["answer"]
        predictions.append(predicted_answer)
    
    precision = precision_score(true_answers, predictions, average='weighted')
    recall = recall_score(true_answers, predictions, average='weighted')
    f1 = f1_score(true_answers, predictions, average='weighted')
    avg_response_time = sum(response_times) / len(response_times)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_response_time": avg_response_time
    }

def main():
    queries = ["What is the CRD number of FORD FINANCIAL?", "Tell me about FORD ASSET MANAGEMENT, LLC"]
    true_answers = ["173099", "173099"]
    
    rag_systems = {
        "Chroma Azure RAG": ChromaAzureRAG(
            persist_directory=os.getenv("PERSIST_DIRECTORY"),
            collection_name="chroma_docs"
        ),
        "Azure Search RAG": AzureSearchRAG(
            collection_name="docs-index"
        ),
        "Light RAG": LightRAG(
            persist_directory="./document_store/light_db",
            collection_name="light_docs"
        )
    }
    
    results = {}
    for name, rag_system in rag_systems.items():
        logger.info(f"Evaluating {name}")
        results[name] = evaluate_rag_system(rag_system, queries, true_answers)
    
    for name, metrics in results.items():
        logger.info(f"Results for {name}:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value}")

if __name__ == "__main__":
    main()