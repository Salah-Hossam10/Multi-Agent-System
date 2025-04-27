from langsmith import Client
from langchain.smith import RunEvalConfig
from rag_manager import RAGManager
import os
import time

# Set environment variables
# These environment variables enable access to the LangChain API and enable tracing for debugging
os.environ["LANGCHAIN_API_KEY"] = "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"  # Set the LangChain API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enable tracing for tracking API interactions

client = Client()  # Initialize the LangChain client to interact with the LangChain API

# Custom evaluators
def retrieval_accuracy(run, example):
    """
    Custom evaluator to check if the expected answer is found in any of the retrieved documents.
    """
    expected = example.outputs["expected"].lower()  # Convert the expected answer to lowercase
    score = sum(expected in doc["content"].lower() for doc in run.outputs["docs"])  # Count how many docs contain the expected answer
    return {"score": score, "passed": score > 0}  # Return the score and whether it passed (score > 0 means the answer was found)

def relevance_scoring(run, example):
    """
    Custom evaluator to check how many relevant keywords are found in the retrieved documents.
    """
    keywords = example.outputs.get("keywords", [])  # Get the keywords to look for from the example
    score = sum(kw in doc["content"].lower() for doc in run.outputs["docs"] for kw in keywords)  # Count how many keywords are in the documents
    return {"score": score, "max_score": len(keywords)}  # Return the total score and the maximum possible score

def run_rag_evaluation():
    """
    Function to evaluate the Retrieval-Augmented Generation (RAG) system using custom evaluators.
    """
    # Initialize the RAG system (RAGManager is assumed to manage RAG-related functionalities)
    rag = RAGManager()  # Create an instance of the RAG manager
    if not hasattr(rag, "query"):  # Ensure that the RAGManager has a 'query' method
        raise AttributeError("RAGManager must have a 'query' method.")  # Raise an error if 'query' is missing

    rag.load_vectorstore("faiss_index")  # Load the vector store (FAISS index) used for document retrieval in RAG

    # Create a unique dataset name based on the current timestamp
    timestamp = str(int(time.time()))[-6:]  # Get the last 6 digits of the current time as a unique identifier
    dataset_name = f"RAG-Medical-Eval-{timestamp}"  # Name the dataset with the timestamp for uniqueness

    # Create or fetch the dataset from LangChain
    try:
        dataset = client.create_dataset(dataset_name)  # Try to create a new dataset with the given name
    except Exception:
        # If dataset creation fails (e.g., the dataset already exists), try to fetch the existing dataset
        existing = list(client.list_datasets(dataset_name=dataset_name))
        dataset = existing[0] if existing else client.create_dataset(dataset_name)  # Use the first found dataset or create a new one

    # Define test cases with questions, expected answers, and relevant keywords
    test_cases = [
        {"question": "What are flu symptoms?", "expected": "fever", "keywords": ["fever", "cough", "throat"]},  # Flu symptoms query
        {"question": "How to treat flu?", "expected": "rest", "keywords": ["rest", "fluids"]}  # Flu treatment query
    ]

    # Add examples to the dataset
    for case in test_cases:
        client.create_example(
            inputs={"question": case["question"]},  # Use the question as input
            outputs={"expected": case["expected"], "keywords": case["keywords"]},  # Use expected output and keywords
            dataset_id=dataset.id  # Assign these examples to the dataset
        )

    # Define the RAG chain that will be used to retrieve documents
    def rag_chain(inputs):
        docs = rag.query(inputs["question"])  # Retrieve documents related to the question using the RAG system
        return {"docs": docs}  # Return the retrieved documents

    # Run the evaluation using the created dataset and the RAG chain
    client.run_on_dataset(
        dataset_name=dataset_name,  # Specify the dataset to evaluate
        llm_or_chain_factory=rag_chain,  # Specify the RAG chain function to generate answers
        evaluation=RunEvalConfig(
            custom_evaluators=[retrieval_accuracy, relevance_scoring],  # Specify custom evaluators for accuracy and relevance
            evaluation_key="docs"  # Use 'docs' as the key to evaluate the documents retrieved by RAG
        ),
        project_name=f"rag-eval-{timestamp}",  # Name the project with a unique identifier
        verbose=True  # Enable verbose output for debugging
    )

if __name__ == "__main__":
    run_rag_evaluation()  # Call the function to run the RAG evaluation when the script is executed directly
