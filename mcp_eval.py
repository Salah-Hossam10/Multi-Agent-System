from langsmith import Client
from langchain.smith import RunEvalConfig
from mcp_server import app
from fastapi.testclient import TestClient
import os
import time

# Set environment variables
# These environment variables are required to enable tracing and API calls to LangChain's API
os.environ["LANGCHAIN_API_KEY"] = "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"  # Setting the API key for LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enabling tracing to collect detailed information about API calls

# Initialize clients
client = Client()  # Initializing LangChain's client to interact with LangChain API
test_client = TestClient(app)  # Initializing FastAPI's TestClient to simulate HTTP requests to the FastAPI app

# Custom evaluators
def medical_accuracy(run, example):
    """
    Custom evaluator for medical accuracy. Checks if the expected output is found within the agent's response.
    """
    return {"score": int(example.outputs["expected"].lower() in run.outputs["response"].lower())}

def response_relevance(run, example):
    """
    Custom evaluator for response relevance. Checks how many relevant keywords are in the response.
    """
    response = run.outputs["response"].lower()  # Convert the agent's response to lowercase
    keywords = ["dose", "mg", "hours"]  # Defining relevant medical keywords
    score = sum(word in response for word in keywords)  # Counting how many keywords are found in the response
    return {"score": min(score, 3) / 3}  # Normalizing the score, ensuring it’s between 0 and 1

def run_mcp_evaluation():
    """
    Function that sets up and runs the evaluation of the MCP (Medical Conversation Processor).
    It creates the dataset, adds examples, and evaluates the model using custom evaluators.
    """
    timestamp = str(int(time.time()))[-6:]  # Generating a unique timestamp (last 6 digits of current time)
    dataset_name = f"MCP-Medical-Eval-{timestamp}"  # Generating a unique dataset name with the timestamp
    project_name = f"mcp-eval-{timestamp}"  # Generating a unique project name

    # Create or fetch dataset
    try:
        dataset = client.create_dataset(dataset_name)  # Try creating a new dataset
    except Exception:
        # If dataset creation fails (because it might already exist), fetch the existing dataset
        dataset = list(client.list_datasets(dataset_name=dataset_name))[0]

    # Test cases
    test_cases = [
        {"message": "I have fever and headache", "expected": "flu"},  # Simulating a user asking about flu symptoms
        {"message": "Can I take ibuprofen?", "expected": "200-400mg"},  # User asking for the dosage of ibuprofen
        {"message": "What about paracetamol?", "expected": "500mg"}  # User asking for the dosage of paracetamol
    ]

    # Add examples
    # Loop through each test case, creating examples in the dataset for evaluation
    for case in test_cases:
        client.create_example(
            inputs={"message": case["message"]},  # Using the message as input
            outputs={"expected": case["expected"]},  # Using the expected output (e.g., diagnosis or dosage) as expected output
            dataset_id=dataset.id  # Assigning the example to the dataset
        )

    # Chain runner
    def run_chain(inputs):
        """
        This function mimics the agent’s response by sending a POST request to the `/process` endpoint of the FastAPI app.
        """
        response = test_client.post("/process", json=inputs)  # Sending a POST request with input data to the FastAPI app
        return {"response": response.json()["response"]}  # Extracting and returning the agent's response from the JSON

    # Evaluation config
    eval_config = RunEvalConfig(
        custom_evaluators=[medical_accuracy, response_relevance],  # Adding the custom evaluators for accuracy and relevance
        evaluation_key="response"  # The key used to evaluate the response (using 'response' key in the output)
    )

    # Run evaluation with retries
    for attempt in range(3):  # Try to run the evaluation up to 3 times in case of failure
        try:
            # Run the evaluation using the dataset, evaluation config, and the chain runner
            client.run_on_dataset(
                dataset_name=dataset_name,  # The dataset to evaluate
                llm_or_chain_factory=run_chain,  # The function that handles the agent's response generation
                evaluation=eval_config,  # The evaluation configuration
                project_name=project_name,  # The project name
                verbose=True  # Enable verbose output for debugging purposes
            )
            break  # Exit the loop if the evaluation runs successfully
        except Exception:
            if attempt == 2:  # If it's the third attempt and it fails, raise an error
                raise
            time.sleep(2)  # Wait for 2 seconds before retrying
            print(f"Retry {attempt + 1}/3...")  # Print a message indicating which attempt is being made

if __name__ == "__main__":
    run_mcp_evaluation()  # Start the MCP evaluation when the script is executed directly
