# Medical Conversation Processor (MCP) Evaluation

## Project Overview

This project evaluates the performance of a Medical Conversation Processor (MCP) using LangChain and FastAPI. The MCP is designed to process and respond to medical-related queries, including diagnostics and medication dosage. The evaluation includes custom metrics for medical accuracy and response relevance, with a focus on how well the model handles medical queries.

## Table of Contents

- [Installation](#installation)
- [Setup and Configuration](#setup-and-configuration)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Installation

### Prerequisites

To run this project, you will need:

- Python 3.8 or higher
- LangChain and FastAPI libraries

### Steps to Install

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mcp-evaluation.git
   cd mcp-evaluation


Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set environment variables for LangChain API key and tracing:

bash
Copy
Edit
export LANGCHAIN_API_KEY="your-langchain-api-key"
export LANGCHAIN_TRACING_V2="true"
Setup and Configuration
This project includes a set of custom evaluators designed to evaluate the model’s performance based on medical accuracy and relevance.

Custom Evaluators
Medical Accuracy: Checks if the expected medical term (e.g., diagnosis, dosage) appears in the agent’s response.

Response Relevance: Evaluates how many relevant medical keywords (e.g., "dose", "mg") are found in the response.

The evaluation is run on a dataset of test cases containing medical queries and expected answers.

Usage
Running the Evaluation
Ensure that the necessary environment variables are set:

LANGCHAIN_API_KEY

LANGCHAIN_TRACING_V2

Execute the evaluation script:

bash
Copy
Edit
python mcp_eval.py
This will:

Create or fetch a dataset from LangChain.

Add test cases to the dataset.

Run the evaluation using the mcp_eval function.

Display the evaluation results.

Evaluation
Evaluation Metrics
Medical Accuracy: Measures whether the expected answer is found in the agent’s response.

Response Relevance: Measures how many relevant medical keywords are found in the agent’s response.

For more detailed information about each metric, please refer to the custom evaluator functions:

medical_accuracy: Custom evaluator for checking if the expected output is present in the response.

response_relevance: Custom evaluator that checks how many relevant keywords appear in the response.

Example Test Cases
Query: "I have fever and headache"

Expected Output: "flu"

Query: "Can I take ibuprofen?"

Expected Output: "200-400mg"

Query: "What about paracetamol?"

Expected Output: "500mg"

Results
After running the evaluation, view the results at:

MCP Evaluation Results

All Test Results

License
This project is licensed under the MIT License - see the LICENSE file for details.

Made with ❤️ by Your Name

markdown
Copy
Edit

### Key Sections:

- **Project Overview**: Explains what the project is about.
- **Table of Contents**: Helps navigate the sections of the README.
- **Installation**: Step-by-step instructions for setting up the project.
- **Setup and Configuration**: Describes environment variables and custom evaluators.
- **Usage**: Instructions on how to run the evaluation script.
- **Evaluation**: Explains the evaluation metrics and test cases.
- **Results**: Provides links to view the results after running the evaluation.
- **License**: Specifies the project's license.

You can copy this template and modify it as needed for your repository.






You said:
please put it in one with good layout for github


ChatGPT said:
Here is your README.md file, formatted for GitHub with a clean and professional layout:

markdown
Copy
Edit
# Medical Conversation Processor (MCP) Evaluation

## Project Overview

This project evaluates the performance of a **Medical Conversation Processor (MCP)** using **LangChain** and **FastAPI**. The MCP is designed to process and respond to medical-related queries, including diagnostics and medication dosage. The evaluation includes custom metrics for medical accuracy and response relevance, with a focus on how well the model handles medical queries.


## Installation

### Prerequisites

To run this project, you will need:

- Python 3.8 or higher
- LangChain and FastAPI libraries

### Steps to Install

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mcp-evaluation.git
   cd mcp-evaluation
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set environment variables for LangChain API key and tracing:

bash
Copy
Edit
export LANGCHAIN_API_KEY="your-langchain-api-key"
export LANGCHAIN_TRACING_V2="true"
Setup and Configuration
This project includes a set of custom evaluators designed to evaluate the model’s performance based on medical accuracy and relevance.

Custom Evaluators
Medical Accuracy: Checks if the expected medical term (e.g., diagnosis, dosage) appears in the agent’s response.

Response Relevance: Evaluates how many relevant medical keywords (e.g., "dose", "mg") are found in the response.

The evaluation is run on a dataset of test cases containing medical queries and expected answers.

Usage
Running the Evaluation
Ensure that the necessary environment variables are set:

LANGCHAIN_API_KEY

LANGCHAIN_TRACING_V2

Execute the evaluation script:

bash
Copy
Edit
python mcp_eval.py
This will:

Create or fetch a dataset from LangChain.

Add test cases to the dataset.

Run the evaluation using the mcp_eval function.

Display the evaluation results.

Evaluation
Evaluation Metrics
Medical Accuracy: Measures whether the expected answer is found in the agent’s response.

Response Relevance: Measures how many relevant medical keywords are found in the agent’s response.

For more detailed information about each metric, please refer to the custom evaluator functions:

medical_accuracy: Custom evaluator for checking if the expected output is present in the response.

response_relevance: Custom evaluator that checks how many relevant keywords appear in the response.

Example Test Cases
Query: "I have fever and headache"
Expected Output: "flu"

Query: "Can I take ibuprofen?"
Expected Output: "200-400mg"

Query: "What about paracetamol?"
Expected Output: "500mg"

Results
After running the evaluation, view the results at:

MCP Evaluation Results

All Test Results
