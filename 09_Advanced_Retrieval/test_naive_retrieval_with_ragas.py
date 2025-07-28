import os
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from ragas import generate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    answer_correctness,
    answer_similarity
)
from datasets import Dataset
import numpy as np

# Set up API keys (you'll need to set these)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Load the complaints data
loader = CSVLoader(
    file_path="./data/complaints.csv",
    metadata_columns=[
        "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
        "Consumer complaint narrative", "Company public response", "Company",
        "State", "ZIP code", "Tags", "Consumer consent provided?",
        "Submitted via", "Date sent to company", "Company response to consumer",
        "Timely response?", "Consumer disputed?", "Complaint ID"
    ]
)

loan_complaint_data = loader.load()

for doc in loan_complaint_data:
    doc.page_content = doc.metadata["Consumer complaint narrative"]

# Create the naive retrieval chain
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Qdrant.from_documents(
    loan_complaint_data,
    embeddings,
    location=":memory:",
    collection_name="LoanComplaints"
)

naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
chat_model = ChatOpenAI(model="gpt-4.1-nano")

naive_retrieval_chain = (
    {"context": itemgetter("question") | naive_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

# Create synthetic dataset using Ragas
def create_synthetic_dataset():
    """Create a synthetic dataset using Ragas for evaluation"""
    
    # Convert documents to text for Ragas
    documents = [doc.page_content for doc in loan_complaint_data[:100]]  # Use first 100 for efficiency
    
    # Generate synthetic questions and answers
    synthetic_data = generate(
        documents,
        n_questions=20,  # Generate 20 questions
        llm=chat_model,
        embeddings=embeddings
    )
    
    return synthetic_data

# Create the synthetic dataset
print("Creating synthetic dataset with Ragas...")
synthetic_dataset = create_synthetic_dataset()

# Convert to the format expected by the retrieval chain
def prepare_evaluation_data(dataset):
    """Prepare the dataset for evaluation"""
    questions = []
    ground_truth_answers = []
    
    for item in dataset:
        questions.append(item['question'])
        ground_truth_answers.append(item['answer'])
    
    return questions, ground_truth_answers

# Prepare evaluation data
questions, ground_truth_answers = prepare_evaluation_data(synthetic_dataset)

# Test the naive retrieval chain
def test_naive_retrieval_chain(questions, ground_truth_answers):
    """Test the naive retrieval chain and collect results"""
    results = []
    
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truth_answers)):
        print(f"Testing question {i+1}/{len(questions)}: {question[:100]}...")
        
        try:
            # Get response from the chain
            response = naive_retrieval_chain.invoke({"question": question})
            
            # Get retrieved context
            retrieved_context = response["context"]
            generated_answer = response["response"].content
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "context": retrieved_context
            })
            
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": "Error occurred",
                "context": []
            })
    
    return results

# Run the evaluation
print("Testing naive retrieval chain...")
evaluation_results = test_naive_retrieval_chain(questions, ground_truth_answers)

# Create dataset for Ragas evaluation
def create_ragas_dataset(evaluation_results):
    """Create a dataset in the format expected by Ragas"""
    ragas_data = []
    
    for result in evaluation_results:
        # Convert context list to string
        context_str = "\n".join([doc.page_content for doc in result["context"]])
        
        ragas_data.append({
            "question": result["question"],
            "answer": result["generated_answer"],
            "contexts": [context_str],
            "ground_truth": result["ground_truth"]
        })
    
    return Dataset.from_list(ragas_data)

# Create Ragas dataset
ragas_evaluation_dataset = create_ragas_dataset(evaluation_results)

# Evaluate using Ragas metrics
def evaluate_with_ragas(dataset):
    """Evaluate the retrieval chain using Ragas metrics"""
    print("Evaluating with Ragas metrics...")
    
    # Define metrics to evaluate
    metrics = [
        faithfulness,
        answer_relevancy,
        context_relevancy,
        context_recall,
        answer_correctness,
        answer_similarity
    ]
    
    # Run evaluation
    results = {}
    for metric in metrics:
        try:
            score = metric.score(dataset)
            results[metric.name] = score
            print(f"{metric.name}: {score:.4f}")
        except Exception as e:
            print(f"Error evaluating {metric.name}: {e}")
            results[metric.name] = None
    
    return results

# Run evaluation
evaluation_scores = evaluate_with_ragas(ragas_evaluation_dataset)

# Print summary
print("\n" + "="*50)
print("EVALUATION SUMMARY")
print("="*50)
for metric, score in evaluation_scores.items():
    if score is not None:
        print(f"{metric}: {score:.4f}")
    else:
        print(f"{metric}: Error occurred")

# Save results
def save_results(evaluation_results, evaluation_scores):
    """Save evaluation results to files"""
    
    # Save detailed results
    detailed_results = []
    for result in evaluation_results:
        detailed_results.append({
            "question": result["question"],
            "ground_truth": result["ground_truth"],
            "generated_answer": result["generated_answer"],
            "context_length": len(result["context"])
        })
    
    df_detailed = pd.DataFrame(detailed_results)
    df_detailed.to_csv("naive_retrieval_detailed_results.csv", index=False)
    
    # Save metrics
    df_metrics = pd.DataFrame([evaluation_scores])
    df_metrics.to_csv("naive_retrieval_metrics.csv", index=False)
    
    print("\nResults saved to:")
    print("- naive_retrieval_detailed_results.csv")
    print("- naive_retrieval_metrics.csv")

save_results(evaluation_results, evaluation_scores)

# Additional analysis
def analyze_performance(evaluation_results):
    """Analyze the performance in detail"""
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Calculate average context length
    context_lengths = [len(result["context"]) for result in evaluation_results]
    avg_context_length = np.mean(context_lengths)
    print(f"Average context length: {avg_context_length:.2f} documents")
    
    # Check for errors
    errors = [r for r in evaluation_results if r["generated_answer"] == "Error occurred"]
    print(f"Number of errors: {len(errors)}")
    
    # Sample some results
    print("\nSample Results:")
    for i, result in enumerate(evaluation_results[:3]):
        print(f"\nQuestion {i+1}: {result['question'][:100]}...")
        print(f"Ground Truth: {result['ground_truth'][:100]}...")
        print(f"Generated Answer: {result['generated_answer'][:100]}...")

analyze_performance(evaluation_results) 