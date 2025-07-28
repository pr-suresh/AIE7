# How to Test naive_retrieval_chain with Ragas in Your Notebook

## Step 1: Add a new cell to install Ragas
```python
!pip install ragas datasets
```

## Step 2: Add imports cell
```python
import os
import pandas as pd
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
```

## Step 3: Create synthetic dataset
```python
def create_synthetic_dataset():
    """Create a synthetic dataset using Ragas for evaluation"""
    
    # Convert documents to text for Ragas
    documents = [doc.page_content for doc in loan_complaint_data[:50]]  # Use first 50 for efficiency
    
    # Generate synthetic questions and answers
    synthetic_data = generate(
        documents,
        n_questions=10,  # Generate 10 questions for testing
        llm=chat_model,
        embeddings=embeddings
    )
    
    return synthetic_data

# Create the synthetic dataset
print("Creating synthetic dataset with Ragas...")
synthetic_dataset = create_synthetic_dataset()
print(f"Generated {len(synthetic_dataset)} questions")
```

## Step 4: Prepare evaluation data
```python
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
```

## Step 5: Test the naive retrieval chain
```python
def test_naive_retrieval_chain(questions, ground_truth_answers):
    """Test the naive retrieval chain and collect results"""
    results = []
    
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truth_answers)):
        print(f"Testing question {i+1}/{len(questions)}: {question[:80]}...")
        
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
```

## Step 6: Create dataset for Ragas evaluation
```python
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
```

## Step 7: Evaluate using Ragas metrics
```python
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
```

## Step 8: Display results and analysis
```python
print("\n" + "="*50)
print("EVALUATION SUMMARY")
print("="*50)
for metric, score in evaluation_scores.items():
    if score is not None:
        print(f"{metric}: {score:.4f}")
    else:
        print(f"{metric}: Error occurred")

# Additional analysis
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
    print(f"\nQuestion {i+1}: {result['question'][:80]}...")
    print(f"Ground Truth: {result['ground_truth'][:80]}...")
    print(f"Generated Answer: {result['generated_answer'][:80]}...")
```

## Step 9: Optional - Save results
```python
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

# Uncomment to save results
# save_results(evaluation_results, evaluation_scores)
```

## Step 10: Optional - Compare with other retrievers
```python
def compare_retrievers():
    """Compare different retrievers using the same synthetic dataset"""
    retrievers = {
        "naive": naive_retriever,
        "bm25": bm25_retriever,
        # Add other retrievers here
    }
    
    comparison_results = {}
    
    for name, retriever in retrievers.items():
        print(f"\nTesting {name} retriever...")
        
        # Create chain with this retriever
        retrieval_chain = (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
        )
        
        # Test the chain
        results = test_naive_retrieval_chain(questions, ground_truth_answers)
        
        # Evaluate with Ragas
        ragas_dataset = create_ragas_dataset(results)
        scores = evaluate_with_ragas(ragas_dataset)
        
        comparison_results[name] = scores
    
    return comparison_results

# Uncomment to run comparison
# comparison_results = compare_retrievers()
```

## What This Will Do:

1. **Create Synthetic Dataset**: Ragas will generate realistic questions and answers based on your loan complaints data
2. **Test Retrieval Chain**: Your naive_retrieval_chain will be tested against these synthetic questions
3. **Evaluate Performance**: Ragas will calculate multiple metrics:
   - **Faithfulness**: How well the answer follows the retrieved context
   - **Answer Relevancy**: How relevant the answer is to the question
   - **Context Relevancy**: How relevant the retrieved context is to the question
   - **Context Recall**: How much of the relevant information was retrieved
   - **Answer Correctness**: How accurate the answer is compared to ground truth
   - **Answer Similarity**: How similar the generated answer is to the ground truth

4. **Provide Analysis**: You'll get detailed performance metrics and sample results

## Notes:
- Make sure you have your API keys set up
- Start with a small number of questions (10) for testing
- You can increase the number of questions and documents for more comprehensive evaluation
- The comparison function allows you to test multiple retrievers against the same dataset 