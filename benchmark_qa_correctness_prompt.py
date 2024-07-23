"""Script to evaluate QA Correctness Guard on benchmark dataset.
The 2.0 version of the large-scale dataset Stanford Question Answering Dataset (SQuAD 2.0) allows
researchers to design AI models for reading comprehension tasks under challenging constraints.
https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15785042.pdf

Model: gpt-4o-mini

Guard Results
              precision    recall  f1-score   support

       False       0.99      0.96      0.98       148
        True       0.96      0.99      0.98       152

    accuracy                           0.98       300
   macro avg       0.98      0.98      0.98       300
weighted avg       0.98      0.98      0.98       300

Latency
count    300.000000
mean       2.157875
std        0.907331
min        0.985851
25%        1.537722
50%        1.855992
75%        2.492588
max        6.124077
Name: guard_latency_gpt-4o-mini, dtype: float64
median latency
1.8559920205152594

Model: gpt-3.5-turbo

Guard Results
              precision    recall  f1-score   support

       False       0.98      0.84      0.91       148
        True       0.87      0.98      0.92       152

    accuracy                           0.91       300
   macro avg       0.92      0.91      0.91       300
weighted avg       0.92      0.91      0.91       300

Latency
count    300.000000
mean       1.346867
std        0.467461
min        0.960516
25%        1.189675
50%        1.277035
75%        1.384363
max        6.908191
Name: guard_latency_gpt-3.5-turbo, dtype: float64
median latency
1.277035374485422
"""
import os
import time
from getpass import getpass
from typing import List, Tuple

import openai
import pandas as pd
from sklearn.metrics import classification_report

from guardrails import Guard
from main import QACorrectnessPrompt, LlmRagEvaluator
from phoenix.evals import download_benchmark_dataset
from sklearn.utils import shuffle


RANDOM_STATE = 119
MODELS = ["gpt-4o-mini", "gpt-3.5-turbo"]
N_EVAL_SAMPLE_SIZE = 300
SAVE_RESULTS_PATH = "qa_correctness_guard_results.csv"


def evaluate_guard_on_dataset(test_dataset: pd.DataFrame, guard: Guard, model: str) -> Tuple[List[float], List[bool]]:
    """Evaluate guard on benchmark dataset.

    :param test_dataset: Dataframe of test examples.
    :param guard: Guard we want to evaluate.

    :return: Tuple where the first lists contains latency, and the second list contains a boolean indicating whether the guard passed.
    """
    latency_measurements = []
    guard_passed = []
    for _, rag_example in test_dataset.iterrows():
        start_time = time.perf_counter()
        response = guard(
            llm_api=openai.chat.completions.create,
            prompt=rag_example["question"],
            model=model,
            max_tokens=1024,
            temperature=0.5,
            metadata={
                "user_message": rag_example["question"],
                "context": rag_example["context"],
                "llm_response": rag_example["sampled_answer"],
            }
        )
        latency_measurements.append(time.perf_counter() - start_time)
        guard_passed.append(response.validation_passed)
    return latency_measurements, guard_passed


if __name__ == "__main__":
    if not (openai_api_key := os.getenv("OPENAI_API_KEY")):
        openai_api_key = getpass("🔑 Enter your OpenAI API key: ")
    openai.api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Columns: Index(['id', 'title', 'context', 'question', 'answers', 'correct_answer', 'wrong_answer', 'sampled_answer', 'answer_true']
    test_dataset = df = download_benchmark_dataset(
        task="qa-classification",
        dataset_name="qa_generated_dataset")
    test_dataset = shuffle(test_dataset, random_state=RANDOM_STATE)
    test_dataset = test_dataset[:N_EVAL_SAMPLE_SIZE]
    
    for model in MODELS:
        guard = Guard.from_string(
            validators=[
                LlmRagEvaluator(
                    eval_llm_prompt_generator=QACorrectnessPrompt(prompt_name="qa_correctness_judge_llm"),
                    llm_evaluator_fail_response="incorrect",
                    llm_evaluator_pass_response="correct",
                    llm_callable=model,
                    on_fail="noop",
                    on="prompt")
            ],
        )
        
        latency_measurements, guard_passed = evaluate_guard_on_dataset(test_dataset=test_dataset, guard=guard, model=model)
        test_dataset[f"guard_passed_{model}"] = guard_passed
        test_dataset[f"guard_latency_{model}"] = latency_measurements
        
        print(f"\nModel: {model}")
        print("\nGuard Results")
        # Calculate precision, recall and f1-score for when the Guard fails (e.g. flags an incorrect answer)
        print(classification_report(~test_dataset["answer_true"], ~test_dataset[f"guard_passed_{model}"]))
        
        print("Latency")
        print(test_dataset[f"guard_latency_{model}"].describe())
        print("median latency")
        print(test_dataset[f"guard_latency_{model}"].median())

    if SAVE_RESULTS_PATH:
        test_dataset.to_csv(SAVE_RESULTS_PATH)
