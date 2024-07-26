"""Script to evaluate Context Relevancy Guard on "wiki_qa-train" benchmark dataset.
* https://huggingface.co/datasets/microsoft/wiki_qa

Model: gpt-4o-mini
Guard Results
              precision    recall  f1-score   support

    relevant       0.70      0.86      0.77        93
   unrelated       0.85      0.68      0.76       107

    accuracy                           0.77       200
   macro avg       0.78      0.77      0.76       200
weighted avg       0.78      0.77      0.76       200

Latency
count    200.000000
mean       2.812122
std        1.753805
min        1.067620
25%        1.708051
50%        2.248962
75%        3.321251
max       14.102804
Name: guard_latency_gpt-4o-mini, dtype: float64
median latency
2.2489616039965767

Model: gpt-4-turbo
Guard Results
              precision    recall  f1-score   support

    relevant       0.64      0.90      0.75        93
   unrelated       0.87      0.56      0.68       107

    accuracy                           0.72       200
   macro avg       0.76      0.73      0.72       200
weighted avg       0.76      0.72      0.71       200

Latency
count    200.000000
mean       8.561413
std        6.425799
min        1.624563
25%        3.957226
50%        5.979291
75%       11.579224
max       34.342637
Name: guard_latency_gpt-4-turbo, dtype: float64
median latency
5.979290812509134
"""
import os
import time
from getpass import getpass
from typing import List, Tuple

import openai
import pandas as pd
from sklearn.metrics import classification_report

from guardrails import Guard
from main import ContextRelevancyPrompt, LlmRagEvaluator
from phoenix.evals import download_benchmark_dataset
from sklearn.utils import shuffle


RANDOM_STATE = 119
MODELS = ["gpt-4o-mini", "gpt-4-turbo"]
N_EVAL_SAMPLE_SIZE = 200
SAVE_RESULTS_PATH = "context_relevancy_guard_results.csv"


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
            prompt=rag_example["query_text"],
            model=model,
            max_tokens=1024,
            temperature=0.5,
            metadata={
                "user_message": rag_example["query_text"],
                "context": rag_example["document_text"],
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
    
    # Columns: Index(['query_id', 'query_text', 'document_title', 'document_text', 'document_text_with_emphasis', 'relevant']
    test_dataset = download_benchmark_dataset(
        task="binary-relevance-classification",
        dataset_name="wiki_qa-train")
    test_dataset = shuffle(test_dataset, random_state=RANDOM_STATE)
    test_dataset = test_dataset[:N_EVAL_SAMPLE_SIZE]
    
    for model in MODELS:
        guard = Guard.from_string(
            validators=[
                LlmRagEvaluator(
                    eval_llm_prompt_generator=ContextRelevancyPrompt(prompt_name="context_relevancy_judge_llm"),
                    llm_evaluator_fail_response="unrelated",
                    llm_evaluator_pass_response="relevant",
                    llm_callable=model,
                    on_fail="noop",
                    on="prompt")
            ],
        )
        
        latency_measurements, guard_passed = evaluate_guard_on_dataset(test_dataset=test_dataset, guard=guard, model=model)
        test_dataset[f"guard_passed_{model}"] = guard_passed
        test_dataset[f"guard_latency_{model}"] = latency_measurements
        
        print(f"\nModel: {model}")
        print("Guard Results")
        # Calculate precision, recall and f1-score for when the Guard fails (e.g. flags an irrelevant answer)
        print(classification_report(
            test_dataset["relevant"].replace(True, "relevant").replace(False, "unrelated"),
            test_dataset[f"guard_passed_{model}"].replace(True, "relevant").replace(False, "unrelated")))
        print("Latency")
        print(test_dataset[f"guard_latency_{model}"].describe())
        print("median latency")
        print(test_dataset[f"guard_latency_{model}"].median())
    
    if SAVE_RESULTS_PATH:
        test_dataset.to_csv(SAVE_RESULTS_PATH)
