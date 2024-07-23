"""Script to evaluate Hallucination Guard on benchmark dataset.
Currently supported datasets include "halueval_qa_data" from the HaluEval benchmark:
* https://arxiv.org/abs/2305.11747
* https://github.com/RUCAIBox/HaluEval

MODEL = "gpt-4o-mini"
INFO:root:Guard Results
INFO:root:              precision    recall  f1-score   support

       False       0.78      0.98      0.86       265
        True       0.96      0.68      0.80       235

    accuracy                           0.84       500
   macro avg       0.87      0.83      0.83       500
weighted avg       0.86      0.84      0.83       500

INFO:root:Latency
INFO:root:count    500.000000
mean       1.843508
std        1.634371
min        1.056757
25%        1.377324
50%        1.547401
75%        1.862995
max       26.142654
Name: guard_latency, dtype: float64
INFO:root:median latency
INFO:root:1.5474009165191092

MODEL = "gpt-3.5-turbo"
INFO:root:Guard Results
INFO:root:              precision    recall  f1-score   support

       False       0.84      0.80      0.82       275
        True       0.77      0.82      0.79       225

    accuracy                           0.81       500
   macro avg       0.80      0.81      0.81       500
weighted avg       0.81      0.81      0.81       500

INFO:root:Latency
INFO:root:count    500.000000
mean       1.337587
std        0.343144
min        0.950819
25%        1.176967
50%        1.282560
75%        1.439100
max        7.220026
Name: guard_latency, dtype: float64
INFO:root:median latency
INFO:root:1.2825598954805173
"""
import os
import time
from getpass import getpass
from typing import List, Tuple

import openai
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from guardrails import Guard
from main import HallucinationPrompt, LlmRagEvaluator
from phoenix.evals import download_benchmark_dataset


RANDOM_STATE = 119
MODELS = ["gpt-4o-mini", "gpt-3.5-turbo"]
N_EVAL_SAMPLE_SIZE = 300
SAVE_RESULTS_PATH = "hallucination_guard_results.csv"


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
            prompt=rag_example["query"],
            model=model,
            max_tokens=1024,
            temperature=0.5,
            metadata={
                "user_message": rag_example["query"],
                "context": rag_example["reference"],
                "llm_response": rag_example["response"],
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
    
    # Columns: ['reference', 'query', 'response', 'is_hallucination']
    test_dataset = download_benchmark_dataset(
        task="binary-hallucination-classification",
        dataset_name="halueval_qa_data")
    test_dataset = shuffle(test_dataset, random_state=119)
    test_dataset = test_dataset[:N_EVAL_SAMPLE_SIZE]
    
    for model in MODELS:
        guard = Guard.from_string(
            validators=[
                LlmRagEvaluator(
                    eval_llm_prompt_generator=HallucinationPrompt(prompt_name="hallucination_judge_llm"),
                    llm_evaluator_fail_response="hallucinated",
                    llm_evaluator_pass_response="factual",
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
        # Calculate precision, recall and f1-score for when the Guard fails (e.g. flags a hallucination)
        print(classification_report(test_dataset["is_hallucination"], ~test_dataset[f"guard_passed_{model}"]))
        
        print("Latency")
        print(test_dataset[f"guard_latency_{model}"].describe())
        print("median latency")
        print(test_dataset[f"guard_latency_{model}"].median())

    if SAVE_RESULTS_PATH:
        test_dataset.to_csv(SAVE_RESULTS_PATH)
