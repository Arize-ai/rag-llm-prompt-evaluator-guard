"""Script to evaluate Hallucination Guard on benchmark dataset.
Currently supported datasets include "halueval_qa_data" from the HaluEval benchmark:
* https://arxiv.org/abs/2305.11747
* https://github.com/RUCAIBox/HaluEval

Model: gpt-4o-mini
Guard Results
              precision    recall  f1-score   support

     factual       0.79      0.97      0.87       129
hallucinated       0.96      0.73      0.83       121

    accuracy                           0.85       250
   macro avg       0.87      0.85      0.85       250
weighted avg       0.87      0.85      0.85       250

Latency
count    250.000000
mean       1.865513
std        0.603700
min        1.139974
25%        1.531160
50%        1.758210
75%        2.026153
max        6.403010
Name: guard_latency_gpt-4o-mini, dtype: float64
median latency
1.7582097915001214

Model: gpt-4-turbo
Guard Results
              precision    recall  f1-score   support

     factual       0.83      0.88      0.85       129
hallucinated       0.87      0.80      0.83       121

    accuracy                           0.84       250
   macro avg       0.85      0.84      0.84       250
weighted avg       0.85      0.84      0.84       250

Latency
count    250.000000
mean       4.295613
std        2.393394
min        1.460899
25%        2.868255
50%        3.724649
75%        4.939440
max       23.465773
Name: guard_latency_gpt-4-turbo, dtype: float64
median latency
3.724648874514969
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
MODELS = ["gpt-4o-mini", "gpt-4-turbo"]
N_EVAL_SAMPLE_SIZE = 250
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
        print(classification_report(
            test_dataset["is_hallucination"].replace(True, "hallucinated").replace(False, "factual"),
            test_dataset[f"guard_passed_{model}"].replace(True, "factual").replace(False, "hallucinated")))
        
        print("Latency")
        print(test_dataset[f"guard_latency_{model}"].describe())
        print("median latency")
        print(test_dataset[f"guard_latency_{model}"].median())

    if SAVE_RESULTS_PATH:
        test_dataset.to_csv(SAVE_RESULTS_PATH)
