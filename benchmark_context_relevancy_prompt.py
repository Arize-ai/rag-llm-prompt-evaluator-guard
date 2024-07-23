"""Script to evaluate Context Relevancy Guard on "wiki_qa-train" benchmark dataset.
* https://huggingface.co/datasets/microsoft/wiki_qa

Model: gpt-4o-mini
Guard Results
              precision    recall  f1-score   support

       False       0.66      0.87      0.75       197
        True       0.89      0.70      0.79       303

    accuracy                           0.77       500
   macro avg       0.77      0.79      0.77       500
weighted avg       0.80      0.77      0.77       500

Latency
count    500.000000
mean       2.464671
std        1.350076
min        1.066755
25%        1.643355
50%        2.083322
75%        2.821537
max       17.161242
Name: guard_latency_gpt-4o-mini, dtype: float64
median latency
2.0833217084873468

Model: gpt-3.5-turbo
Guard Results
              precision    recall  f1-score   support

       False       0.40      1.00      0.58       197
        True       1.00      0.04      0.08       303

    accuracy                           0.42       500
   macro avg       0.70      0.52      0.33       500
weighted avg       0.77      0.42      0.28       500

Latency
count    500.000000
mean       1.425171
std        0.305953
min        0.957211
25%        1.228940
50%        1.365073
75%        1.552721
max        4.420569
Name: guard_latency_gpt-3.5-turbo, dtype: float64
median latency
1.3650730834924616
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
MODELS = ["gpt-4o-mini", "gpt-3.5-turbo"]
N_EVAL_SAMPLE_SIZE = 500
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
        print(classification_report(~test_dataset["relevant"], ~test_dataset[f"guard_passed_{model}"]))
        print("Latency")
        print(test_dataset[f"guard_latency_{model}"].describe())
        print("median latency")
        print(test_dataset[f"guard_latency_{model}"].median())
    
    if SAVE_RESULTS_PATH:
        test_dataset.to_csv(SAVE_RESULTS_PATH)
