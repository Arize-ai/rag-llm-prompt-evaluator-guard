"""Script to evaluate Hallucination Guard on benchmark dataset.
Currently supported datasets include "halueval_qa_data" from the HaluEval benchmark:
* https://arxiv.org/abs/2305.11747
* https://github.com/RUCAIBox/HaluEval

INFO:root:Guard Results
INFO:root:              precision    recall  f1-score   support

       False       0.83      0.93      0.88        54
        True       0.90      0.78      0.84        46

    accuracy                           0.86       100
   macro avg       0.87      0.85      0.86       100
weighted avg       0.86      0.86      0.86       100

INFO:root:Latency
INFO:root:count    100.000000
mean       1.533940
std        0.552186
min        1.069116
25%        1.256626
50%        1.393182
75%        1.617315
max        4.579247
Name: guard_latency, dtype: float64
"""
import os
import time
from getpass import getpass
from typing import List, Tuple
import logging
import random

import openai
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from guardrails import Guard
from main import HallucinationPrompt, LlmRagEvaluator
from phoenix.evals import download_benchmark_dataset

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

random.seed(119)


MODEL = "gpt-4o-mini"
N_EVAL_SAMPLE_SIZE = 500


def evaluate_guard_on_dataset(test_dataset: pd.DataFrame, guard: Guard) -> Tuple[List[float], List[bool]]:
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
            model=MODEL,
            max_tokens=1024,
            temperature=0.5,
            metadata={
                "user_message": rag_example["query"],
                "context": rag_example["reference"],
                "llm_response": rag_example["response"],
            }
        )
        latency_measurements.append(time.perf_counter() - start_time)
        logging.info(response)
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
    test_dataset = shuffle(test_dataset)
    test_dataset = test_dataset[:N_EVAL_SAMPLE_SIZE]
    
    guard = Guard.from_string(
        validators=[
            LlmRagEvaluator(
                eval_llm_prompt_generator=HallucinationPrompt(prompt_name="hallucination_judge_llm"),
                llm_evaluator_fail_response="hallucinated",
                llm_evaluator_pass_response="factual",
                llm_callable=MODEL,
                on_fail="noop",
                on="prompt")
        ],
    )
    
    latency_measurements, guard_passed = evaluate_guard_on_dataset(test_dataset=test_dataset, guard=guard)
    test_dataset["guard_passed"] = guard_passed
    test_dataset["guard_latency"] = latency_measurements
    
    logging.info("Guard Results")
    # Calculate precision, recall and f1-score for when the Guard fails (e.g. flags a hallucination)
    logging.info(classification_report(test_dataset["is_hallucination"], ~test_dataset["guard_passed"]))
    
    logging.info("Latency")
    logging.info(test_dataset["guard_latency"].describe())
    logging.info("median latency")
    logging.info(test_dataset["guard_latency"].median())
