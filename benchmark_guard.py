"""Script to evaluate Guard on benchmark dataset. Currently supported datasets include "halueval_qa_data" from the HaluEval benchmark:
* https://arxiv.org/abs/2305.11747
* https://github.com/RUCAIBox/HaluEval
"""
import os
import time
from getpass import getpass
from typing import List, Tuple
import logging

import openai
import pandas as pd
from sklearn.metrics import classification_report

from guardrails import Guard
from main import HallucinationPrompt, LlmRagEvaluator
from phoenix.evals import download_benchmark_dataset

logger = logging.getLogger(__name__)


MODEL = "gpt-4-turbo"
N_EVAL_SAMPLE_SIZE = 100


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
        logging.info(response)
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
    logging.info(classification_report(test_dataset["is_hallucination"], ~test_dataset["guard_passed"]))
    
    logging.info("Latency")
    logging.info(test_dataset["guard_latency"].describe())
