"""Script to evaluate Guard on benchmark dataset. Currently supported datasets include "halueval_qa_data" from the HaluEval benchmark:
* https://arxiv.org/abs/2305.11747
* https://github.com/RUCAIBox/HaluEval
"""
import os
import time
from getpass import getpass
from typing import List, Tuple, Type
import logging

import openai
import pandas as pd
from sklearn.metrics import classification_report

from guardrails import Guard
from main import HallucinationPrompt, QACorrectnessPrompt, ContextRelevancyPrompt, LlmRagEvaluator
from phoenix.evals import download_benchmark_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MODEL = "gpt-4-turbo"
N_EVAL_SAMPLE_SIZE = 100
# Choose one of HallucinationPrompt, QACorrectnessPrompt, ContextRelevancyPrompt
PROMPT_TEMPLATE = HallucinationPrompt(prompt_name="hallucination_judge_llm")
# hallucinated, incorrect, unrelated
FAIL_RESPONSE = "hallucinated"
# factual, correct, relevant
PASS_RESPONSE = "factual"
# is_hallucinated, correct_answer, relevant
GT_COLUMN_NAME = "is_hallucinated"


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


def get_benchmark_dataset(prompt_template: LlmRagEvaluator) -> pd.DataFrame:
    if isinstance(prompt_template, HallucinationPrompt):
        test_dataset = download_benchmark_dataset(task="binary-hallucination-classification", dataset_name="halueval_qa_data")
    if isinstance(prompt_template, QACorrectnessPrompt):
        test_dataset = download_benchmark_dataset(task="qa-classification", dataset_name="qa_generated_dataset")
    if isinstance(prompt_template, ContextRelevancyPrompt):
        test_dataset = download_benchmark_dataset(task="binary-relevance-classification", dataset_name="wiki_qa-train")
    return test_dataset[:N_EVAL_SAMPLE_SIZE]


if __name__ == "__main__":
    if not (openai_api_key := os.getenv("OPENAI_API_KEY")):
        openai_api_key = getpass("🔑 Enter your OpenAI API key: ")
    openai.api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Columns: ['reference', 'query', 'response', 'is_hallucination']
    test_dataset = get_benchmark_dataset(prompt_template=PROMPT_TEMPLATE)
    
    guard = Guard.from_string(
        validators=[
            LlmRagEvaluator(
                eval_llm_prompt_generator=PROMPT_TEMPLATE,
                llm_evaluator_fail_response=FAIL_RESPONSE,
                llm_evaluator_pass_response=PASS_RESPONSE,
                llm_callable=MODEL,
                on_fail="noop",
                on="prompt")
        ],
    )
    
    latency_measurements, guard_passed = evaluate_guard_on_dataset(test_dataset=test_dataset, guard=guard)
    test_dataset["guard_passed"] = guard_passed
    test_dataset["guard_latency"] = latency_measurements
    
    logging.info("Guard Results")
    logging.info(classification_report(test_dataset[GT_COLUMN_NAME], ~test_dataset["guard_passed"]))
    
    logging.info("Latency")
    logging.info(test_dataset["guard_latency"].describe())
