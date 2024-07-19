"""Script to evaluate Context Relevancy Guard on "wiki_qa-train" benchmark dataset.
* https://huggingface.co/datasets/microsoft/wiki_qa


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
from main import ContextRelevancyPrompt, LlmRagEvaluator
from phoenix.evals import download_benchmark_dataset
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


MODEL = "gpt-4o-mini"
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
            prompt=rag_example["query_text"],
            model=MODEL,
            max_tokens=1024,
            temperature=0.5,
            metadata={
                "user_message": rag_example["query_text"],
                "context": rag_example["document_text"],
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
    
    # Columns: Index(['query_id', 'query_text', 'document_title', 'document_text', 'document_text_with_emphasis', 'relevant']
    test_dataset = download_benchmark_dataset(
        task="binary-relevance-classification",
        dataset_name="wiki_qa-train")
    test_dataset = shuffle(test_dataset)
    test_dataset = test_dataset[:N_EVAL_SAMPLE_SIZE]
    
    guard = Guard.from_string(
        validators=[
            LlmRagEvaluator(
                eval_llm_prompt_generator=ContextRelevancyPrompt(prompt_name="context_relevancy_judge_llm"),
                llm_evaluator_fail_response="unrelated",
                llm_evaluator_pass_response="relevant",
                llm_callable=MODEL,
                on_fail="noop",
                on="prompt")
        ],
    )
    
    latency_measurements, guard_passed = evaluate_guard_on_dataset(test_dataset=test_dataset, guard=guard)
    test_dataset["guard_passed"] = guard_passed
    test_dataset["guard_latency"] = latency_measurements
    
    logging.info("Guard Results")
    logging.info(classification_report(test_dataset["relevant"], ~test_dataset["guard_passed"]))
    
    logging.info("Latency")
    logging.info(test_dataset["guard_latency"].describe())
