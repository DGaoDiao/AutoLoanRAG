"""Evaluate the automotive-loan RAG pipeline with local RAGAS judges.

The script runs the real BGE-M3 -> Milvus hybrid search -> CrossEncoder
reranking pipeline, generates an answer with the configured Ollama model, and
then evaluates the resulting answer/context pairs with RAGAS.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.config import Config
from rag_qa.core.prompts import RAGPrompts
from rag_qa.core.vector_store import VectorStore


ASSESSMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = ASSESSMENT_DIR / "rag_evaluate_data.json"
DEFAULT_RUN_DATA = ASSESSMENT_DIR / "rag_evaluate_run.json"
DEFAULT_RESULT_JSON = ASSESSMENT_DIR / "ragas_result.json"
DEFAULT_RESULT_CSV = ASSESSMENT_DIR / "ragas_result.csv"


def load_cases(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """执行 load_cases 函数。
        
        params:
            path: 参数说明。
            limit: 参数说明。
        
        return:
            函数返回值。"""
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    required = {"id", "question", "reference", "source_filter"}
    for index, case in enumerate(cases):
        missing = required - case.keys()
        if missing:
            raise ValueError(f"Case {index} is missing fields: {sorted(missing)}")
    return cases[:limit] if limit else cases


def generate_answer(client: OpenAI, prompt: str, model: str) -> str:
    """执行 generate_answer 函数。
        
        params:
            client: 参数说明。
            prompt: 参数说明。
            model: 参数说明。
        
        return:
            函数返回值。"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是审慎的汽车贷款知识库问答助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        timeout=90,
    )
    return response.choices[0].message.content.strip()


def run_rag(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """执行 run_rag 函数。
        
        params:
            cases: 参数说明。
        
        return:
            函数返回值。"""
    config = Config()
    client = OpenAI(
        api_key=config.LLM_DASHSCOPE_API_KEY,
        base_url=config.LLM_DASHSCOPE_BASE_URL,
    )
    prompt_template = RAGPrompts.rag_prompt()
    records: list[dict[str, Any]] = []

    with VectorStore() as vector_store:
        for index, case in enumerate(cases, start=1):
            started_at = time.perf_counter()
            documents = vector_store.hybrid_search(
                case["question"], source_filter=case["source_filter"]
            )
            contexts = [document.page_content for document in documents]
            prompt = prompt_template.format(
                history="",
                context="\n\n".join(contexts),
                question=case["question"],
                phone=config.APP_CUSTOMER_SERVICE_PHONE,
            )
            answer = generate_answer(client, prompt, config.LLM_MODEL)
            elapsed = time.perf_counter() - started_at
            record = {
                **case,
                "response": answer,
                "retrieved_contexts": contexts,
                "latency_seconds": round(elapsed, 3),
            }
            records.append(record)
            print(
                f"[{index}/{len(cases)}] {case['id']}: "
                f"contexts={len(contexts)}, latency={elapsed:.2f}s"
            )
    return records


def build_ragas_dataset(records: list[dict[str, Any]]) -> Dataset:
    """执行 build_ragas_dataset 函数。
        
        params:
            records: 参数说明。
        
        return:
            函数返回值。"""
    return Dataset.from_dict(
        {
            "user_input": [record["question"] for record in records],
            "response": [record["response"] for record in records],
            "retrieved_contexts": [record["retrieved_contexts"] for record in records],
            "reference": [record["reference"] for record in records],
        }
    )


def save_json(path: Path, value: Any) -> None:
    """执行 save_json 函数。
        
        params:
            path: 参数说明。
            value: 参数说明。
        
        return:
            函数返回值。"""
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def main() -> None:
    """执行 main 函数。
        
        params:
            无。
        
        return:
            函数返回值。"""
    parser = argparse.ArgumentParser(description="Run local RAGAS evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--reuse-run",
        action="store_true",
        help="Reuse rag_evaluate_run.json instead of executing RAG again",
    )
    parser.add_argument("--judge-model", default="qwen2.5:3b")
    parser.add_argument("--embedding-model", default="mxbai-embed-large")
    args = parser.parse_args()

    if args.reuse_run:
        records = load_cases(DEFAULT_RUN_DATA, limit=args.limit)
    else:
        records = run_rag(load_cases(args.dataset, limit=args.limit))
        save_json(DEFAULT_RUN_DATA, records)

    dataset = build_ragas_dataset(records)
    judge_llm = ChatOpenAI(
        model=args.judge_model,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
        max_tokens=1536,
        timeout=300,
        extra_body={"think": False},
    )
    judge_embeddings = OpenAIEmbeddings(
        model=args.embedding_model,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        check_embedding_ctx_length=False,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(timeout=180, max_retries=1, max_wait=15, max_workers=1),
        raise_exceptions=False,
    )
    frame = result.to_pandas()
    frame.insert(0, "id", [record["id"] for record in records])
    frame.insert(1, "category", [record.get("category", "") for record in records])
    frame.insert(2, "latency_seconds", [record["latency_seconds"] for record in records])
    frame.to_csv(DEFAULT_RESULT_CSV, index=False, encoding="utf-8-sig")

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    summary: dict[str, Any] = {
        metric: round(float(frame[metric].dropna().mean()), 4)
        for metric in metric_names
        if metric in frame
    }
    summary["valid_samples_by_metric"] = {
        metric: int(frame[metric].notna().sum())
        for metric in metric_names
        if metric in frame
    }
    summary["average_latency_seconds"] = round(
        float(pd.Series([record["latency_seconds"] for record in records]).mean()), 3
    )
    summary["evaluated_samples"] = len(records)
    json_frame = frame.astype(object).where(pd.notna(frame), None)
    save_json(DEFAULT_RESULT_JSON, {"summary": summary, "rows": json_frame.to_dict("records")})

    print("\nRAGAS summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Detailed CSV: {DEFAULT_RESULT_CSV}")
    print(f"Detailed JSON: {DEFAULT_RESULT_JSON}")


if __name__ == "__main__":
    main()
