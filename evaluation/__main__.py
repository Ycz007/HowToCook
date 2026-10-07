"""python -m evaluation --help"""
import argparse
from collections import Counter
from datetime import datetime
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import time

from .core import capture_answer, document_id, load_cases, retrieval_metrics

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def check_index(index_module, chunks, data_root):
    def signature(doc):
        return document_id(doc, data_root), doc.page_content
    store = index_module.vectorstore
    current = Counter(signature(doc) for doc in chunks)
    saved = Counter(signature(store.docstore.search(doc_id))
                    for doc_id in store.index_to_docstore_id.values())
    if saved != current or store.index.ntotal != len(chunks):
        raise ValueError("旧索引与当前分块不一致。请使用 --fresh-index 在内存中重建评估索引；原索引不会被覆盖。")


def prepare(args):
    from config import RAGConfig
    from rag_modules import DataPreparationModule, IndexConstructionModule, RetrievalOptimizationModule

    config = RAGConfig()
    if args.top_k:
        config.top_k = args.top_k
    data = DataPreparationModule(config.data_path)
    data.load_documents()
    chunks = data.chunk_documents()
    if not chunks:
        raise ValueError("未生成任何文档块")
    index = IndexConstructionModule(config.embedding_model, config.index_save_path)
    if args.fresh_index:
        index.build_vector_index(chunks)
    else:
        index.load_index()
        if index.vectorstore is None:
            raise ValueError("没有可加载的索引，请加 --fresh-index 在内存中构建。")
        check_index(index, chunks, config.data_path)
    retrieval = RetrievalOptimizationModule(index.vectorstore, chunks)
    return config, data, index, retrieval


def collect(args, output):
    from config import RAGConfig
    cases = load_cases(args.dataset, RAGConfig().data_path)
    if args.limit:
        cases = cases[:args.limit]
    config, data, index, retrieval = prepare(args)
    system = None
    if args.mode == "collect":
        from main import RecipeRAGSystem
        from rag_modules import GenerationIntegrationModule
        system = RecipeRAGSystem(config)
        system.data_module, system.index_module = data, index
        system.retrieval_module = retrieval
        system.generation_module = GenerationIntegrationModule(
            config.llm_model, config.temperature, config.max_tokens)
    versions = {}
    for name in ("langchain", "langchain-huggingface", "sentence-transformers", "ragas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(output / "run.json", {
        "mode": args.mode, "top_k": config.top_k, "dataset": str(args.dataset),
        "fresh_index": args.fresh_index, "model": config.embedding_model,
        "llm_model": config.llm_model, "versions": versions,
        "documents": len(data.documents), "chunks": len(data.chunks),
        "indexed_vectors": index.vectorstore.index.ntotal,
    })
    records = []
    # 每题立即落盘，后面的网络失败不会丢掉已经完成的结果。
    with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            started = time.perf_counter()
            row = dict(case, mode=args.mode, top_k=config.top_k)
            try:
                if system:
                    trace = capture_answer(system, case["user_input"])
                    chunks = trace.pop("chunks")
                    parents = trace.pop("parent_docs")
                    row.update(trace)
                    row["parent_doc_ids"] = [document_id(d, config.data_path) for d in parents]
                else:
                    # 独立检索基线：原始问题直接混合检索，无 LLM 路由/改写/生成。
                    chunks = retrieval.hybrid_search(case["user_input"], top_k=config.top_k)
                    row["rewritten_query"] = case["user_input"]
                row["retrieved_doc_ids"] = [document_id(d, config.data_path) for d in chunks]
                row["retrieved_chunks"] = [d.page_content for d in chunks]
                row.update(retrieval_metrics(case["reference_doc_ids"], row["retrieved_doc_ids"]))
                row["status"] = "ok"
            except Exception as exc:
                # 不将第三方异常正文写入报告，避免其中包含请求凭据。
                row.update(status="error", error_type=type(exc).__name__)
            row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            records.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{row['id']}: {row['status']} hit={row.get('hit_at_k')}")
    summary = {"total": len(records), "successful": sum(r["status"] == "ok" for r in records),
               "failed": sum(r["status"] == "error" for r in records), "top_k": config.top_k}
    for metric in ("hit_at_k", "recall_at_k", "mrr_at_k"):
        values = [r[metric] for r in records if r.get(metric) is not None]
        summary[metric] = sum(values) / len(values) if values else None
        summary[metric + "_samples"] = len(values)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed"] else 0


def score(args, output):
    # 对保存的同一批回答评分，不重新运行待测系统。
    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    usable, skipped = [], []
    for row in records:
        if (row.get("status") == "ok" and row.get("response")
                and row.get("retrieved_contexts") and row.get("reference")):
            usable.append(row)
        else:
            skipped.append(row.get("id", "unknown"))
    if not usable:
        raise ValueError("没有可评分记录，请先用 collect 模式采集回答和真实上下文。")
    if args.limit:
        usable = usable[:args.limit]
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextRecall, FactualCorrectness
    from ragas.run_config import RunConfig
    from langchain_openai import ChatOpenAI
    key = os.getenv("RAGAS_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("请设置 RAGAS_API_KEY 或 DEEPSEEK_API_KEY")
    base_url = os.getenv("RAGAS_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("RAGAS_MODEL", "deepseek-chat")
    judge = LangchainLLMWrapper(ChatOpenAI(
        model=model, api_key=key, base_url=base_url, temperature=0,
        timeout=120, max_retries=2))
    fields = ("user_input", "response", "retrieved_contexts", "reference")
    dataset = EvaluationDataset.from_list([{k: r[k] for k in fields} for r in usable])
    result = evaluate(dataset, metrics=[Faithfulness(), LLMContextRecall(), FactualCorrectness()],
                      llm=judge, run_config=RunConfig(max_workers=1, timeout=180),
                      raise_exceptions=False)
    frame = result.to_pandas()
    frame.insert(0, "id", [r["id"] for r in usable])
    frame.to_csv(output / "ragas_scores.csv", index=False, encoding="utf-8-sig")
    metrics = [c for c in frame.columns if c not in (*fields, "id")]
    summary = {"judge_model": model, "input_records": str(args.records),
               "scored": len(usable), "skipped_ids": skipped, "metrics": {}}
    for metric in metrics:
        valid = frame[metric].dropna()
        summary["metrics"][metric] = {
            "mean": float(valid.mean()) if len(valid) else None,
            "valid_samples": len(valid), "failed_samples": int(frame[metric].isna().sum()),
        }
    write_json(output / "ragas_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if any(v["failed_samples"] for v in summary["metrics"].values()) else 0


def main():
    parser = argparse.ArgumentParser(description="食谱 RAG 评估：检索基线、完整问答采集、Ragas 离线记录评分")
    parser.add_argument("mode", choices=("retrieval", "collect", "score"))
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/cases.jsonl")
    parser.add_argument("--records", type=Path, help="score 模式使用 collect 生成的 records.jsonl")
    parser.add_argument("--output", type=Path, help="新的报告目录（必须尚不存在）")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fresh-index", action="store_true", help="在内存中重建索引，不覆盖原索引")
    args = parser.parse_args()
    if any(v is not None and v <= 0 for v in (args.top_k, args.limit)):
        parser.error("--top-k 和 --limit 必须为正整数")
    if args.mode == "score" and not args.records:
        parser.error("score 需要 --records")
    # 用户传入的路径先按启动目录解析，项目数据路径再固定到项目根目录。
    args.dataset = args.dataset.resolve()
    if args.records:
        args.records = args.records.resolve()
    output = (args.output or ROOT / "evaluation/reports" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    os.chdir(ROOT)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    logging.basicConfig(level=logging.WARNING)
    output.mkdir(parents=True, exist_ok=False)
    try:
        status = score(args, output) if args.mode == "score" else collect(args, output)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"评估未完成: {exc}", file=sys.stderr)
        return 1
    print(f"报告目录: {output}")
    return status


if __name__ == "__main__":
    sys.exit(main())
