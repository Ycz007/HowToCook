"""数据校验、文档级指标及同一次问答的数据采集。"""
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def load_cases(path, data_root):
    cases = []
    seen = set()
    root = Path(data_root).resolve()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "user_input", "reference"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"第 {number} 行缺少非空字符串 {key}")
        if row["id"] in seen:
            raise ValueError(f"重复题目 ID: {row['id']}")
        seen.add(row["id"])
        refs = row.get("reference_doc_ids")
        if not isinstance(refs, list) or not all(isinstance(x, str) and x for x in refs):
            raise ValueError(f"第 {number} 行 reference_doc_ids 必须是字符串列表")
        normalized = []
        for ref in refs:
            target = (root / ref.replace("\\", "/")).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                raise ValueError(f"第 {number} 行参考文档不存在或超出数据目录: {ref}")
            normalized.append(target.relative_to(root).as_posix())
        row["reference_doc_ids"] = list(dict.fromkeys(normalized))
        cases.append(row)
    if not cases:
        raise ValueError("测试集为空")
    return cases


def document_id(doc, data_root):
    return Path(doc.metadata["source"]).resolve().relative_to(Path(data_root).resolve()).as_posix()


def retrieval_metrics(reference_ids, retrieved_ids):
    """在实际返回的前 k 个子块中，按文档去重后计算指标。"""
    expected = set(reference_ids)
    ranked = list(dict.fromkeys(retrieved_ids))
    if not expected:
        # 无答案题不能把未标注相关文档解释成检索满分。
        return {"hit_at_k": None, "recall_at_k": None, "mrr_at_k": None}
    matched = expected.intersection(ranked)
    first = next((i for i, item in enumerate(ranked, 1) if item in expected), None)
    return {
        "hit_at_k": float(bool(matched)),
        "recall_at_k": len(matched) / len(expected),
        "mrr_at_k": 1 / first if first else 0.0,
    }


def capture_answer(system, question):
    """旁路记录真实 ask_question 调用；不二次检索或二次生成。仅供串行评估。"""
    trace = {"route": None, "rewritten_query": question, "chunks": [],
             "parent_docs": [], "retrieved_contexts": []}

    def recorder(original, key):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            trace[key] = result
            return result
        return call

    with ExitStack() as stack:
        for obj, method, key in (
            (system.generation_module, "query_router", "route"),
            (system.generation_module, "query_rewrite", "rewritten_query"),
            (system.retrieval_module, "hybrid_search", "chunks"),
            (system.retrieval_module, "metadata_filtered_search", "chunks"),
            (system.data_module, "get_parent_documents", "parent_docs"),
            (system.generation_module, "_build_context", "prompt_context"),
        ):
            stack.enter_context(patch.object(obj, method, recorder(getattr(obj, method), key)))
        trace["response"] = system.ask_question(question, stream=False)
    if "prompt_context" in trace:
        # 保存截断、添加元数据后的真实文本，而非完整父文档的假定内容。
        trace["retrieved_contexts"] = [trace.pop("prompt_context")]
    return trace
