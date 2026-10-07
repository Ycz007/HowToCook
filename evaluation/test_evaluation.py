"""不调用网络、不加载模型的回归测试。"""
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation.core import capture_answer, load_cases, retrieval_metrics
from evaluation.__main__ import check_index


class EvaluationTests(unittest.TestCase):
    def test_deduplication_and_recall(self):
        result = retrieval_metrics(["a", "b"], ["x", "x", "a"])
        self.assertEqual(result, {"hit_at_k": 1.0, "recall_at_k": 0.5, "mrr_at_k": 0.5})
        self.assertEqual(retrieval_metrics(["a"], ["x"])["hit_at_k"], 0)
        self.assertIsNone(retrieval_metrics([], ["x"])["hit_at_k"])

    def test_sample_sources_exist(self):
        root = Path(__file__).resolve().parents[1]
        cases = load_cases(root / "evaluation/cases.jsonl", root / "data/cook")
        self.assertGreaterEqual(len(cases), 50)
        self.assertEqual(len({c["user_input"] for c in cases}), len(cases))

    def test_same_count_stale_index_rejected(self):
        directory = Path(__file__).resolve().parent
        source = str(directory / "a.md")
        old = SimpleNamespace(metadata={"source": source}, page_content="old")
        new = SimpleNamespace(metadata={"source": source}, page_content="new")
        store = SimpleNamespace(index=SimpleNamespace(ntotal=1),
                                index_to_docstore_id={0: "a"},
                                docstore=SimpleNamespace(search=lambda _: old))
        module = SimpleNamespace(vectorstore=store)
        check_index(module, [old], directory)
        with self.assertRaisesRegex(ValueError, "fresh-index"):
            check_index(module, [new], directory)

    def test_capture_uses_actual_context_and_restores_methods(self):
        calls = []
        generation = SimpleNamespace(query_router=lambda q: "detail", query_rewrite=lambda q: "rewritten",
                                     _build_context=lambda docs: "truncated prompt")
        retrieval = SimpleNamespace(hybrid_search=lambda q: ["candidate", "excluded"])
        def filtered(q):
            return retrieval.hybrid_search(q)[:1]
        retrieval.metadata_filtered_search = filtered
        data = SimpleNamespace(get_parent_documents=lambda chunks: ["full parent"])
        original = generation._build_context
        def ask(q, stream=False):
            calls.append(q)
            generation.query_router(q)
            rewritten = generation.query_rewrite(q)
            chunks = retrieval.metadata_filtered_search(rewritten)
            parents = data.get_parent_documents(chunks)
            generation._build_context(parents)
            return "answer"
        system = SimpleNamespace(generation_module=generation, retrieval_module=retrieval,
                                 data_module=data, ask_question=ask)
        trace = capture_answer(system, "question")
        self.assertEqual(calls, ["question"])
        self.assertEqual(trace["chunks"], ["candidate"])
        self.assertEqual(trace["retrieved_contexts"], ["truncated prompt"])
        self.assertEqual(trace["rewritten_query"], "rewritten")
        self.assertIs(generation._build_context, original)
        system.ask_question = lambda *a, **kw: (_ for _ in ()).throw(ValueError("failed"))
        with self.assertRaises(ValueError):
            capture_answer(system, "question")
        self.assertIs(generation._build_context, original)


if __name__ == "__main__":
    unittest.main()
