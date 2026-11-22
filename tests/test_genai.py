from auditlens.config import resolve
from auditlens.genai.eval import evaluate, load_questions
from auditlens.genai.ingest import chunk_documents
from auditlens.genai.llm import ExtractiveLLM
from auditlens.genai.rag import NOT_FOUND, RAGPipeline


def _rag():
    from auditlens.config import get_config
    cfg = get_config()
    cfg["genai"]["embedder"], cfg["genai"]["llm_provider"] = "tfidf", "extractive"
    return RAGPipeline.build(cfg=cfg, persist=False, llm=ExtractiveLLM())


def test_chunking_keeps_sources():
    chunks = chunk_documents(resolve("docs/policies"))
    assert len(chunks) > 20
    assert all(c.source.endswith(".md") and c.source != "README.md" for c in chunks)


def test_ask_returns_citations():
    a = _rag().ask("How quickly must a first-line analyst triage an alert?")
    assert "2 business days" in a.answer
    assert a.citations and a.citations[0]["source"] == "transaction_monitoring_policy.md"


def test_out_of_scope_declines():
    a = _rag().ask("What is the Firm's policy on cryptocurrency custody for clients?")
    assert a.answer == NOT_FOUND


def test_eval_set_has_25_questions_and_passes_baseline():
    qs = load_questions(resolve("eval/rag_questions.jsonl"))
    assert len(qs) == 25
    _, summary = evaluate(_rag(), qs)
    assert summary["retrieval_hit_rate"] >= 0.9
    assert summary["accuracy"] >= 0.8
