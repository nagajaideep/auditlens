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


def test_gemini_provider_selected_and_called(monkeypatch):
    """GeminiLLM is chosen when a key is set and sends system prompt + question (SDK mocked)."""
    import sys
    import types as pytypes

    from auditlens.genai import llm as llm_mod

    calls = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.update(model=model, contents=contents, config=config)
            return pytypes.SimpleNamespace(text="Alerts are triaged within 2 business days [1].", candidates=[])

    class FakeClient:
        def __init__(self, api_key=None):
            calls["api_key"] = api_key
            self.models = FakeModels()

    class FakeConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    google_mod = pytypes.ModuleType("google")
    genai_mod = pytypes.ModuleType("google.genai")
    genai_mod.Client = FakeClient
    genai_mod.types = pytypes.SimpleNamespace(GenerateContentConfig=FakeConfig)
    google_mod.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_mod.types)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    llm = llm_mod.get_llm("auto")
    assert isinstance(llm, llm_mod.GeminiLLM)
    out = llm.complete("SYS", "QUESTION: x")
    assert "2 business days" in out
    assert calls["api_key"] == "test-key" and calls["config"].system_instruction == "SYS"


def test_no_key_falls_back_to_extractive(monkeypatch):
    from auditlens.genai import llm as llm_mod
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert isinstance(llm_mod.get_llm("auto"), llm_mod.ExtractiveLLM)
