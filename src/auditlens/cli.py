"""`auditlens` command-line entry point."""
from __future__ import annotations

import argparse
import json


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="auditlens")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate synthetic PaySim-schema data")
    g.add_argument("--rows", type=int, default=200_000); g.add_argument("--out", default="data/raw/paysim.csv")
    g.add_argument("--steps", type=int, default=744); g.add_argument("--seed", type=int, default=42)
    sub.add_parser("load-db", help="load the raw CSV into PostgreSQL").add_argument("--csv", default=None)
    t = sub.add_parser("train", help="run the full training pipeline")
    t.add_argument("--trials", type=int, default=None)
    b = sub.add_parser("batch", help="nightly batch scoring")
    b.add_argument("--input", default=None); b.add_argument("--to-postgres", action="store_true")
    sub.add_parser("index", help="build the RAG vector index")
    a = sub.add_parser("ask", help="ask the policy assistant"); a.add_argument("question")
    e = sub.add_parser("eval-rag", help="run the 25-question RAG evaluation")
    e.add_argument("--questions", default="eval/rag_questions.jsonl")
    s = sub.add_parser("serve", help="start the API"); s.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    if args.cmd == "generate":
        from pathlib import Path

        from auditlens.data.generate import generate
        df = generate(args.rows, args.steps, seed=args.seed)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote {len(df):,} rows ({df.isFraud.mean():.3%} fraud) -> {args.out}")
    elif args.cmd == "load-db":
        from auditlens.config import get_config, resolve
        from auditlens.data.load import csv_to_postgres
        print(csv_to_postgres(args.csv or resolve(get_config()["data"]["raw_csv"])), "rows loaded")
    elif args.cmd == "train":
        from auditlens.pipelines.train import run
        s = run(n_trials=args.trials)
        print(json.dumps({k: s[k] for k in ["best_model", "model_version", "test", "forecast_weighted_mape"]},
                         indent=2, default=str))
    elif args.cmd == "batch":
        from auditlens.pipelines.batch_score import run
        print(run(args.input, args.to_postgres))
    elif args.cmd == "index":
        from auditlens.genai.rag import RAGPipeline
        r = RAGPipeline.build(persist=True)
        print(f"indexed with {r.embedder.name}; llm={r.llm.name}")
    elif args.cmd == "ask":
        from auditlens.genai.rag import RAGPipeline
        ans = RAGPipeline.build(persist=False).ask(args.question)
        print(ans.answer)
        for c in ans.citations:
            print(f"  [{c['n']}] {c['citation']}")
    elif args.cmd == "eval-rag":
        from auditlens.config import resolve
        from auditlens.genai.eval import evaluate, load_questions
        from auditlens.genai.rag import RAGPipeline
        df, summary = evaluate(RAGPipeline.build(persist=False), load_questions(resolve(args.questions)))
        out = resolve("reports"); out.mkdir(exist_ok=True)
        df.to_csv(out / "rag_eval.csv", index=False)
        (out / "rag_eval_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("auditlens.api.main:app", host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
