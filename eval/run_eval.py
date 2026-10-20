"""Runner delle prove real-LLM K5.2.

Uso: ``python eval/run_eval.py --validate-only`` oppure, con GROQ_API_KEY,
``python eval/run_eval.py``. Il runner non modifica il DB applicativo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from backend.agents.groq_config import GROQ_MODEL, safe_groq_error

load_dotenv()

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
RESULTS = ROOT / "results"
REQUIRED = {"id", "category", "reference", "input", "expected", "obtained", "outcome", "comment"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases() -> list[dict]:
    cases = []
    for path in sorted(CASES.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = REQUIRED - set(data)
        if missing:
            raise ValueError(f"{path.name}: campi mancanti: {sorted(missing)}")
        ref = data["reference"]
        material = ROOT / ref["file"]
        if not material.exists():
            raise ValueError(f"{path.name}: materiale non trovato: {ref['file']}")
        actual_hash = _sha256(material)
        if str(ref.get("sha256", "")).lower() != actual_hash.lower():
            raise ValueError(f"{path.name}: hash materiale non aggiornato")
        if data["outcome"] not in {"ok", "parziale", "ko"}:
            raise ValueError(f"{path.name}: esito non valido")
        cases.append(data)
    if len(cases) < 24:
        raise ValueError(f"Servono almeno 24 casi, trovati {len(cases)}")
    return cases


def _metadata() -> dict:
    dependencies = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in distributions() if dist.metadata.get("Name")
    )
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "model": GROQ_MODEL,
        "prompt_versions": ["eval_runner_v2"],
        "method": "Agno/Groq diretto con testo di riferimento completo nel prompt; non usa il retrieval dell'app",
        "temperature": os.getenv("LLM_TEMPERATURE", "default client"),
        "seed": os.getenv("LLM_SEED", "non impostato"),
        "python": platform.python_version(),
        "dependencies": dependencies,
    }


def _execute_real_case(case: dict) -> dict:
    """Esegue un caso in isolamento con l'agente reale e conserva il grezzo."""
    from agno.agent import Agent
    from agno.models.groq import Groq
    material = (ROOT / case["reference"]["file"]).read_text(encoding="utf-8")
    scored = "score" in case["expected"]
    instruction = (
        "Valuta la risposta dello studente alla domanda. Assegna un voto intero da 0 a 10 "
        "alla risposta fornita, senza sostituirla con quella corretta. "
        "Una risposta vuota, fuori tema o non testuale vale 0. "
        "Usa score per il voto."
        if scored else
        "Rispondi alla richiesta usando solo il materiale. Se il materiale non supporta la "
        "richiesta, imposta supported=false e answer=null. Usa score=null. "
        "Per un quiz usa anche quiz come lista di domande; per appunti inserisci il testo in answer."
    )
    prompt = (
        "Sei un valutatore per una prova di validazione. Rispondi con un solo oggetto JSON valido, "
        "senza markdown, con le chiavi obbligatorie answer, score, cited_pages, supported, motivation. "
        "cited_pages deve contenere numeri di pagina interi. Non inventare citazioni.\n"
        f"{instruction}\nMateriale di riferimento:\n{material}\n\nCaso:\n"
        f"{json.dumps(case['input'], ensure_ascii=False)}"
    )
    agent = Agent(model=Groq(id=GROQ_MODEL, timeout=30), markdown=False)
    output = agent.run(prompt, stream=False)
    raw = output.get_content_as_string() if hasattr(output, "get_content_as_string") else str(output)
    try:
        parsed = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except (json.JSONDecodeError, TypeError):
        parsed = None
    return {"raw": raw, "parsed": parsed, "model": GROQ_MODEL}


def run(validate_only: bool = False, case_ids: set[str] | None = None) -> Path:
    cases = load_cases()
    if validate_only:
        return CASES
    if case_ids is not None:
        cases = [case for case in cases if case["id"] in case_ids]
        missing = case_ids - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"Casi non trovati: {sorted(missing)}")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY è richiesto per le prove real-LLM")
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for case in cases:
        try:
            case["obtained"] = _execute_real_case(case)
        except Exception as exc:
            case["obtained"] = {"error": safe_groq_error(exc), "model": GROQ_MODEL}
    summary = {
        "total": len(cases),
        "real_responses": sum(bool(c["obtained"] and c["obtained"].get("raw")) for c in cases),
        "errors": sum(bool(c["obtained"] and c["obtained"].get("error")) for c in cases),
        "not_executed": sum(c["obtained"] is None for c in cases),
    }
    scored_cases = [c for c in cases if "score" in c["expected"]]
    measured_scores = [c for c in scored_cases if isinstance((c["obtained"] or {}).get("parsed"), dict)
                       and type(c["obtained"]["parsed"].get("score")) is int]
    summary["scores_measured"] = len(measured_scores)
    summary["scores_exact"] = sum(c["obtained"]["parsed"]["score"] == c["expected"]["score"]
                                  for c in measured_scores)
    summary["score_mae"] = (round(sum(abs(c["obtained"]["parsed"]["score"] - c["expected"]["score"])
                                  for c in measured_scores) / len(measured_scores), 2)
                            if measured_scores else None)
    summary["scores_within_2"] = sum(abs(c["obtained"]["parsed"]["score"] - c["expected"]["score"]) <= 2
                                   for c in measured_scores)
    cited = [c for c in cases if isinstance((c["obtained"] or {}).get("parsed"), dict)
             and isinstance(c["obtained"]["parsed"].get("cited_pages"), list)]
    summary["citations_measured"] = len(cited)
    summary["citation_exact_match"] = sum(set(c["obtained"]["parsed"]["cited_pages"]) == set(c["expected"].get("cited_pages", []))
                                          for c in cited)
    result = {"metadata": _metadata(), "cases": cases, "summary": summary}
    output = RESULTS / f"eval_{stamp}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = RESULTS / f"eval_{stamp}.md"
    summary.write_text("| Misura | Casi |\n|---|---:|\n" + "\n".join(
        f"| {key} | {value} |" for key, value in result["summary"].items()
    ) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--case-id", action="append", dest="case_ids",
                        help="Esegue solo il caso indicato; ripetibile.")
    args = parser.parse_args()
    print(run(args.validate_only, set(args.case_ids) if args.case_ids else None))
