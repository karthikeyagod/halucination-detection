from flask import Flask, render_template, request, jsonify
from datetime import datetime, timezone
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import re

from services.evidence import retrieve_evidence
from services.verification import (
    verify_claim,
    model_status,
    SUPPORTED,
    CONTRADICTED,
    INSUFFICIENT_EVIDENCE
)
from services.correction import build_corrected_paragraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

app = Flask(__name__)

# Maintain recent analysis history (last 20 analyses in memory)
RECENT_HISTORY = deque(maxlen=20)

ABBREVIATIONS = {
    'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr', 'vs', 'etc', 'e.g', 'i.e', 'u.s', 'sq', 'km',
    'approx', 'fig', 'al', 'inc', 'ltd', 'corp', 'dept', 'no', 'univ', 'est', 'st',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
}

def split_sentences(text: str) -> list:
    text = (text or "").strip()
    if not text:
        return []

    # Protect decimals like 3.14 or numbers like $10.50
    text = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', text)
    # Protect common abbreviations using case-insensitive escaped matching
    for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
        text = re.sub(rf'(?i)\b{re.escape(abbr)}\.', r'\g<0><DOT>', text)
        text = text.replace('.<DOT>', '<DOT>')
    # Protect numbered list prefixes at start of line OR directly following a sentence boundary
    text = re.sub(r'(?m)^(\d+)\.\s+', r'\1<NUMDOT> ', text)
    text = re.sub(r'(?<=[.!?])\s+(\d{1,3})\.\s+', r' \1<NUMDOT> ', text)

    # Split on sentence boundaries: newlines or [.!?] followed by whitespace
    raw_splits = re.split(r'(?:[\r\n]+|(?<=[.!?])\s+)', text)
    sentences = []
    for s in raw_splits:
        s = s.replace('<DECIMAL>', '.').replace('<DOT>', '.').replace('<NUMDOT>', '.')
        s = re.sub(r'^\d+[\.\)]\s*', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        if s:
            sentences.append(s)
    return sentences

def _process_single_sentence(idx: int, sentence: str) -> dict:
    try:
        evidence_pack = retrieve_evidence(sentence)
        verification = verify_claim(sentence, evidence_pack)

        verdict = verification.get("verdict", INSUFFICIENT_EVIDENCE)
        confidence = verification.get("confidence", 0.0)
        confidence_type = verification.get("confidence_type", "estimated")
        explanation = verification.get("explanation", "")
        evidence_items = verification.get("evidence", [])
        correction = verification.get("correction")
        source_title = verification.get("source_title", "")
        source_url = verification.get("source_url", "")
        passage = verification.get("passage", "")

        return {
            "id": idx + 1,
            "sentence": sentence,
            "claim": sentence,
            "verdict": verdict,
            "status": verdict,
            "confidence": confidence,
            "confidence_type": confidence_type,
            "explanation": explanation,
            "evidence": evidence_items,
            "evidence_text": passage,
            "source": source_title,
            "source_title": source_title,
            "source_url": source_url,
            "passage": passage,
            "correction": correction,
            "correction_validated": bool(correction)
        }
    except Exception as exc:
        logger.exception("Analysis failed for sentence: %s. Error: %s", sentence, exc)
        return {
            "id": idx + 1,
            "sentence": sentence,
            "claim": sentence,
            "verdict": INSUFFICIENT_EVIDENCE,
            "status": INSUFFICIENT_EVIDENCE,
            "confidence": 0.0,
            "confidence_type": "estimated",
            "explanation": "Verification failed safely; no reliable conclusion was produced.",
            "evidence": [],
            "evidence_text": "",
            "source": "",
            "source_title": "",
            "source_url": "",
            "passage": "",
            "correction": None,
            "correction_validated": False
        }

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "service": "LLM Hallucination Detection and Correction Framework",
        "verification_model": model_status(),
        "time": datetime.now(timezone.utc).isoformat()
    })

@app.post("/api/verify")
def verify():
    """Verify a single proposition directly and return standardized Section 8 response schema."""
    data = request.get_json(silent=True) or request.form or {}
    claim = (data.get("claim") or data.get("text") or "").strip()
    if not claim:
        return jsonify({"error": "Please provide a 'claim' to verify."}), 400

    evidence_pack = retrieve_evidence(claim)
    result = verify_claim(claim, evidence_pack)

    response = {
        "claim": result["claim"],
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "confidence_type": result.get("confidence_type", "estimated"),
        "explanation": result["explanation"],
        "evidence": result["evidence"],
        "correction": result["correction"]
    }
    return jsonify(response)

@app.post("/api/analyze")
def analyze():
    data = request.get_json(silent=True) or request.form or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please paste text to analyze."}), 400
    if len(text) > 10000:
        return jsonify({"error": "Please keep input under 10,000 characters to prevent excessive latency."}), 413

    sentences = split_sentences(text)
    if not sentences:
        return jsonify({"error": "No valid sentences found in input."}), 400
    if len(sentences) > 30:
        return jsonify({"error": "Maximum 30 sentences per analysis allowed."}), 400

    # Process sentences concurrently using a thread pool for parallel retrieval & inference
    results = [None] * len(sentences)
    max_workers = min(6, len(sentences))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_process_single_sentence, i, sentence): i
            for i, sentence in enumerate(sentences)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("Unexpected worker error on sentence %d: %s", idx, exc)
                results[idx] = {
                    "id": idx + 1,
                    "sentence": sentences[idx],
                    "claim": sentences[idx],
                    "verdict": INSUFFICIENT_EVIDENCE,
                    "status": INSUFFICIENT_EVIDENCE,
                    "confidence": 0.0,
                    "explanation": "Worker execution error.",
                    "evidence": [],
                    "evidence_text": "",
                    "source": "",
                    "source_title": "",
                    "source_url": "",
                    "passage": "",
                    "correction": None,
                    "correction_validated": False
                }

    counts = {
        SUPPORTED: sum(1 for x in results if x and x.get("verdict") == SUPPORTED),
        CONTRADICTED: sum(1 for x in results if x and x.get("verdict") == CONTRADICTED),
        INSUFFICIENT_EVIDENCE: sum(1 for x in results if x and x.get("verdict") == INSUFFICIENT_EVIDENCE),
        # Legacy UI aliases
        "Supported": sum(1 for x in results if x and x.get("verdict") == SUPPORTED),
        "Contradicted": sum(1 for x in results if x and x.get("verdict") == CONTRADICTED),
        "Unverified": sum(1 for x in results if x and x.get("verdict") == INSUFFICIENT_EVIDENCE)
    }
    corrected_paragraph = build_corrected_paragraph(results)

    response_payload = {
        "results": results,
        "analysis": results,
        "counts": counts,
        "summary": counts,
        "corrected_paragraph": corrected_paragraph,
        "total_sentences": len(results),
        "supported_count": counts[SUPPORTED],
        "contradicted_count": counts[CONTRADICTED],
        "insufficient_evidence_count": counts[INSUFFICIENT_EVIDENCE],
        "unverified_count": counts[INSUFFICIENT_EVIDENCE]
    }

    # Store in recent history
    preview = text[:80] + ("…" if len(text) > 80 else "")
    RECENT_HISTORY.appendleft({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "preview": preview,
        "total_sentences": len(results),
        "counts": counts,
        "corrected_paragraph": corrected_paragraph
    })

    return jsonify(response_payload)

@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or data.get("claim") or "").strip()
    if not question:
        return jsonify({"reply": "Ask a factual question or statement to verify."})

    evidence = retrieve_evidence(question)
    verif = verify_claim(question, evidence)
    verdict = verif.get("verdict", INSUFFICIENT_EVIDENCE)
    conf = verif.get("confidence", 0.0)
    src = verif.get("source_title") or evidence.get("source") or "Reference source"
    expl = verif.get("explanation") or ""
    corr = verif.get("correction")

    reply = f"Claim verdict: {verdict} ({expl})"

    return jsonify({
        "claim": question,
        "verdict": verdict,
        "status": verdict,
        "confidence": conf,
        "confidence_type": verif.get("confidence_type", "estimated"),
        "explanation": expl,
        "evidence": verif.get("evidence", []),
        "source": src,
        "source_title": src,
        "source_url": verif.get("source_url") or evidence.get("source_url", ""),
        "correction": corr,
        "reply": reply
    })

@app.get("/api/history")
def history():
    return jsonify({"history": list(RECENT_HISTORY)})

@app.post("/api/history/clear")
@app.delete("/api/history")
def clear_history():
    RECENT_HISTORY.clear()
    return jsonify({"ok": True, "message": "History cleared successfully."})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
