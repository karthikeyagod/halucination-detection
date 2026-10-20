"""Local Ollama correction generation and NLI-grounded validation."""
import os
import re
import logging
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))

def _ollama(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 128}
            },
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        return (response.json().get("response") or "").strip()
    except requests.exceptions.ConnectionError:
        logger.debug("Ollama is not running or unreachable at %s", OLLAMA_URL)
        return ""
    except Exception as exc:
        logger.warning("Ollama generation failed: %s", exc)
        return ""

def _clean_correction_output(raw: str) -> str:
    """Extract the single corrected sentence from potential LLM conversational preamble."""
    if not raw:
        return ""
    text = raw.strip().strip('"\'')

    # Remove conversational prefixes
    patterns = [
        r"^(here\s+(is|are)\s+(the\s+)?(corrected\s+sentence|correction|factual\s+statement):?\s*)",
        r"^(corrected\s+sentence:?\s*)",
        r"^(correction:?\s*)",
        r"^(sure,?\s*(here\s+is\s+the\s+correction:?)?\s*)"
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip().strip('"\'')

    # Extract the first sentence or line
    lines = [line.strip().strip('"\'') for line in text.split("\n") if line.strip()]
    if not lines:
        return ""

    candidate = lines[0]
    # If the candidate says NONE or indicates unable to correct
    if candidate.upper() in {"NONE", "NONE.", "N/A", "UNKNOWN"}:
        return ""

    if len(candidate) > 400:
        return ""

    return candidate

def suggest_correction(claim: str, evidence_pack: Optional[Dict[str, Any]] = None) -> str:
    evidence_pack = evidence_pack or {}
    evidence = (evidence_pack.get("evidence") or "").strip()
    if not evidence:
        return ""

    prompt = f"""You are a precise factual editor. Using ONLY the facts in the evidence below, correct the contradicted claim.
Return ONE concise corrected sentence. Do NOT add commentary, explanations, or facts not mentioned in the evidence.
If the evidence does not clearly provide facts to correct the claim, reply with exactly NONE.

<claim>
{claim}
</claim>

<evidence>
{evidence[:3000]}
</evidence>

Corrected sentence:"""

    raw_result = _ollama(prompt)
    correction = _clean_correction_output(raw_result)

    if not correction:
        return ""

    # Don't return if it just repeats the original claim
    if correction.rstrip(".!?").strip().lower() == claim.rstrip(".!?").strip().lower():
        return ""

    return correction

def validate_correction(original: str, correction: str, evidence_pack: Optional[Dict[str, Any]] = None) -> bool:
    """Validate the proposed correction using the NLI model (grounded) with Ollama fallback."""
    evidence_pack = evidence_pack or {}
    evidence = (evidence_pack.get("evidence") or "").strip()
    if not correction or not evidence:
        return False

    # 1. Primary validation: Use the DeBERTa NLI model to verify if evidence supports the correction
    try:
        from services.verification import verify_claim
        nli_result = verify_claim(correction, evidence_pack)
        if nli_result.get("status") == "Supported":
            logger.info("Correction verified by NLI model: '%s'", correction)
            return True
    except Exception as exc:
        logger.debug("NLI validation check unavailable: %s", exc)

    # 2. Fallback validation: Query Ollama
    prompt = f"""Decide whether the following statement is directly supported by the provided evidence.
Answer strictly YES or NO.

<statement>
{correction}
</statement>

<evidence>
{evidence[:3000]}
</evidence>

Is the statement supported by the evidence? Answer YES or NO:"""

    answer = _ollama(prompt).strip().upper()
    return answer.startswith("YES")

def build_corrected_paragraph(results: List[Dict[str, Any]]) -> str:
    """Build a coherent paragraph preserving text flow without dropping unverified claims."""
    sentences = []
    for item in results:
        status = item.get("status")
        orig = item.get("sentence", "").strip()

        if status == "Contradicted" and item.get("correction") and item.get("correction_validated") is True:
            # Replace contradicted sentence with validated correction
            sentences.append(item.get("correction", "").strip())
        else:
            # Preserve original supported or unverified sentence so document integrity is not destroyed
            if orig:
                sentences.append(orig)

    return " ".join(sentences)

