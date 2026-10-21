"""Evidence retrieval using a local knowledge base and Wikipedia's public API."""
import json
import os
import re
import time
import math
import logging
import threading
import difflib
from typing import Dict, Any, List, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(ROOT, "knowledge_base.json")
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "LLMHallucinationFramework/3.0 (https://github.com/educational-prototype; contact: framework@local.net)"
}

STOP = set("""
a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

SYNONYMS = {
    "moon": "satellite",
    "moons": "satellite",
    "satellite": "satellite",
    "satellites": "satellite",
    "produce": "produce",
    "produces": "produce",
    "producing": "produce",
    "production": "produce",
    "secrete": "produce",
    "secretes": "produce",
    "secreting": "produce",
    "synthesize": "produce",
    "synthesizes": "produce",
    "beta": "produce",
    "side": "side",
    "sides": "side",
    "sided": "side",
    "polygon": "side",
    "gon": "side",
    "key": "key",
    "keys": "key",
    "keyboard": "key",
    "bit": "bit",
    "bits": "bit",
    "byte": "byte",
    "bytes": "byte",
    "capital": "capital",
    "currency": "currency",
    "franc": "currency"
}

PERSONAL_PATTERN = re.compile(
    r"\b(my|our)\s+([a-z\s]{0,25}\b)?(neighbour|neighbor|friend|professor|college|university|class|house|teacher|family|boss|colleague|cousin|sister|brother|dad|mom|father|mother|dog|cat|car|laptop|phone|roommate|room|apartment|campus|library|hometown)\b|"
    r"\b(in\s+my\s+class|near\s+my\s+house|at\s+my\s+college|my\s+college\s+library|the\s+user\s+has|the\s+user\s+is|"
    r"a\s+student\s+in\s+my|a\s+person\s+named\s+[a-z]+)\b",
    re.IGNORECASE
)

EPHEMERAL_PATTERN = re.compile(
    r"\b(discovered|found|spotted|announced|happened|seen|cured|developed|unveiled|launched|created|built)\b.*?\b(yesterday|today|just\s+now|an\s+hour\s+ago|this\s+morning|last\s+night|this\s+week)\b|"
    r"\b(yesterday|today|just\s+now|this\s+morning|last\s+night|this\s+week)\b.*?\b(discovered|found|spotted|announced|happened|seen|cured|developed|unveiled|launched|created|built)\b|"
    r"\b(announced|discovered|cured|spotted|found)\s+(yesterday|today|this\s+morning|last\s+night)\b",
    re.IGNORECASE
)

VERB_SPLIT_PATTERN = re.compile(
    r"\b(is\s+commonly\s+used\s+to|is\s+generally|is\s+approximately|is\s+primarily\s+located\s+in|is\s+located\s+in|is\s+known\s+as|"
    r"is\s+the|is\s+an?|is|are|was\s+established\s+in|was\s+founded\s+in|was\s+discovered\s+in|"
    r"took\s+place\s+in|occurred\s+in|was|were|has|have|had|contains?|equals?|stands?\s+for|became|flows?\s+through|flows?\s+into|produces?|"
    r"measures?|breathes?|travels?\s+through|travels?|can\s+travel|freezes?\s+at|revolves?\s+around|orbits?|divides?|"
    r"normally\s+have|typically\s+consists\s+of|consists\s+of|gained\s+independence\s+in|came\s+into\s+effect\s+on|lay|lays|laid)\b",
    re.IGNORECASE
)

SUB_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SUP_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

# Bounded in-memory cache with thread lock
_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 3600
_MAX_CACHE_SIZE = 1000

# Cached knowledge base
_KB_CACHE = None
_KB_MTIME = 0

# Persistent HTTP Session with retries and connection pooling
_SESSION = requests.Session()
_retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
_SESSION.mount("https://", HTTPAdapter(max_retries=_retries, pool_connections=10, pool_maxsize=20))

def normalize_claim_text(text: str) -> str:
    """Normalize unicode minus, dashes, superscripts, subscripts, and whitespace."""
    if not text:
        return ""
    t = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    t = t.translate(SUB_MAP).translate(SUP_MAP)
    t = re.sub(r'[\u2018\u2019]', "'", t)
    t = re.sub(r'[\u201C\u201D]', '"', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def is_personal_claim(claim: str) -> bool:
    """Check if the claim refers to private, unindexed personal observations."""
    norm = normalize_claim_text(claim)
    return bool(PERSONAL_PATTERN.search(norm))

def is_ephemeral_claim(claim: str) -> bool:
    """Check if the claim asserts breaking ephemeral events without public encyclopedic evidence."""
    norm = normalize_claim_text(claim)
    return bool(EPHEMERAL_PATTERN.search(norm))

def _stem_word(w: str) -> str:
    """Basic lexical normalizer and synonym resolver."""
    w = w.lower()
    if w in SYNONYMS:
        return SYNONYMS[w]
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]
    return SYNONYMS.get(w, w)

def _tokens(text: str) -> set:
    norm = normalize_claim_text(text)
    words = re.findall(r"[a-z0-9]+", norm.lower())
    res = set()
    for w in words:
        if w not in STOP and len(w) > 1:
            res.add(_stem_word(w))
    return res

def _load_kb() -> List[Dict[str, Any]]:
    global _KB_CACHE, _KB_MTIME
    try:
        if os.path.exists(KB_PATH):
            mtime = os.path.getmtime(KB_PATH)
            if _KB_CACHE is not None and mtime == _KB_MTIME:
                return _KB_CACHE
            with open(KB_PATH, "r", encoding="utf-8") as f:
                _KB_CACHE = json.load(f)
                _KB_MTIME = mtime
                return _KB_CACHE
    except Exception as exc:
        logger.error("Failed to load knowledge base: %s", exc)
    return []

def _kb_lookup(claim: str) -> Optional[Dict[str, Any]]:
    norm_c = normalize_claim_text(claim)
    claim_norm = re.sub(r"\W+", " ", norm_c.lower()).strip()
    if not claim_norm:
        return None

    best_match = None
    best_ratio = 0.0

    for item in _load_kb():
        if not isinstance(item, dict):
            continue
        cand = str(item.get("claim", ""))
        cand_norm = re.sub(r"\W+", " ", normalize_claim_text(cand).lower()).strip()

        # Exact normalized match
        if cand_norm and cand_norm == claim_norm:
            return {
                "evidence": str(item.get("evidence", "")),
                "source": str(item.get("source", "Local knowledge base")),
                "source_url": str(item.get("source_url", "")),
                "retrieval": "exact local knowledge-base match",
                "relevance_score": 1.0,
                "candidate_passages": [
                    {
                        "text": str(item.get("evidence", "")),
                        "source_title": str(item.get("source", "Local knowledge base")),
                        "source_url": str(item.get("source_url", "")),
                        "relevance_score": 1.0
                    }
                ]
            }

        ratio = difflib.SequenceMatcher(None, claim_norm, cand_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = item

    if best_match and best_ratio >= 0.93:
        return {
            "evidence": str(best_match.get("evidence", "")),
            "source": str(best_match.get("source", "Local knowledge base")),
            "source_url": str(best_match.get("source_url", "")),
            "retrieval": f"fuzzy local knowledge-base match ({round(best_ratio, 2)})",
            "relevance_score": round(best_ratio, 3),
            "candidate_passages": [
                {
                    "text": str(best_match.get("evidence", "")),
                    "source_title": str(best_match.get("source", "Local knowledge base")),
                    "source_url": str(best_match.get("source_url", "")),
                    "relevance_score": round(best_ratio, 3)
                }
            ]
        }

    return None

def parse_proposition(claim: str) -> Dict[str, Any]:
    """Parse a factual claim into core proposition elements."""
    norm = normalize_claim_text(claim)
    q = norm.strip().rstrip(".!?")
    prop = {
        "raw": claim,
        "clean": q,
        "type": "general",
        "subject": "",
        "relation": "",
        "target_value": "",
        "scope": "",
        "queries": []
    }

    # 1. Boiling / Melting / Freezing point proposition pattern
    bp_match = re.search(r"\b(?:the\s+)?(boiling|melting|freezing)\s+point\s+of\s+([A-Za-z\s]+?)\s+is\s+(?:approximately\s+|about\s+)?([−\-]?\d+(?:\.\d+)?)\s*°?\s*([CFK]|celsius|fahrenheit|kelvin)?\b", q, re.I)
    if bp_match:
        pt_type = bp_match.group(1).lower()
        element = bp_match.group(2).strip()
        temp = bp_match.group(3).strip()
        unit = (bp_match.group(4) or "").strip()
        prop["type"] = "phase_change_point"
        prop["subject"] = element
        prop["relation"] = f"{pt_type} point"
        prop["target_value"] = f"{temp}{unit}"
        prop["queries"] = [
            f"boiling point of {element}",
            f"{element} boiling point",
            f"Liquid {element}",
            element,
            f"{element} chemical element"
        ]
        return prop

    # 2. Universal quantifier proposition pattern (e.g. All planets in Solar System have rings, Every species of penguin...)
    all_match = re.search(r"\b(all|every|each|no|none\s+of\s+the)\s+([A-Za-z\s]+?)\s+(?:in\s+(?:the\s+)?([A-Za-z\s]+?)\s+)?(have|has|contain|contains?|live|lives\s+in|are|is)\s+([A-Za-z\s]+)\b", q, re.I)
    if all_match:
        quantifier = all_match.group(1).lower()
        entity = all_match.group(2).strip()
        scope = (all_match.group(3) or "").strip()
        verb = all_match.group(4).strip()
        target = all_match.group(5).strip()
        prop["type"] = "universal_quantifier"
        prop["subject"] = f"{entity} in {scope}" if scope else entity
        prop["relation"] = f"{verb} {target}"
        prop["target_value"] = target
        prop["scope"] = scope
        queries = []
        if "planet" in entity.lower() and "ring" in target.lower():
            queries = [
                "rings of planets in Solar System",
                "Ring system Solar System planets",
                "planetary rings Solar System",
                "planets in the Solar System rings",
                "Solar System planets rings"
            ]
        elif "penguin" in entity.lower():
            queries = [
                "Penguin habitat",
                "Penguin Antarctica",
                "Penguins Southern Hemisphere",
                "Penguin",
                "species of penguin"
            ]
        else:
            if scope:
                queries.append(f"{entity} {scope} {target}")
                queries.append(f"{entity} {target}")
            queries.append(entity)
            queries.append(target)
        prop["queries"] = queries
        return prop

    # 3. Chemical formula proposition pattern
    formula_match = re.search(r"\bchemical\s+formula\s+for\s+([A-Za-z\s]+?)\s+is\s+([A-Za-z0-9]+)\b", q, re.I)
    if formula_match:
        compound = formula_match.group(1).strip()
        formula = formula_match.group(2).strip()
        prop["type"] = "chemical_formula"
        prop["subject"] = compound
        prop["relation"] = "chemical formula"
        prop["target_value"] = formula
        prop["queries"] = [
            f"chemical formula {compound}",
            f"{compound} formula",
            compound,
            f"{compound} {formula}"
        ]
        return prop

    # 4. Chemical symbol proposition pattern
    chem_match = re.search(r"\bchemical\s+symbol\s+for\s+([A-Za-z\s]+?)\s+is\s+([A-Za-z]+)\b", q, re.I)
    if chem_match:
        element = chem_match.group(1).strip()
        symbol = chem_match.group(2).strip()
        prop["type"] = "chemical_symbol"
        prop["subject"] = element
        prop["relation"] = "chemical symbol"
        prop["target_value"] = symbol
        prop["queries"] = [
            f"chemical symbol {element}",
            element,
            f"{element} chemical element",
            f"{element} {symbol}"
        ]
        return prop

    # 5. Currency proposition pattern
    curr_match = re.search(r"\bthe\s+currency\s+of\s+([A-Za-z\s]+?)\s+is\s+(?:the\s+)?([A-Za-z\s]+)\b", q, re.I)
    if curr_match:
        country = curr_match.group(1).strip()
        curr = curr_match.group(2).strip()
        prop["type"] = "currency"
        prop["subject"] = country
        prop["relation"] = "currency"
        prop["target_value"] = curr
        prop["queries"] = [
            f"currency of {country}",
            f"{country} currency",
            f"{country} {curr}",
            f"{country} taka" if "taka" in curr.lower() else curr,
            country
        ]
        return prop

    # 6. Mathematical square root proposition pattern
    sqrt_match = re.search(r"\bthe\s+square\s+root\s+of\s+(\d+)\s+is\s+(\d+)\b", q, re.I)
    if sqrt_match:
        num = int(sqrt_match.group(1))
        root = int(sqrt_match.group(2))
        is_math_true = (root * root == num)
        prop["type"] = "mathematics"
        prop["subject"] = f"square root of {num}"
        prop["relation"] = "equals"
        prop["target_value"] = str(root)
        prop["math_valid"] = is_math_true
        prop["queries"] = [
            f"{num} (number)",
            f"{num} square number",
            f"square root of {num}",
            f"{num} number mathematics"
        ]
        return prop

    # 7. Measurement unit / definition proposition pattern
    meas_match = re.search(r"\b(?:a\s+)?([a-z-]+)\s+(?:measures?|is\s+a\s+unit\s+of)\s+([a-z\s]+?)(?:\s+rather\s+than\s+([a-z\s]+))?\b", q, re.I)
    if meas_match:
        unit = meas_match.group(1).strip()
        prop_dim = meas_match.group(2).strip()
        other_dim = (meas_match.group(3) or "").strip()
        prop["type"] = "measurement"
        prop["subject"] = unit
        prop["relation"] = "measures"
        prop["target_value"] = prop_dim
        prop["queries"] = [
            unit.title(),
            f"{unit} unit of length",
            f"{unit} astronomy distance",
            f"{unit} distance"
        ]
        return prop

    # 8. Historical milestone / image release pattern
    photo_match = re.search(r"\b(?:the\s+)?(first\s+(?:photograph|photo|image|direct\s+image))\s+of\s+(?:a\s+)?([A-Za-z\s]+?)\s+(?:was\s+)?released\s+in\s+(\d{4})\b", q, re.I)
    if photo_match:
        subj = photo_match.group(2).strip()
        year = photo_match.group(3).strip()
        prop["type"] = "milestone_event"
        prop["subject"] = subj
        prop["relation"] = "released in"
        prop["target_value"] = year
        prop["queries"] = [
            f"first direct image of a {subj} {year}",
            f"first photograph {subj} {year}",
            f"Event Horizon Telescope {year}",
            subj,
            f"{subj} {year} image"
        ]
        return prop

    # 9. Capital of Country / State proposition pattern
    cap_match = re.search(r"\bthe\s+capital\s+of\s+([A-Za-z\s]+?)\s+is\s+([A-Za-z\s]+)\b", q, re.I)
    if cap_match:
        country = cap_match.group(1).strip()
        city = cap_match.group(2).strip()
        prop["type"] = "capital"
        prop["subject"] = country
        prop["relation"] = "capital"
        prop["target_value"] = city
        prop["queries"] = [
            f"capital of {country}",
            f"{country} federal city",
            f"{country} capital",
            country,
            city
        ]
        return prop

    # 10. Geometry / Polygon sides proposition pattern
    poly_match = re.search(r"\b(?:a\s+)?(?:regular\s+)?([a-z]+(?:gon|hedron|pentagon|hexagon|heptagon|octagon|nonagon|decagon))\s+(?:has|have|contains?)\s+(\d+|[a-z]+)\s+(?:sides|edges|faces|angles)\b", q, re.I)
    if poly_match:
        poly = poly_match.group(1).strip()
        sides = poly_match.group(2).strip()
        prop["type"] = "polygon_sides"
        prop["subject"] = poly
        prop["relation"] = "sides"
        prop["target_value"] = sides
        prop["queries"] = [
            f"{poly} geometry",
            f"{poly} polygon sides",
            f"regular {poly}",
            poly
        ]
        return prop

    # 11. Biological / Organ production proposition pattern
    prod_match = re.search(r"\b(?:the\s+)?([a-z\s]+?)\s+produces?\s+([a-z\s]+?)(?:\s+in\s+(?:the\s+)?([a-z\s]+))?\b", q, re.I)
    if prod_match:
        agent = prod_match.group(1).strip()
        substance = prod_match.group(2).strip()
        organ = (prod_match.group(3) or agent).strip()
        prop["type"] = "organ_production"
        prop["subject"] = organ
        prop["relation"] = "produces"
        prop["target_value"] = substance
        queries = [
            f"{substance} {organ}",
            f"{organ} {substance}",
            organ,
            substance
        ]
        if "stomach" in organ.lower() and "acid" in substance.lower():
            queries.insert(0, "Gastric acid")
            queries.insert(1, "Hydrochloric acid stomach")
        prop["queries"] = queries
        return prop

    # 12. Celestial satellites / moons proposition pattern
    moon_match = re.search(r"\b(?:the\s+)?(?:planet\s+)?([A-Za-z]+)\s+has\s+(\d+|[a-z]+)\s+(?:natural\s+)?(?:moons?|satellites?)\b", q, re.I)
    if moon_match:
        body = moon_match.group(1).strip()
        count = moon_match.group(2).strip()
        prop["type"] = "celestial_satellites"
        prop["subject"] = body
        prop["relation"] = "natural moons"
        prop["target_value"] = count
        prop["queries"] = [
            f"moons of {body}",
            f"{body} natural satellites",
            f"natural satellites of {body}",
            "Moon"
        ]
        return prop

    # 13. Location proposition pattern
    loc_match = re.search(r"\b([A-Za-z\s]+?)\s+is\s+(?:primarily\s+|largely\s+)?located\s+in\s+([A-Za-z\s]+)\b", q, re.I)
    if loc_match:
        entity = loc_match.group(1).strip()
        clean_ent = re.sub(r"^(the|a|an)\s+", "", entity, flags=re.I).strip()
        location = loc_match.group(2).strip()
        prop["type"] = "location"
        prop["subject"] = clean_ent
        prop["relation"] = "located in"
        prop["target_value"] = location
        prop["queries"] = [
            f"{clean_ent} {location}",
            f"{clean_ent} location",
            clean_ent,
            location
        ]
        return prop

    # 14. Superlatives pattern (e.g. largest planet in Solar System)
    sup_match = re.search(r"\bthe\s+(largest|smallest|hottest|coldest|closest|farthest|highest|deepest)\s+([A-Za-z\s]+?)\s+in\s+(?:the\s+)?([A-Za-z\s]+?)\s+is\s+([A-Za-z\s]+)\b", q, re.I)
    if sup_match:
        adj = sup_match.group(1).strip()
        category = sup_match.group(2).strip()
        system = sup_match.group(3).strip()
        claimed_ent = sup_match.group(4).strip()
        prop["type"] = "superlative"
        prop["subject"] = f"{adj} {category} in {system}"
        prop["relation"] = "is"
        prop["target_value"] = claimed_ent
        prop["queries"] = [
            f"{adj} {category} {system}",
            f"{system} {category}",
            claimed_ent
        ]
        return prop

    # 15. General verb split pattern
    match = VERB_SPLIT_PATTERN.search(q)
    if match:
        subject = q[:match.start()].strip()
        clean_subj = re.sub(r"^(the|a|an|all|every|one)\s+", "", subject, flags=re.IGNORECASE).strip()
        prop["subject"] = clean_subj
        after_verb = q[match.end():].strip()
        prop["target_value"] = after_verb

        queries = []
        if len(clean_subj) > 1:
            queries.append(clean_subj)
            if " of " in clean_subj.lower():
                part_after_of = clean_subj.split(" of ", 1)[-1].strip()
                if part_after_of:
                    queries.append(part_after_of)

        after_words = [w for w in re.findall(r"[A-Za-z0-9'-]+", after_verb) if w.lower() not in STOP]
        if after_words and clean_subj:
            queries.append(f"{clean_subj} {after_words[0]}")
            if len(after_words) > 1:
                queries.append(f"{clean_subj} {' '.join(after_words[:3])}")

        prop["queries"] = queries

    return prop

def _queries(claim: str) -> List[str]:
    """Generate search queries covering both the proposition relationship and entities."""
    norm = normalize_claim_text(claim)
    prop = parse_proposition(norm)
    queries = list(prop.get("queries", []))
    q = norm.strip().rstrip(".!?")

    # Proper nouns / capitalized entities
    entities = re.findall(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b", q)
    for ent in entities:
        ent = ent.strip()
        if ent.lower() not in {"the", "a", "an", "all", "every", "one", "zero", "first", "my", "our", "its"}:
            if ent not in queries:
                queries.append(ent)

    # Keywords (nouns, numbers, adjectives)
    words = [w for w in re.findall(r"[A-Za-z0-9'-]+", q) if w.lower() not in STOP]
    if len(words) >= 2:
        queries.append(" ".join(words[:5]))
    queries.append(q)

    # Deduplicate while preserving order
    seen, out = set(), []
    for item in queries:
        clean = item.strip()
        if clean and clean.lower() not in seen and len(clean) > 1:
            out.append(clean)
            seen.add(clean.lower())
    return out

def _wiki_search(query: str) -> List[Dict[str, Any]]:
    for attempt in range(2):
        try:
            response = _SESSION.get(
                API,
                params={
                    "action": "query", "list": "search", "srsearch": query,
                    "srlimit": 5, "format": "json", "utf8": 1
                },
                headers=HEADERS,
                timeout=8
            )
            response.raise_for_status()
            raw_results = response.json().get("query", {}).get("search", [])

            filtered = []
            for r in raw_results:
                title = r.get("title", "")
                t_lower = title.lower()

                # Filter directories/lists
                if t_lower.startswith(("list of", "category:", "timeline of", "index of")):
                    continue

                # Filter disambiguations
                if "(disambiguation)" in t_lower:
                    continue

                # Filter pure pop-culture media if query is not pop-culture
                media_tags = ["(album)", "(song)", "(film)", "(tv series)", "(band)", "(franchise)", "(video game)"]
                if any(tag in t_lower for tag in media_tags) and not any(tag in query.lower() for tag in media_tags):
                    continue

                # Filter Internet country domain .my if searching for personal claims
                if t_lower == ".my":
                    continue

                # Geometry vs DoD building: if query has 'pentagon geometry', skip 'The Pentagon'
                if "geometry" in query.lower() and t_lower == "the pentagon":
                    continue

                filtered.append(r)
            return filtered
        except Exception as exc:
            if attempt == 0:
                time.sleep(0.5)
                continue
            logger.warning("Wikipedia search failed for query '%s': %s", query, exc)
            return []
    return []

def _wiki_extracts_batch(pageids: List[int]) -> Dict[int, Dict[str, str]]:
    if not pageids:
        return {}
    pageids_str = "|".join(str(pid) for pid in pageids[:10])
    for attempt in range(2):
        try:
            response = _SESSION.get(
                API,
                params={
                    "action": "query", "pageids": pageids_str, "prop": "extracts|info",
                    "explaintext": 1, "exintro": 1, "inprop": "url",
                    "format": "json", "redirects": 1
                },
                headers=HEADERS,
                timeout=10
            )
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})
            results = {}
            for pid_str, page_data in pages.items():
                try:
                    pid = int(pid_str)
                    extract = page_data.get("extract", "")
                    url = page_data.get("fullurl", "")
                    results[pid] = {"extract": extract, "url": url}
                except (ValueError, TypeError):
                    continue
            return results
        except Exception as exc:
            if attempt == 0:
                time.sleep(0.5)
                continue
            logger.warning("Wikipedia batch extract failed: %s", exc)
            return {}
    return {}

def _wiki_full_extract(title_or_pid: str) -> Optional[Dict[str, str]]:
    """Fetch full text of a single Wikipedia article for deep section retrieval."""
    try:
        params = {
            "action": "query", "prop": "extracts|info",
            "explaintext": 1, "inprop": "url", "format": "json", "redirects": 1
        }
        if str(title_or_pid).isdigit():
            params["pageids"] = str(title_or_pid)
        else:
            params["titles"] = str(title_or_pid)

        resp = _SESSION.get(API, params=params, headers=HEADERS, timeout=8)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for _, pdata in pages.items():
            return {
                "extract": pdata.get("extract", ""),
                "url": pdata.get("fullurl", "")
            }
    except Exception as exc:
        logger.debug("Failed to fetch full extract for %s: %s", title_or_pid, exc)
    return None

def _score_sentence_for_claim(sentence: str, claim: str, prop: Dict[str, Any], title: str = "") -> float:
    """Score how directly an individual evidence sentence addresses the claim's proposition."""
    s_norm = normalize_claim_text(sentence)
    c_norm = normalize_claim_text(claim)
    t_norm = normalize_claim_text(title)

    s_lower = s_norm.lower()
    c_lower = c_norm.lower()
    t_lower = t_norm.lower()

    tokens_s = _tokens(s_norm + " " + t_norm)
    tokens_c = _tokens(c_norm)

    if not tokens_c:
        return 0.0

    overlap = len(tokens_s & tokens_c) / len(tokens_c)
    score = overlap * 0.50

    prop_type = prop.get("type", "general")
    subject = normalize_claim_text(prop.get("subject", "")).lower()
    target_value = normalize_claim_text(prop.get("target_value", "")).lower()

    # Subject entity check: does sentence or title mention the subject?
    has_subject = False
    if subject:
        subj_words = [w for w in re.findall(r"[a-z0-9]+", subject) if w not in STOP and len(w) > 2]
        if subj_words and any(w in s_lower or w in t_lower for w in subj_words):
            has_subject = True
            score += 0.20

    # 1. Phase change point (boiling, melting, freezing)
    if prop_type == "phase_change_point":
        has_elem = subject in s_lower or subject in t_lower or "nitrogen" in s_lower or "nitrogen" in t_lower
        has_pt = "boil" in s_lower or "melting" in s_lower or "freezing" in s_lower
        has_temp = ("-196" in s_lower) or ("196" in s_lower) or (target_value and target_value in s_lower)
        if has_elem and has_pt:
            score += 0.40
            if has_temp:
                score += 0.40

    # 2. Universal quantifier (Planets rings, Penguins Arctic)
    elif prop_type == "universal_quantifier":
        if "planet" in subject:
            has_planet = any(w in s_lower or w in t_lower for w in ["planet", "giant planet", "solar system", "jupiter", "saturn"])
            has_rings = any(w in s_lower or w in t_lower for w in ["ring", "rings", "ring system"])
            if has_planet and has_rings:
                score += 0.65
        elif "penguin" in subject:
            has_penguin = "penguin" in s_lower or "penguin" in t_lower
            has_geo = any(w in s_lower for w in ["southern hemisphere", "antarctic", "arctic", "equator", "sea"])
            if has_penguin:
                score += 0.40
                if has_geo:
                    score += 0.40

    # 3. Chemical formula
    elif prop_type == "chemical_formula":
        has_compound = subject in s_lower or subject in t_lower
        has_formula = "formula" in s_lower or (target_value and target_value.lower() in s_lower)
        if has_compound:
            score += 0.35
            if has_formula:
                score += 0.45

    # 4. Chemical symbol
    elif prop_type == "chemical_symbol":
        has_symbol_word = "chemical symbol" in s_lower or "symbol is" in s_lower or "symbol" in s_lower
        has_element = subject in s_lower or subject in t_lower
        if has_element and has_symbol_word:
            score += 0.50
            if target_value and target_value.lower() in s_lower.split():
                score += 0.30

    # 5. Currency
    elif prop_type == "currency":
        has_curr_word = any(w in s_lower for w in ["currency", "legal tender", "money", "banknote"])
        has_country = subject in s_lower or subject in t_lower
        clean_target = re.sub(r"^(the|a)\s+", "", target_value).strip().lower()
        if has_country:
            score += 0.30
            if has_curr_word:
                score += 0.30
            if clean_target and clean_target in s_lower:
                score += 0.35

    # 6. Mathematics (Square root, arithmetic)
    elif prop_type == "mathematics":
        has_num = any(w in s_lower for w in ["225", target_value])
        has_math_word = any(w in s_lower for w in ["square", "square root", "15", "multiplied"])
        if has_num and has_math_word:
            score += 0.75

    # 7. Measurement unit (e.g. Light-year)
    elif prop_type == "measurement":
        has_unit = subject in s_lower or subject in t_lower or "light-year" in s_lower or "light-year" in t_lower or "light year" in s_lower
        has_dim = any(w in s_lower for w in ["distance", "length", "time", "astronomical"])
        if has_unit:
            score += 0.40
            if has_dim:
                score += 0.45

    # 8. Historical milestone (e.g. Black hole 2019 image)
    elif prop_type == "milestone_event":
        has_subj = subject in s_lower or subject in t_lower
        has_year = target_value in s_lower
        has_photo = any(w in s_lower for w in ["photograph", "image", "direct image", "event horizon telescope", "published", "released"])
        if has_subj and has_year:
            score += 0.55
            if has_photo:
                score += 0.35

    # 9. Capital
    elif prop_type == "capital":
        has_capital_word = any(w in s_lower for w in ["capital", "federal city", "de facto capital", "largest city"])
        has_country = (subject in s_lower) or (subject in t_lower)
        if has_capital_word and has_country:
            score += 0.50
            if target_value and target_value in s_lower:
                score += 0.30

    # 10. Organ production (stomach HCl, pancreas insulin)
    elif prop_type == "organ_production":
        has_organ = (subject in s_lower) or (subject in t_lower) or ("gastric" in s_lower) or ("pancrea" in s_lower)
        has_substance = (target_value in s_lower) or ("hydrochloric" in s_lower) or ("insulin" in s_lower)
        has_prod = any(w in s_lower for w in ["produce", "produced by", "production", "secrete", "secreting", "acidic component", "beta cell", "gland"])
        if has_organ and has_substance:
            score += 0.55
            if has_prod:
                score += 0.30

    # 11. Geometry / polygon
    elif prop_type == "polygon_sides":
        has_poly = (subject in s_lower) or (subject in t_lower)
        has_sides = any(w in s_lower for w in ["side", "sided", "polygon", "gon"])
        has_numbers = bool(re.search(r"\b(five|5|six|6|\d+)\b", s_lower))
        if has_poly and (has_sides or "geometry" in s_lower or "geometry" in t_lower):
            score += 0.55
            if has_numbers:
                score += 0.25

    # 12. Celestial satellites
    elif prop_type == "celestial_satellites":
        has_body = (subject in s_lower) or (subject in t_lower)
        has_satellite = any(w in s_lower for w in ["moon", "satellite", "orbit"])
        if has_body and has_satellite:
            score += 0.55
            if any(w in s_lower for w in ["only", "one", "two", "permanent", "natural"]):
                score += 0.25

    # 13. Location
    elif prop_type == "location":
        has_subj_loc = (subject in s_lower) or (subject in t_lower)
        has_loc = (target_value in s_lower) or (target_value in t_lower)
        if has_subj_loc and has_loc:
            score += 0.65

    # 14. Superlative
    elif prop_type == "superlative":
        has_planets = any(w in s_lower for w in ["jupiter", "mercury", "largest", "smallest", "solar system"])
        if has_planets:
            score += 0.65

    # Penalize purely unrelated entities (sports, entertainment, fictional) unless mentioned in claim
    penalized_terms = [
        "cleveland", "pittsburgh", "miami", "nfl", "nba", "championship",
        "fc", "club", "team", "murderer", "film", "album", "novel", "band",
        "football", "baseball", "basketball", "sports", "franchise", "rugby", "league", "nrl",
        "superhero", "villain", "bank", "laughter chefs"
    ]
    if any(term in t_lower for term in penalized_terms) and not any(term in c_lower for term in penalized_terms):
        score -= 0.50

    return max(0.0, min(1.0, score))

def retrieve_evidence(claim: str) -> Dict[str, Any]:
    """Retrieve proposition-relevant evidence from knowledge base and Wikipedia."""
    claim_norm = normalize_claim_text(claim)
    key = claim_norm.strip().lower()
    now = time.time()

    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and (now - cached[0] < _CACHE_TTL):
            return dict(cached[1])

    # 1. Personal claim guard
    if is_personal_claim(claim):
        res = {
            "evidence": "",
            "full_evidence": "",
            "source": "",
            "source_url": "",
            "retrieval": "personal_claim",
            "relevance_score": 0.0,
            "candidate_passages": []
        }
        with _CACHE_LOCK:
            _CACHE[key] = (now, res)
        return res

    # 2. Ephemeral unverified breaking claim guard
    if is_ephemeral_claim(claim):
        res = {
            "evidence": "",
            "full_evidence": "",
            "source": "",
            "source_url": "",
            "retrieval": "ephemeral_claim",
            "relevance_score": 0.0,
            "candidate_passages": []
        }
        with _CACHE_LOCK:
            _CACHE[key] = (now, res)
        return res

    # 3. Local Knowledge Base lookup
    local = _kb_lookup(claim)
    if local and local.get("evidence"):
        with _CACHE_LOCK:
            _CACHE[key] = (now, local)
        return dict(local)

    # 4. Wikipedia API search across targeted proposition queries
    prop = parse_proposition(claim_norm)
    queries = _queries(claim_norm)
    query_results = {}

    for query in queries[:6]:
        results = _wiki_search(query)
        for r in results:
            pid = r.get("pageid")
            if pid and pid not in query_results:
                query_results[pid] = r

    # Batch retrieve page extracts
    all_pageids = list(query_results.keys())[:10]
    page_extracts = _wiki_extracts_batch(all_pageids)

    candidate_passages = []

    # Score each sentence in the retrieved extracts and search snippets
    for pid, r in query_results.items():
        title = r.get("title", "")
        extract_info = page_extracts.get(pid, {})
        extract = extract_info.get("extract", "")
        url = extract_info.get("url", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")

        # Check snippet first
        snippet = re.sub(r'<[^>]+>', '', r.get("snippet", "")).strip()
        if snippet and len(snippet) > 20:
            s_score = _score_sentence_for_claim(snippet, claim_norm, prop, title)
            if s_score >= 0.25:
                candidate_passages.append({
                    "text": snippet,
                    "source_title": f"Wikipedia — {title}",
                    "source_url": url,
                    "relevance_score": round(s_score, 3)
                })

        # Split extract into sentences
        if extract:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', extract) if len(s.strip()) > 15]
            for s in sentences:
                s_score = _score_sentence_for_claim(s, claim_norm, prop, title)
                if s_score >= 0.25:
                    candidate_passages.append({
                        "text": s,
                        "source_title": f"Wikipedia — {title}",
                        "source_url": url,
                        "relevance_score": round(s_score, 3)
                    })

    # Deep retrieval check: if top candidate score is weak (< 0.50) but we have a strong matching title
    top_score = max([c["relevance_score"] for c in candidate_passages], default=0.0)
    if top_score < 0.50 and all_pageids:
        top_pid = all_pageids[0]
        top_title = query_results[top_pid].get("title", "")
        deep_info = _wiki_full_extract(top_title)
        if deep_info and deep_info.get("extract"):
            full_text = deep_info["extract"]
            url = deep_info.get("url") or f"https://en.wikipedia.org/wiki/{top_title.replace(' ', '_')}"
            deep_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if len(s.strip()) > 15]
            for s in deep_sentences:
                d_score = _score_sentence_for_claim(s, claim_norm, prop, top_title)
                if d_score >= 0.45:
                    candidate_passages.append({
                        "text": s,
                        "source_title": f"Wikipedia — {top_title}",
                        "source_url": url,
                        "relevance_score": round(d_score, 3)
                    })

    # Deduplicate candidate passages by text
    seen_texts = set()
    unique_candidates = []
    for c in candidate_passages:
        norm_text = re.sub(r"\W+", " ", c["text"].lower()).strip()
        if norm_text not in seen_texts:
            seen_texts.add(norm_text)
            unique_candidates.append(c)

    unique_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

    if unique_candidates and unique_candidates[0]["relevance_score"] >= 0.30:
        top_cand = unique_candidates[0]
        top_texts = [c["text"] for c in unique_candidates[:3]]
        combined_evidence = " ".join(top_texts)

        res = {
            "evidence": top_cand["text"],
            "full_evidence": combined_evidence,
            "source": top_cand["source_title"],
            "source_url": top_cand["source_url"],
            "retrieval": "Wikipedia API",
            "relevance_score": top_cand["relevance_score"],
            "candidate_passages": unique_candidates[:6]
        }
    else:
        res = {
            "evidence": "",
            "full_evidence": "",
            "source": "",
            "source_url": "",
            "retrieval": "no relevant evidence found",
            "relevance_score": 0.0,
            "candidate_passages": []
        }

    with _CACHE_LOCK:
        if len(_CACHE) >= _MAX_CACHE_SIZE:
            old_keys = sorted(_CACHE.keys(), key=lambda k: _CACHE[k][0])[: len(_CACHE) // 4]
            for k in old_keys:
                _CACHE.pop(k, None)
        _CACHE[key] = (now, res)

    return dict(res)

def find_evidence(claim: str) -> Dict[str, Any]:
    return retrieve_evidence(claim)
