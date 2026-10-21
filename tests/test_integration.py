"""End-to-end integration tests verifying live retrieval, verification, and API endpoints."""
import unittest
from app import app
from services.evidence import retrieve_evidence
from services.verification import (
    verify_claim,
    SUPPORTED,
    CONTRADICTED,
    INSUFFICIENT_EVIDENCE
)

class TestLivePipelineIntegration(unittest.TestCase):
    """End-to-end tests through live retrieval and verification pipeline."""

    def setUp(self):
        self.client = app.test_client()

    def test_live_case_1_seoul_supported(self):
        claim = "The capital of South Korea is Seoul."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertEqual(res["confidence_type"], "estimated")
        self.assertIsNone(res["correction"])
        self.assertGreater(len(res["evidence"]), 0)

    def test_live_case_2_piano_88_supported(self):
        claim = "A standard piano has 88 keys."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])

    def test_live_case_3_amazon_rainforest_supported(self):
        claim = "The Amazon rainforest is primarily located in South America."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])

    def test_live_case_4_byte_bits_supported(self):
        claim = "A byte typically consists of eight bits."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])

    def test_live_case_5_swiss_franc_supported(self):
        claim = "The currency of Switzerland is the Swiss franc."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])

    def test_live_case_6_pyramid_giza_supported(self):
        claim = "The Great Pyramid of Giza is located in Egypt."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])

    def test_live_case_7_insulin_pancreas_supported(self):
        claim = "The human body produces insulin in the pancreas."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])

    def test_live_case_8_earth_two_moons_contradicted(self):
        claim = "The planet Earth has two natural moons."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], CONTRADICTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNotNone(res["correction"])
        self.assertIn("one natural moon", res["correction"].lower())

    def test_live_case_9_first_flight_1903_supported(self):
        claim = "The first successful powered airplane flight took place in 1903."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])

    def test_live_case_10_capital_switzerland_zurich_contradicted(self):
        claim = "The capital of Switzerland is Zurich."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], CONTRADICTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIn("Bern", res["correction"])

    def test_live_case_11_pentagon_six_sides_contradicted(self):
        claim = "A regular pentagon has six sides."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], CONTRADICTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIn("five sides", res["correction"].lower())

    def test_live_case_12_chemical_symbol_gold_ag_contradicted(self):
        claim = "The chemical symbol for gold is Ag."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], CONTRADICTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertEqual(res["correction"], "The chemical symbol for gold is Au.")

    def test_live_case_13_personal_claim_insufficient(self):
        claim = "My neighbour owns a blue car."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], INSUFFICIENT_EVIDENCE)
        self.assertEqual(res["confidence"], 0.0)
        self.assertIsNone(res["correction"])

    def test_live_case_14_ephemeral_claim_insufficient(self):
        claim = "Scientists discovered a new planet orbiting Earth yesterday."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], INSUFFICIENT_EVIDENCE)
        self.assertEqual(res["confidence"], 0.0)
        self.assertIsNone(res["correction"])

    def test_live_case_15_breaking_cure_insufficient(self):
        claim = "A cure for every type of cancer was announced this morning."
        ev = retrieve_evidence(claim)
        res = verify_claim(claim, ev)
        self.assertEqual(res["verdict"], INSUFFICIENT_EVIDENCE)
        self.assertEqual(res["confidence"], 0.0)
        self.assertIsNone(res["correction"])

    def test_api_verify_endpoint_section_8_schema(self):
        """Test POST /api/verify response conforms strictly to Section 8 schema."""
        resp = self.client.post("/api/verify", json={"claim": "A standard piano has 88 keys."})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["claim"], "A standard piano has 88 keys.")
        self.assertEqual(data["verdict"], SUPPORTED)
        self.assertIsInstance(data["confidence"], (int, float))
        self.assertEqual(data["confidence_type"], "estimated")
        self.assertIsInstance(data["explanation"], str)
        self.assertIsInstance(data["evidence"], list)
        self.assertIsNone(data["correction"])

        for ev_item in data["evidence"]:
            self.assertIn("text", ev_item)
            self.assertIn("source_title", ev_item)
            self.assertIn("source_url", ev_item)
            self.assertIn("relationship", ev_item)
            self.assertIn("relevance_score", ev_item)
            self.assertIn(ev_item["relationship"], ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"])

    def test_api_analyze_multisentence_paragraph_and_duplicate_numbers(self):
        """Test POST /api/analyze on a multi-claim paragraph and verify no duplicate numbering."""
        paragraph = (
            "1. The capital of South Korea is Seoul. "
            "2. A regular pentagon has six sides. "
            "3. My neighbour owns a blue car."
        )
        resp = self.client.post("/api/analyze", json={"text": paragraph})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["total_sentences"], 3)
        self.assertEqual(data["supported_count"], 1)
        self.assertEqual(data["contradicted_count"], 1)
        self.assertEqual(data["insufficient_evidence_count"], 1)

        # Check each sentence has no duplicate number in the claim field
        for res in data["results"]:
            self.assertFalse(res["claim"].startswith(f"{res['id']}. "))
            self.assertIn("confidence_type", res)

        # Check corrected paragraph replaces the contradicted sentence with validated correction
        corrected = data["corrected_paragraph"]
        self.assertIn("The capital of South Korea is Seoul.", corrected)
        self.assertIn("five sides", corrected.lower())
        self.assertIn("My neighbour owns a blue car.", corrected)

    def test_api_chat_endpoint(self):
        """Test POST /api/chat returns factual answer and grounded citations."""
        resp = self.client.post("/api/chat", json={"message": "The capital of South Korea is Seoul."})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["verdict"], SUPPORTED)
        self.assertIn("reply", data)
        self.assertIn("Seoul", data["reply"])
        self.assertEqual(data["confidence_type"], "estimated")

    def test_api_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["ok"])
        self.assertIn("verification_model", data)

if __name__ == "__main__":
    unittest.main()
