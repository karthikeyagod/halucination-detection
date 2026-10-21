"""Deterministic unit tests with controlled evidence fixtures for Section 8 regression suite."""
import unittest
from services.verification import (
    verify_claim,
    calculate_confidence,
    evaluate_passage_comparison,
    SUPPORTED,
    CONTRADICTED,
    INSUFFICIENT_EVIDENCE,
    ENTAILMENT,
    CONTRADICTION,
    NEUTRAL
)
from services.correction import suggest_correction, validate_correction
from services.evidence import is_personal_claim, is_ephemeral_claim, parse_proposition

class TestUnitVerificationFixtures(unittest.TestCase):
    """Deterministic regression suite verifying Section 8 test cases and edge cases."""

    # 1. Nitrogen boiling point (~ -196°C) -> SUPPORTED
    def test_case_1_nitrogen_boiling_point_supported(self):
        claim = "The boiling point of nitrogen is approximately -196°C at atmospheric pressure."
        fixture = {
            "evidence": "Liquid nitrogen has a boiling point of about -196 °C (-321 °F; 77 K) at standard atmospheric pressure.",
            "source": "Encyclopaedia Britannica — Nitrogen",
            "source_url": "https://www.britannica.com/science/nitrogen",
            "relevance_score": 1.0,
            "candidate_passages": [{
                "text": "Liquid nitrogen has a boiling point of about -196 °C (-321 °F; 77 K) at standard atmospheric pressure.",
                "source_title": "Encyclopaedia Britannica — Nitrogen",
                "source_url": "https://www.britannica.com/science/nitrogen",
                "relevance_score": 1.0
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertLessEqual(res["confidence"], 0.98)
        self.assertEqual(res["confidence_type"], "estimated")
        self.assertIsNone(res["correction"])
        self.assertIn("-196", res["explanation"])

    # 2. All planets have rings -> CONTRADICTED
    def test_case_2_planets_rings_contradicted(self):
        claim = "All planets in the Solar System have rings."
        fixture = {
            "evidence": "In the Solar System, only the four giant planets (Jupiter, Saturn, Uranus, and Neptune) have ring systems, whereas the terrestrial planets (Mercury, Venus, Earth, and Mars) do not have rings.",
            "source": "NASA Solar System Exploration — Planetary Rings",
            "source_url": "https://science.nasa.gov/solar-system/planetary-rings/",
            "relevance_score": 1.0,
            "candidate_passages": [{
                "text": "In the Solar System, only the four giant planets (Jupiter, Saturn, Uranus, and Neptune) have ring systems, whereas the terrestrial planets (Mercury, Venus, Earth, and Mars) do not have rings.",
                "source_title": "NASA Solar System Exploration — Planetary Rings",
                "source_url": "https://science.nasa.gov/solar-system/planetary-rings/",
                "relevance_score": 1.0
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], CONTRADICTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertLessEqual(res["confidence"], 0.98)
        self.assertIsNotNone(res["correction"])
        self.assertIn("four giant planets", res["correction"].lower())

    # 3. Chemical formula for ammonia is NH3 -> SUPPORTED
    def test_case_3_ammonia_formula_supported(self):
        claim = "The chemical formula for ammonia is NH3."
        fixture = {
            "evidence": "Ammonia is an inorganic chemical compound composed of nitrogen and hydrogen with the chemical formula NH3.",
            "source": "Encyclopaedia Britannica — Ammonia",
            "source_url": "https://www.britannica.com/science/ammonia",
            "relevance_score": 1.0,
            "candidate_passages": [{
                "text": "Ammonia is an inorganic chemical compound composed of nitrogen and hydrogen with the chemical formula NH3.",
                "source_title": "Encyclopaedia Britannica — Ammonia",
                "source_url": "https://www.britannica.com/science/ammonia",
                "relevance_score": 1.0
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])
        self.assertIn("NH3", res["explanation"])

    # 4. Currency of Bangladesh is the taka -> SUPPORTED
    def test_case_4_bangladesh_taka_supported(self):
        claim = "The currency of Bangladesh is the taka."
        fixture = {
            "evidence": "The Bangladeshi taka is the official currency and legal tender of Bangladesh.",
            "source": "Encyclopaedia Britannica — Bangladesh",
            "source_url": "https://www.britannica.com/place/Bangladesh",
            "relevance_score": 1.0,
            "candidate_passages": [{
                "text": "The Bangladeshi taka is the official currency and legal tender of Bangladesh.",
                "source_title": "Encyclopaedia Britannica — Bangladesh",
                "source_url": "https://www.britannica.com/place/Bangladesh",
                "relevance_score": 1.0
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])
        self.assertIn("taka", res["explanation"].lower())

    # 5. Light-year measures distance rather than time -> SUPPORTED
    def test_case_5_light_year_distance_supported(self):
        claim = "A light-year measures distance rather than time."
        fixture = {
            "evidence": "Despite its inclusion of the word 'year', a light-year is a unit of astronomical distance, representing the distance light travels in one year, rather than a unit of time.",
            "source": "Encyclopaedia Britannica — Light-year",
            "source_url": "https://www.britannica.com/science/light-year",
            "relevance_score": 1.0,
            "candidate_passages": [{
                "text": "Despite its inclusion of the word 'year', a light-year is a unit of astronomical distance, representing the distance light travels in one year, rather than a unit of time.",
                "source_title": "Encyclopaedia Britannica — Light-year",
                "source_url": "https://www.britannica.com/science/light-year",
                "relevance_score": 1.0
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])
        self.assertIn("distance", res["explanation"])

    # 6. Stomach produces hydrochloric acid -> SUPPORTED
    def test_case_6_stomach_hydrochloric_acid_supported(self):
        claim = "The human stomach produces hydrochloric acid."
        fixture = {
            "evidence": "Gastric acid or stomach acid is the acidic component – hydrochloric acid – of gastric juice, produced by parietal cells in the gastric glands of the stomach lining.",
            "source": "Encyclopaedia Britannica — Stomach",
            "source_url": "https://www.britannica.com/science/stomach",
            "relevance_score": 1.0,
            "candidate_passages": [{
                "text": "Gastric acid or stomach acid is the acidic component – hydrochloric acid – of gastric juice, produced by parietal cells in the gastric glands of the stomach lining.",
                "source_title": "Encyclopaedia Britannica — Stomach",
                "source_url": "https://www.britannica.com/science/stomach",
                "relevance_score": 1.0
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])

    # 7. Capital of Portugal is Lisbon -> SUPPORTED
    def test_case_7_portugal_lisbon_supported(self):
        claim = "The capital of Portugal is Lisbon."
        fixture = {
            "evidence": "Lisbon is the capital and largest city of Portugal.",
            "source": "Encyclopaedia Britannica — Lisbon",
            "source_url": "https://www.britannica.com/place/Lisbon",
            "relevance_score": 1.0,
            "candidate_passages": [{
                "text": "Lisbon is the capital and largest city of Portugal.",
                "source_title": "Encyclopaedia Britannica — Lisbon",
                "source_url": "https://www.britannica.com/place/Lisbon",
                "relevance_score": 1.0
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])
        self.assertIn("Lisbon", res["explanation"])

    # 8. Square root of 225 is 15 -> SUPPORTED
    def test_case_8_square_root_225_supported(self):
        claim = "The square root of 225 is 15."
        fixture = {
            "evidence": "In mathematics, 225 is a square number, and the principal square root of 225 is 15 (15^2 = 225).",
            "source": "Encyclopaedia Britannica — Square root",
            "source_url": "https://www.britannica.com/science/square-root",
            "relevance_score": 1.0,
            "candidate_passages": [{
                "text": "In mathematics, 225 is a square number, and the principal square root of 225 is 15 (15^2 = 225).",
                "source_title": "Encyclopaedia Britannica — Square root",
                "source_url": "https://www.britannica.com/science/square-root",
                "relevance_score": 1.0
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])
        self.assertIn("225", res["explanation"])

    # 9. Every species of penguin lives in the Arctic -> CONTRADICTED
    def test_case_9_penguin_arctic_contradicted(self):
        claim = "Every species of penguin lives in the Arctic."
        fixture = {
            "evidence": "Penguins are flightless sea birds that live almost exclusively in the Southern Hemisphere (Antarctica and subantarctic regions). No species of penguin lives in the Arctic.",
            "source": "Encyclopaedia Britannica — Penguin",
            "source_url": "https://www.britannica.com/animal/penguin-bird",
            "relevance_score": 1.0,
            "candidate_passages": [{
                "text": "Penguins are flightless sea birds that live almost exclusively in the Southern Hemisphere (Antarctica and subantarctic regions). No species of penguin lives in the Arctic.",
                "source_title": "Encyclopaedia Britannica — Penguin",
                "source_url": "https://www.britannica.com/animal/penguin-bird",
                "relevance_score": 1.0
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], CONTRADICTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNotNone(res["correction"])
        self.assertIn("southern hemisphere", res["correction"].lower())

    # 10. Chemical symbol for calcium is Ca -> SUPPORTED
    def test_case_10_calcium_symbol_ca_supported(self):
        claim = "The chemical symbol for calcium is Ca."
        fixture = {
            "evidence": "Calcium is a chemical element with the chemical symbol Ca and atomic number 20.",
            "source": "Encyclopaedia Britannica — Calcium",
            "source_url": "https://www.britannica.com/science/calcium",
            "relevance_score": 1.0,
            "candidate_passages": [{
                "text": "Calcium is a chemical element with the chemical symbol Ca and atomic number 20.",
                "source_title": "Encyclopaedia Britannica — Calcium",
                "source_url": "https://www.britannica.com/science/calcium",
                "relevance_score": 1.0
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])
        self.assertIn("Ca", res["explanation"])

    # 11. First photograph of a black hole was released in 2019 -> SUPPORTED
    def test_case_11_black_hole_photo_2019_supported(self):
        claim = "The first photograph of a black hole was released in 2019."
        fixture = {
            "evidence": "On 10 April 2019, the first direct image of a black hole and its vicinity was published, following observations made by the Event Horizon Telescope (EHT) of the supermassive black hole in Messier 87.",
            "source": "NASA — Event Horizon Telescope Releases First Black Hole Image",
            "source_url": "https://science.nasa.gov/astrophysics/focus-areas/black-holes/",
            "relevance_score": 1.0,
            "candidate_passages": [{
                "text": "On 10 April 2019, the first direct image of a black hole and its vicinity was published, following observations made by the Event Horizon Telescope (EHT) of the supermassive black hole in Messier 87.",
                "source_title": "NASA — Event Horizon Telescope Releases First Black Hole Image",
                "source_url": "https://science.nasa.gov/astrophysics/focus-areas/black-holes/",
                "relevance_score": 1.0
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])
        self.assertIn("2019", res["explanation"])

    # 12. My best friend moved to Canada last month -> INSUFFICIENT_EVIDENCE
    def test_case_12_personal_friend_canada_insufficient(self):
        claim = "My best friend moved to Canada last month."
        self.assertTrue(is_personal_claim(claim))
        fixture = {
            "evidence": "",
            "source": "",
            "source_url": "",
            "retrieval": "personal_claim",
            "relevance_score": 0.0,
            "candidate_passages": []
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], INSUFFICIENT_EVIDENCE)
        self.assertEqual(res["confidence"], 0.0)
        self.assertIsNone(res["correction"])
        self.assertIn("personal", res["explanation"].lower())

    # 13. A standard piano has 88 keys -> SUPPORTED
    def test_case_13_piano_88_keys_supported(self):
        claim = "A standard piano has 88 keys."
        fixture = {
            "evidence": "Modern standard pianos have a row of 88 black and white keys (52 white keys and 36 black keys).",
            "source": "Encyclopaedia Britannica — Piano",
            "source_url": "https://www.britannica.com/art/piano-musical-instrument",
            "relevance_score": 1.0,
            "candidate_passages": [{
                "text": "Modern standard pianos have a row of 88 black and white keys (52 white keys and 36 black keys).",
                "source_title": "Encyclopaedia Britannica — Piano",
                "source_url": "https://www.britannica.com/art/piano-musical-instrument",
                "relevance_score": 1.0
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], SUPPORTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertIsNone(res["correction"])
        self.assertIn("88 keys", res["explanation"])

    # 14. Capital of South Korea is Busan -> CONTRADICTED
    def test_case_14_south_korea_busan_contradicted(self):
        claim = "The capital of South Korea is Busan."
        fixture = {
            "evidence": "Seoul, officially Seoul Special Metropolitan City, is the capital and largest city of South Korea.",
            "source": "Encyclopaedia Britannica — Seoul",
            "source_url": "https://www.britannica.com/place/Seoul",
            "relevance_score": 1.0,
            "candidate_passages": [{
                "text": "Seoul, officially Seoul Special Metropolitan City, is the capital and largest city of South Korea.",
                "source_title": "Encyclopaedia Britannica — Seoul",
                "source_url": "https://www.britannica.com/place/Seoul",
                "relevance_score": 1.0
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], CONTRADICTED)
        self.assertGreaterEqual(res["confidence"], 0.70)
        self.assertEqual(res["correction"], "The capital of South Korea is Seoul.")
        self.assertIn("Seoul", res["explanation"])

    # Additional negative and edge-case unit tests
    def test_irrelevant_source_does_not_contradict_personal_claim(self):
        """Passage about .my Malaysia domain must not contradict personal friend claim."""
        claim = "My best friend moved to Canada last month."
        fixture = {
            "evidence": ".my is the Internet country code top-level domain (ccTLD) for Malaysia.",
            "source": "Wikipedia — .my",
            "source_url": "https://en.wikipedia.org/wiki/.my",
            "relevance_score": 0.20,
            "candidate_passages": [{
                "text": ".my is the Internet country code top-level domain (ccTLD) for Malaysia.",
                "source_title": "Wikipedia — .my",
                "source_url": "https://en.wikipedia.org/wiki/.my",
                "relevance_score": 0.20
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], INSUFFICIENT_EVIDENCE)
        self.assertEqual(res["confidence"], 0.0)
        self.assertIsNone(res["correction"])

    def test_dod_pentagon_building_does_not_contradict_geometry_pentagon(self):
        """A passage about US DoD Pentagon building must not contradict a geometry pentagon claim."""
        claim = "A regular pentagon has five sides."
        fixture = {
            "evidence": "The Pentagon is the headquarters of the United States Department of Defense.",
            "source": "Wikipedia — The Pentagon",
            "source_url": "https://en.wikipedia.org/wiki/The_Pentagon",
            "relevance_score": 0.20,
            "candidate_passages": [{
                "text": "The Pentagon is the headquarters of the United States Department of Defense.",
                "source_title": "Wikipedia — The Pentagon",
                "source_url": "https://en.wikipedia.org/wiki/The_Pentagon",
                "relevance_score": 0.20
            }]
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], INSUFFICIENT_EVIDENCE)
        self.assertIsNone(res["correction"])

    def test_empty_or_missing_evidence_yields_insufficient(self):
        claim = "An obscure undiscovered element exists in a distant galaxy."
        fixture = {
            "evidence": "",
            "source": "",
            "source_url": "",
            "relevance_score": 0.0,
            "candidate_passages": []
        }
        res = verify_claim(claim, fixture)
        self.assertEqual(res["verdict"], INSUFFICIENT_EVIDENCE)
        self.assertEqual(res["confidence"], 0.0)
        self.assertIsNone(res["correction"])

    def test_unsupported_correction_rejected(self):
        """Corrections must be rejected if not addressing the proposition or corporate/census text."""
        claim = "The capital of Switzerland is Zurich."
        fixture = {
            "evidence": "Bank (SA), Geneva Van Lanschot Switzerland, Zurich None Sberbank, Zurich VTB Capital, Zug",
            "source": "Wikipedia — List of banks in Switzerland",
            "source_url": "https://en.wikipedia.org/wiki/List_of_banks_in_Switzerland"
        }
        valid = validate_correction(claim, "Bank (SA), Geneva Van Lanschot Switzerland, Zurich None Sberbank, Zurich VTB Capital, Zug", fixture)
        self.assertFalse(valid)

    def test_census_correction_rejected(self):
        """Census population text must never be accepted as a correction for human organ discovery."""
        claim = "Scientists discovered a new human organ in Hyderabad yesterday."
        fixture = {
            "evidence": "According to the 2011 census of India, Hyderabad is the fourth-most populous city in India.",
            "source": "Wikipedia — Hyderabad"
        }
        valid = validate_correction(claim, "According to the 2011 census of India, Hyderabad is the fourth-most populous city in India.", fixture)
        self.assertFalse(valid)

    def test_confidence_calibration_bounds(self):
        """Confidence score must never exceed 0.98 or be forced to 1.0."""
        conf_max = calculate_confidence(SUPPORTED, relevance_score=1.0, strength_score=1.0, num_supporting_passages=5)
        self.assertLessEqual(conf_max, 0.98)
        self.assertGreaterEqual(conf_max, 0.70)

        conf_ins = calculate_confidence(INSUFFICIENT_EVIDENCE, relevance_score=0.0, strength_score=0.0)
        self.assertEqual(conf_ins, 0.0)

    def test_compound_claim_handling(self):
        """Test compound claims evaluation and aggregation policy."""
        c_true = "The capital of Portugal is Lisbon and a standard piano has 88 keys."
        res_true = verify_claim(c_true)
        self.assertEqual(res_true["verdict"], SUPPORTED)
        self.assertIsNone(res_true["correction"])

        c_mixed = "The capital of Portugal is Lisbon and the capital of South Korea is Busan."
        res_mixed = verify_claim(c_mixed)
        self.assertEqual(res_mixed["verdict"], CONTRADICTED)
        self.assertIsNotNone(res_mixed["correction"])
        self.assertIn("Seoul", res_mixed["correction"])

    def test_section_4_structured_passage_schema(self):
        """Test evaluate_passage_comparison returns the exact structured schema specified in Section 4."""
        claim = "A light-year measures distance rather than time."
        cand = {
            "text": "Despite its inclusion of the word 'year', a light-year is a unit of distance, not time.",
            "source_title": "Wikipedia — Light-year",
            "source_url": "https://en.wikipedia.org/wiki/Light-year",
            "relevance_score": 1.0
        }
        out = evaluate_passage_comparison(claim, cand)
        self.assertIn("claim", out)
        self.assertIn("evidence_relationship", out)
        self.assertIn("evidence_relevant", out)
        self.assertIn("reasoning", out)
        self.assertIn("supporting_text", out)
        self.assertIn("source_url", out)
        self.assertEqual(out["evidence_relationship"], ENTAILMENT)
        self.assertTrue(out["evidence_relevant"])

if __name__ == "__main__":
    unittest.main()
