from __future__ import annotations

import unittest

from valuation_engine import (
    CANONICAL_FUZZY_SCORE,
    LIKELY_FUZZY_SCORE,
    MAX_SOLD_COMP_RESULTS,
    EbayMarketDataFetcher,
    ListingNormalizer,
    MarketSummary,
    ValuationEngine,
)


class ListingNormalizerTests(unittest.TestCase):
    """Verify normalisation integrates fuzzy matching metadata."""

    def setUp(self) -> None:
        self.normalizer = ListingNormalizer()

    def test_normalize_cleans_and_combines_fields(self) -> None:
        """Title/brand data should be cleaned and merged into a canonical query."""

        result = self.normalizer.normalize(
            title="  Lîmoges plate-- deluxe  ",
            brand="  Château-d'Or  ",
            base_search_text="Limoges porcelain",
            fuzzy_result={"target": "Limoges porcelain", "score": CANONICAL_FUZZY_SCORE + 2},
        )

        self.assertEqual(result.title, "Limoges plate deluxe")
        self.assertEqual(result.brand, "Chateau d Or")
        self.assertEqual(result.fuzzy_term, "Limoges porcelain")
        self.assertEqual(result.match_quality, "canonical")
        self.assertEqual(result.canonical_query, "Chateau d Or Limoges porcelain")

    def test_likely_quality_range(self) -> None:
        """Scores in the likely range should be surfaced as such."""

        result = self.normalizer.normalize(
            title="Vintage Vase",
            brand=None,
            base_search_text="Vintage Vase",
            fuzzy_result={"target": "Vintage Vase", "score": LIKELY_FUZZY_SCORE + 1},
        )

        self.assertEqual(result.fuzzy_term, "Vintage Vase")
        self.assertEqual(result.match_quality, "likely")
        self.assertEqual(result.canonical_query, "Vintage Vase")

    def test_title_used_when_no_fuzzy_information(self) -> None:
        """Without fuzzy data we still build a canonical query from the title."""

        result = self.normalizer.normalize(
            title="Rare Art Glass Bowl",
            brand=None,
            base_search_text=None,
            fuzzy_result=None,
        )

        self.assertEqual(result.canonical_query, "Rare Art Glass Bowl")
        self.assertEqual(result.match_quality, "unknown")


class ConfidenceScoringTests(unittest.TestCase):
    """Exercise the valuation confidence categorisation rules."""

    def setUp(self) -> None:
        self.engine = ValuationEngine(fetcher=None)

    def test_high_confidence_requires_canonical_and_low_variance(self) -> None:
        """High confidence requires both strong fuzzy match and stable comps."""

        summary = MarketSummary(
            minimum=95,
            maximum=105,
            median=100,
            currency="EUR",
            sample_size=8,
            source="eBay sold",
            prices=[98, 99, 100, 100, 101, 101, 99, 100],
        )

        label, icon = self.engine._score_confidence(CANONICAL_FUZZY_SCORE, summary)
        self.assertEqual(label, "High")
        self.assertEqual(icon, "🟢")

    def test_medium_confidence_for_likely_scores(self) -> None:
        """Likely fuzzy matches downgrade confidence to medium."""

        summary = MarketSummary(
            minimum=80,
            maximum=140,
            median=105,
            currency="EUR",
            sample_size=6,
            source="eBay sold",
            prices=[80, 95, 100, 110, 120, 140],
        )

        label, icon = self.engine._score_confidence(LIKELY_FUZZY_SCORE + 0.5, summary)
        self.assertEqual(label, "Medium")
        self.assertEqual(icon, "🟡")

    def test_medium_confidence_when_many_comps_but_low_score(self) -> None:
        """A large sample offsets a weak fuzzy score to medium confidence."""

        summary = MarketSummary(
            minimum=150,
            maximum=260,
            median=205,
            currency="USD",
            sample_size=9,
            source="eBay sold",
            prices=[150, 180, 190, 200, 205, 210, 215, 230, 260],
        )

        label, icon = self.engine._score_confidence(LIKELY_FUZZY_SCORE - 20, summary)
        self.assertEqual(label, "Medium")
        self.assertEqual(icon, "🟡")

    def test_low_confidence_when_not_enough_comps(self) -> None:
        """Fewer than four comparables always yields low confidence."""

        summary = MarketSummary(
            minimum=40,
            maximum=60,
            median=50,
            currency="EUR",
            sample_size=3,
            source="eBay sold",
            prices=[40, 50, 60],
        )

        label, icon = self.engine._score_confidence(CANONICAL_FUZZY_SCORE + 5, summary)
        self.assertEqual(label, "Low")
        self.assertEqual(icon, "🟠")


class FetchLimitTests(unittest.TestCase):
    """Ensure eBay fetching stays within the narrow sold comp window."""

    def test_valuation_engine_caps_sold_limit(self) -> None:
        """Evaluation should request no more than five sold comparables."""

        class DummyFetcher:
            def __init__(self) -> None:
                self.requested_limits = []

            def pick_global_id(self, currency: Optional[str]) -> str:
                return "EBAY-DE"

            def fetch_sold_comps(
                self, query: str, limit: int, global_id: Optional[str]
            ) -> list:
                self.requested_limits.append(limit)
                return []

            def fetch_active_listings(
                self, query: str, limit: int, global_id: Optional[str]
            ) -> list:
                return []

        fetcher = DummyFetcher()
        engine = ValuationEngine(fetcher=fetcher, active_limit=0)

        engine.evaluate(
            title="Vintage Lamp",
            brand=None,
            price=25.0,
            currency="EUR",
            base_search_text=None,
            fuzzy_result=None,
        )

        self.assertEqual(fetcher.requested_limits, [MAX_SOLD_COMP_RESULTS])

    def test_fetcher_clamps_sold_entries_to_five(self) -> None:
        """Even when asked for more, the fetcher should cap the API page size."""

        class StubResponse:
            def __init__(self) -> None:
                self._json = {
                    "findCompletedItemsResponse": [
                        {"searchResult": [{"item": []}]}
                    ]
                }

            def raise_for_status(self) -> None:  # pragma: no cover - simple stub
                return None

            def json(self) -> dict:
                return self._json

        class StubSession:
            def __init__(self) -> None:
                self.params = None

            def get(self, url: str, params: dict, timeout: int) -> StubResponse:
                self.params = params
                return StubResponse()

        session = StubSession()
        fetcher = EbayMarketDataFetcher(app_id="dummy", session=session)
        fetcher.fetch_sold_comps("foo", limit=20)

        self.assertIsNotNone(session.params)
        self.assertEqual(
            session.params["paginationInput.entriesPerPage"],
            str(MAX_SOLD_COMP_RESULTS),
        )


if __name__ == "__main__":
    unittest.main()
