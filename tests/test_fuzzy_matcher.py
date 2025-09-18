"""Tests for fuzzy matcher search text expansion limits."""

from __future__ import annotations

import unittest
from unittest import mock

import core
from fuzzy_matcher import MAX_VARIANTS, _expand_search_text_variants


class ExpandSearchTextVariantsTests(unittest.TestCase):
    """Verify typo generation respects the configured limit."""

    def test_returns_at_most_max_variants(self) -> None:
        """Ensure `_expand_search_text_variants` caps results at `MAX_VARIANTS`."""

        variants = _expand_search_text_variants("luminarc")

        self.assertLessEqual(len(variants), MAX_VARIANTS)
        self.assertIn("luminarc", variants)


class ProcessQueryVariantsTests(unittest.TestCase):
    """Ensure query processing does not insert more than the allowed variants."""

    def test_process_query_limits_variant_insertions(self) -> None:
        """`process_query` should not insert more than `MAX_VARIANTS` entries."""

        search_text = "luminarc"
        query = f"https://www.vinted.fr/vetements?search_text={search_text}"
        expected_variants = _expand_search_text_variants(search_text)

        with mock.patch.object(
            core.db,
            "is_query_in_db",
            return_value=False,
        ) as mock_is_query_in_db, mock.patch.object(
            core.db,
            "add_query_to_db",
        ) as mock_add_query:
            message, is_new = core.process_query(query)

        self.assertTrue(is_new)
        self.assertEqual(mock_is_query_in_db.call_count, len(expected_variants))
        self.assertEqual(mock_add_query.call_count, len(expected_variants))
        self.assertLessEqual(mock_add_query.call_count, MAX_VARIANTS)
        self.assertIn(f"{len(expected_variants)} variants considered", message)


if __name__ == "__main__":
    unittest.main()
