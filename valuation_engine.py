from __future__ import annotations

import os
from dataclasses import dataclass, field
from statistics import median, pstdev
from typing import Dict, List, Optional, Sequence

import requests
from unidecode import unidecode

from logger import get_logger

logger = get_logger(__name__)

CURRENCY_SYMBOLS: Dict[str, str] = {
    "EUR": "€",
    "USD": "$",
    "GBP": "£",
    "CAD": "C$",
    "AUD": "A$",
    "CHF": "CHF",
    "PLN": "zł",
    "JPY": "¥",
}


@dataclass
class ListingComp:
    """A lightweight representation of a comparable listing."""

    title: str
    price: Optional[float]
    currency: Optional[str]
    url: Optional[str]
    image: Optional[str]
    source: str


@dataclass
class MarketSummary:
    """Aggregate statistics for a set of comparable listings."""

    minimum: Optional[float] = None
    maximum: Optional[float] = None
    median: Optional[float] = None
    currency: Optional[str] = None
    sample_size: int = 0
    source: str = ""
    prices: List[float] = field(default_factory=list)


@dataclass
class NormalizationResult:
    """Result of normalising a Vinted listing for valuation."""

    title: str
    brand: Optional[str]
    canonical_query: str
    fuzzy_term: Optional[str]
    fuzzy_score: Optional[float]
    match_quality: str


@dataclass
class ValuationResult:
    """Outcome of the valuation engine for a Vinted item."""

    normalization: NormalizationResult
    sold_summary: MarketSummary
    active_summary: Optional[MarketSummary]
    profit: Optional[float]
    profit_currency: Optional[str]
    profit_multiple: Optional[float]
    confidence_label: str
    confidence_icon: str
    item_price: float
    item_currency: str
    sold_comps: List[ListingComp] = field(default_factory=list)
    active_listings: List[ListingComp] = field(default_factory=list)


def format_money(amount: Optional[float], currency: Optional[str]) -> str:
    """Return a human readable representation of ``amount`` in ``currency``."""

    if amount is None:
        return "n/a"

    symbol = CURRENCY_SYMBOLS.get(currency or "", "")
    value = f"{amount:.2f}".rstrip("0").rstrip(".")
    if not value:
        value = "0"

    if symbol:
        return f"{symbol}{value}"
    if currency:
        return f"{value} {currency}"
    return value


def format_price_range(summary: MarketSummary) -> str:
    """Format the minimum/maximum price range for a market summary."""

    if summary.sample_size == 0 or summary.minimum is None or summary.maximum is None:
        return "n/a"

    lower = format_money(summary.minimum, summary.currency)
    upper = format_money(summary.maximum, summary.currency)
    if lower == upper:
        return lower
    return f"{lower}–{upper}"


def format_market_summary(summary: MarketSummary) -> str:
    """Return a descriptive string for a market summary."""

    if summary.sample_size == 0:
        return "not enough data"

    range_text = format_price_range(summary)
    median_text = format_money(summary.median, summary.currency)
    source = summary.source or "market"
    return f"{range_text} (median {median_text}, {source}, n={summary.sample_size})"


def format_active_summary(summary: Optional[MarketSummary]) -> str:
    """Return the formatted active listings line."""

    if summary is None or summary.sample_size == 0:
        return ""
    return f"\n🛒 Active listings: {format_market_summary(summary)}"


def format_profit_estimate(result: ValuationResult) -> str:
    """Return a profit estimation string based on the valuation result."""

    if result.profit is None:
        if result.sold_summary.sample_size == 0:
            return "n/a (no comps)"
        if (
            result.sold_summary.currency
            and result.sold_summary.currency != result.item_currency
        ):
            return "n/a (currency mismatch)"
        return "n/a"

    currency = result.profit_currency or result.item_currency
    formatted = format_money(abs(result.profit), currency)
    if result.profit > 0:
        formatted = f"+{formatted}"
    elif result.profit < 0:
        formatted = f"-{formatted}"

    if result.profit_multiple and result.profit_multiple > 0:
        formatted += f" ({result.profit_multiple:.1f}x)"

    return formatted


def format_confidence_line(result: ValuationResult) -> str:
    """Return a confidence line for Telegram messages."""

    return f"\n{result.confidence_icon} Confidence: {result.confidence_label}"


def format_fuzzy_line(normalization: NormalizationResult) -> str:
    """Return a user-friendly fuzzy match description."""

    term = normalization.fuzzy_term
    score = normalization.fuzzy_score
    quality = normalization.match_quality

    if not term and score is None:
        return "No fuzzy match"

    quoted_term = f"“{term}”" if term else "match"
    if score is None:
        return f"{quoted_term}"

    score_str = f"score {int(round(score))}"
    if quality == "canonical":
        return f"{quoted_term} ({score_str}, canonical)"
    if quality == "likely":
        return f"{quoted_term} ({score_str}, likely)"
    if quality == "approximate":
        return f"{quoted_term} ({score_str}, loose)"
    return f"{quoted_term} ({score_str})"


def build_reference_line(result: ValuationResult) -> str:
    """Return a reference image or listing link if available."""

    for comp in result.sold_comps:
        if comp.image:
            return f"\n🖼️ Reference: {comp.image}"
    for comp in result.sold_comps:
        if comp.url:
            return f"\n🖼️ Reference: {comp.url}"
    return ""


class ListingNormalizer:
    """Normalise Vinted data before running valuations."""

    @staticmethod
    def _clean(value: Optional[str]) -> str:
        if not value:
            return ""
        normalized = unidecode(value)
        normalized = normalized.replace("-", " ")
        normalized = normalized.replace("_", " ")
        return " ".join(normalized.split()).strip()

    def normalize(
        self,
        title: Optional[str],
        brand: Optional[str],
        base_search_text: Optional[str],
        fuzzy_result: Optional[Dict[str, object]],
    ) -> NormalizationResult:
        """Return the normalised representation for valuation queries."""

        cleaned_title = self._clean(title)
        cleaned_brand = self._clean(brand)

        fuzzy_term: Optional[str] = None
        fuzzy_score: Optional[float] = None
        match_quality = "unknown"

        if fuzzy_result:
            fuzzy_term = (fuzzy_result.get("target") or "").strip() or None
            try:
                fuzzy_score = float(fuzzy_result.get("score"))
            except (TypeError, ValueError):
                fuzzy_score = None

        if not fuzzy_term and base_search_text:
            fuzzy_term = base_search_text.strip() or None

        if fuzzy_score is not None:
            if fuzzy_score >= 90:
                match_quality = "canonical"
            elif 82 <= fuzzy_score <= 89:
                match_quality = "likely"
            else:
                match_quality = "approximate"
        elif fuzzy_term:
            match_quality = "approximate"

        parts: List[str] = []
        if brand and brand.strip():
            parts.append(brand.strip())
        if fuzzy_term:
            parts.append(fuzzy_term)
        elif title:
            parts.append(title.strip())

        # Remove duplicates while preserving order
        seen: Dict[str, None] = {}
        canonical_parts: List[str] = []
        for part in parts:
            key = part.lower()
            if key in seen:
                continue
            seen[key] = None
            canonical_parts.append(part)

        canonical_query = " ".join(canonical_parts).strip()

        return NormalizationResult(
            title=cleaned_title or (title or ""),
            brand=cleaned_brand or (brand or None),
            canonical_query=canonical_query,
            fuzzy_term=fuzzy_term,
            fuzzy_score=fuzzy_score,
            match_quality=match_quality,
        )


class EbayMarketDataFetcher:
    """Fetch sold comps and active listings from eBay."""

    ENDPOINT = "https://svcs.ebay.com/services/search/FindingService/v1"
    CURRENCY_TO_GLOBAL_ID: Dict[str, str] = {
        "EUR": "EBAY-DE",
        "USD": "EBAY-US",
        "GBP": "EBAY-GB",
        "CAD": "EBAY-ENCA",
        "AUD": "EBAY-AU",
        "CHF": "EBAY-CH",
    }

    def __init__(
        self,
        app_id: Optional[str] = None,
        default_global_id: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: int = 12,
    ) -> None:
        self.app_id = app_id or os.getenv("EBAY_APP_ID") or os.getenv("EBAY_APPID")
        self.default_global_id = (
            default_global_id
            or os.getenv("EBAY_GLOBAL_ID")
            or "EBAY-DE"
        )
        self.session = session or requests.Session()
        self.timeout = timeout

    def pick_global_id(self, currency: Optional[str]) -> str:
        """Return the best eBay global id for ``currency``."""

        if currency and currency in self.CURRENCY_TO_GLOBAL_ID:
            return self.CURRENCY_TO_GLOBAL_ID[currency]
        return self.default_global_id

    def fetch_sold_comps(
        self,
        query: str,
        limit: int = 30,
        global_id: Optional[str] = None,
    ) -> List[ListingComp]:
        """Return a list of sold comparable listings for ``query``."""

        if not self.app_id or not query:
            logger.debug("Skipping sold comps fetch due to missing credentials or query")
            return []

        params = {
            "OPERATION-NAME": "findCompletedItems",
            "SERVICE-VERSION": "1.13.0",
            "SECURITY-APPNAME": self.app_id,
            "RESPONSE-DATA-FORMAT": "JSON",
            "REST-PAYLOAD": "true",
            "keywords": query,
            "itemFilter(0).name": "SoldItemsOnly",
            "itemFilter(0).value": "true",
            "paginationInput.entriesPerPage": str(max(1, min(limit, 100))),
            "sortOrder": "EndTimeSoonest",
        }
        if global_id:
            params["GLOBAL-ID"] = global_id

        data = self._perform_request(params)
        items = self._extract_items(data, "findCompletedItemsResponse")

        comps: List[ListingComp] = []
        for item in items:
            selling_status = item.get("sellingStatus", [{}])[0]
            price_info = selling_status.get("convertedCurrentPrice") or selling_status.get("currentPrice")
            price, currency = self._extract_price(price_info)
            state = selling_status.get("sellingState", [""])[0]
            if state and state != "EndedWithSales":
                continue
            comps.append(
                ListingComp(
                    title=item.get("title", [""])[0],
                    price=price,
                    currency=currency,
                    url=item.get("viewItemURL", [None])[0],
                    image=item.get("galleryURL", [None])[0],
                    source="eBay sold",
                )
            )
        return comps

    def fetch_active_listings(
        self,
        query: str,
        limit: int = 10,
        global_id: Optional[str] = None,
    ) -> List[ListingComp]:
        """Return currently active comparable listings for ``query``."""

        if not self.app_id or not query:
            return []

        params = {
            "OPERATION-NAME": "findItemsAdvanced",
            "SERVICE-VERSION": "1.13.0",
            "SECURITY-APPNAME": self.app_id,
            "RESPONSE-DATA-FORMAT": "JSON",
            "REST-PAYLOAD": "true",
            "keywords": query,
            "paginationInput.entriesPerPage": str(max(1, min(limit, 100))),
            "sortOrder": "PricePlusShippingLowest",
        }
        if global_id:
            params["GLOBAL-ID"] = global_id

        data = self._perform_request(params)
        items = self._extract_items(data, "findItemsAdvancedResponse")

        listings: List[ListingComp] = []
        for item in items:
            selling_status = item.get("sellingStatus", [{}])[0]
            price_info = selling_status.get("convertedCurrentPrice") or selling_status.get("currentPrice")
            price, currency = self._extract_price(price_info)
            listings.append(
                ListingComp(
                    title=item.get("title", [""])[0],
                    price=price,
                    currency=currency,
                    url=item.get("viewItemURL", [None])[0],
                    image=item.get("galleryURL", [None])[0],
                    source="eBay active",
                )
            )
        return listings

    def _perform_request(self, params: Dict[str, str]) -> Dict[str, object]:
        try:
            response = self.session.get(
                self.ENDPOINT,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.warning("Error while fetching data from eBay: %s", exc)
        except ValueError as exc:
            logger.warning("Failed to decode eBay response: %s", exc)
        return {}

    @staticmethod
    def _extract_items(data: Dict[str, object], root_key: str) -> Sequence[Dict[str, object]]:
        try:
            response = data[root_key][0]
            search_result = response.get("searchResult", [{}])[0]
            return search_result.get("item", [])
        except (KeyError, IndexError, TypeError):
            return []

    @staticmethod
    def _extract_price(price_info: Optional[Sequence[Dict[str, object]]]) -> tuple[Optional[float], Optional[str]]:
        if not price_info:
            return None, None
        try:
            info = price_info[0]
        except (IndexError, TypeError):
            info = price_info

        try:
            value = info.get("__value__")
        except AttributeError:
            value = None

        currency = None
        try:
            currency = info.get("@currencyId")
        except AttributeError:
            currency = None

        if value is None:
            return None, currency

        try:
            return float(value), currency
        except (TypeError, ValueError):
            return None, currency


class ValuationEngine:
    """Compute valuation metrics for Vinted items."""

    def __init__(
        self,
        fetcher: Optional[EbayMarketDataFetcher],
        normalizer: Optional[ListingNormalizer] = None,
        sold_limit: int = 30,
        active_limit: int = 10,
        low_variance_threshold: float = 0.35,
    ) -> None:
        self.fetcher = fetcher
        self.normalizer = normalizer or ListingNormalizer()
        self.sold_limit = sold_limit
        self.active_limit = active_limit
        self.low_variance_threshold = low_variance_threshold

    def evaluate(
        self,
        title: Optional[str],
        brand: Optional[str],
        price: float,
        currency: str,
        base_search_text: Optional[str],
        fuzzy_result: Optional[Dict[str, object]],
    ) -> ValuationResult:
        normalization = self.normalizer.normalize(title, brand, base_search_text, fuzzy_result)

        sold_comps: List[ListingComp] = []
        active_listings: List[ListingComp] = []
        sold_summary = MarketSummary(source="eBay sold")
        active_summary: Optional[MarketSummary] = None

        if self.fetcher and normalization.canonical_query:
            global_id = self.fetcher.pick_global_id(currency)
            sold_comps = self.fetcher.fetch_sold_comps(
                normalization.canonical_query,
                limit=self.sold_limit,
                global_id=global_id,
            )
            sold_summary = self._summarize(sold_comps, "eBay sold")

            if self.active_limit > 0:
                active_listings = self.fetcher.fetch_active_listings(
                    normalization.canonical_query,
                    limit=self.active_limit,
                    global_id=global_id,
                )
                active_summary = self._summarize(active_listings, "eBay active")
        else:
            logger.debug(
                "Valuation skipped fetch; query=%s, fetcher=%s",
                normalization.canonical_query,
                bool(self.fetcher),
            )

        profit, profit_currency, profit_multiple = self._estimate_profit(
            sold_summary,
            price,
            currency,
        )
        confidence_label, confidence_icon = self._score_confidence(
            normalization.fuzzy_score,
            sold_summary,
        )

        return ValuationResult(
            normalization=normalization,
            sold_summary=sold_summary,
            active_summary=active_summary,
            profit=profit,
            profit_currency=profit_currency,
            profit_multiple=profit_multiple,
            confidence_label=confidence_label,
            confidence_icon=confidence_icon,
            item_price=price,
            item_currency=currency,
            sold_comps=sold_comps,
            active_listings=active_listings,
        )

    def _summarize(
        self,
        listings: Sequence[ListingComp],
        default_source: str,
    ) -> MarketSummary:
        if not listings:
            return MarketSummary(source=default_source)

        currency_groups: Dict[str, List[float]] = {}
        for listing in listings:
            if listing.price is None or listing.currency is None:
                continue
            currency_groups.setdefault(listing.currency, []).append(listing.price)

        if not currency_groups:
            return MarketSummary(source=default_source)

        currency, prices = max(currency_groups.items(), key=lambda item: len(item[1]))
        sorted_prices = sorted(prices)

        summary = MarketSummary(
            minimum=min(sorted_prices),
            maximum=max(sorted_prices),
            median=median(sorted_prices),
            currency=currency,
            sample_size=len(sorted_prices),
            source=default_source,
            prices=sorted_prices,
        )
        return summary

    def _estimate_profit(
        self,
        summary: MarketSummary,
        item_price: float,
        item_currency: str,
    ) -> tuple[Optional[float], Optional[str], Optional[float]]:
        if summary.sample_size == 0 or summary.median is None:
            return None, None, None
        if summary.currency and summary.currency != item_currency:
            return None, summary.currency, None

        profit = summary.median - item_price
        multiple = None
        if item_price > 0:
            multiple = summary.median / item_price
        return profit, summary.currency or item_currency, multiple

    def _score_confidence(
        self,
        fuzzy_score: Optional[float],
        summary: MarketSummary,
    ) -> tuple[str, str]:
        comps = summary.sample_size
        if comps < 4:
            return "Low", "🟠"

        score = fuzzy_score or 0
        if (
            score >= 90
            and comps >= 8
            and self._has_low_variance(summary.prices)
        ):
            return "High", "🟢"

        if (82 <= score < 90) or (4 <= comps <= 7):
            return "Medium", "🟡"

        if comps >= 8:
            return "Medium", "🟡"

        return "Low", "🟠"

    def _has_low_variance(self, prices: Sequence[float]) -> bool:
        if len(prices) < 2:
            return True
        med = median(prices)
        if med <= 0:
            return False
        deviation = pstdev(prices)
        if deviation == 0:
            return True
        coefficient = deviation / med
        return coefficient <= self.low_variance_threshold
