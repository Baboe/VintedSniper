import db, configuration_values, requests
from pyVintedVN import Vinted, requester
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from typing import Optional
from logger import get_logger

from fuzzy_matcher import (
    DEFAULT_FUZZY_THRESHOLD,
    _expand_search_text_variants,
    decode_query_name,
    encode_query_name,
    find_best_fuzzy_match,
)

from valuation_engine import (
    EbayMarketDataFetcher,
    ValuationEngine,
    build_reference_line,
    format_active_summary,
    format_confidence_line,
    format_fuzzy_line,
    format_market_summary,
    format_money,
    format_profit_estimate,
)

# Get logger for this module
logger = get_logger(__name__)


_valuation_engine: Optional[ValuationEngine] = None


def _get_valuation_engine() -> ValuationEngine:
    """Return a singleton valuation engine instance."""

    global _valuation_engine
    if _valuation_engine is None:
        fetcher = EbayMarketDataFetcher()
        _valuation_engine = ValuationEngine(fetcher)
    return _valuation_engine


def process_query(query, name=None):
    """
    Process a Vinted query URL by:
    1. Parsing the URL and extracting query parameters
    2. Ensuring the order flag is set to "newest_first"
    3. Removing time and search_id parameters
    4. Rebuilding the query string and URL
    5. Checking if the query already exists in the database
    6. Adding the query to the database if it doesn't exist

    Args:
        query (str): The Vinted query URL
        name (str, optional): A name for the query. If provided, it will be used as the query name.

    Returns:
        tuple: (message, is_new_query)
            - message (str): Status message
            - is_new_query (bool): True if query was added, False if it already existed
    """
    # Parse the URL and extract the query parameters
    parsed_url = urlparse(query)
    query_params = parse_qs(parsed_url.query)

    # Ensure the order flag is set to newest_first
    query_params['order'] = ['newest_first']
    # Remove time and search_id if provided
    query_params.pop('time', None)
    query_params.pop('search_id', None)
    query_params.pop('disabled_personalization', None)
    query_params.pop('page', None)

    searched_text = query_params.get('search_text')
    base_search_text = searched_text[0] if searched_text else None

    display_name = name or base_search_text
    stored_name = encode_query_name(display_name, base_search_text)

    base_params = {key: list(value) for key, value in query_params.items()}

    def build_processed_query(params):
        new_query = urlencode(params, doseq=True)
        return urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                new_query,
                parsed_url.fragment,
            )
        )

    variant_queries = []
    if base_search_text and configuration_values.ENABLE_VARIANTS:
        expanded_terms = _expand_search_text_variants(base_search_text)
        logger.info(
            "Expanded search_text '%s' into %d variant(s)",
            base_search_text,
            len(expanded_terms),
        )
        for variant in expanded_terms:
            params = {key: list(value) for key, value in base_params.items()}
            params['search_text'] = [variant]
            variant_queries.append(build_processed_query(params))
    else:
        if base_search_text and not configuration_values.ENABLE_VARIANTS:
            logger.info(
                "Search text variants disabled by configuration; using base search_text '%s' only.",
                base_search_text,
            )
        variant_queries.append(build_processed_query(base_params))

    added_count = 0
    for processed_query in variant_queries:
        if db.is_query_in_db(processed_query):
            continue
        db.add_query_to_db(processed_query, stored_name)
        added_count += 1

    if added_count == 0:
        return "Query already exists.", False

    if len(variant_queries) == 1:
        return "Query added.", True

    return f"Added {added_count} queries ({len(variant_queries)} variants considered).", True

def get_formatted_query_list():
    """
    Get a formatted list of all queries in the database.

    Returns:
        str: A formatted string with all queries, numbered
    """
    all_queries = db.get_queries()
    entries = []
    for query in all_queries:
        parsed_url = urlparse(query[1])
        query_params = parse_qs(parsed_url.query)
        display_name, _ = decode_query_name(query[3])
        fallback = query_params.get('search_text', [None])[0]
        entry = display_name or fallback or query[1]
        entries.append(entry)

    query_list = "\n".join(f"{i + 1}. {value}" for i, value in enumerate(entries))
    return query_list


def process_remove_query(number):
    """
    Process the removal of a query from the database.

    Args:
        number (str): The number of the query to remove or "all" to remove all queries

    Returns:
        tuple: (message, success)
            - message (str): Status message
            - success (bool): True if query was removed successfully
    """
    if number == "all":
        db.remove_all_queries_from_db()
        return "All queries removed.", True

    # Check if number is a valid digit
    if not number[0].isdigit():
        return "Invalid number.", False

    # Remove the query from the database
    db.remove_query_from_db(number)
    return "Query removed.", True


def process_add_country(country):
    """
    Process the addition of a country to the allowlist.

    Args:
        country (str): The country code to add

    Returns:
        tuple: (message, country_list)
            - message (str): Status message
            - country_list (list): Current list of allowed countries
    """
    # Format the country code (remove spaces)
    country = country.replace(" ", "")
    country_list = db.get_allowlist()

    # Validate the country code (check if it's 2 characters long)
    if len(country) != 2:
        return "Invalid country code", country_list

    # Check if the country is already in the allowlist
    # If country_list is 0, it means the allowlist is empty
    if country_list != 0 and country.upper() in country_list:
        return f'Country "{country.upper()}" already in allowlist.', country_list

    # Add the country to the allowlist
    db.add_to_allowlist(country.upper())
    return "Country added.", db.get_allowlist()


def process_remove_country(country):
    """
    Process the removal of a country from the allowlist.

    Args:
        country (str): The country code to remove

    Returns:
        tuple: (message, country_list)
            - message (str): Status message
            - country_list (list): Current list of allowed countries
    """
    # Format the country code (remove spaces)
    country = country.replace(" ", "")

    # Validate the country code (check if it's 2 characters long)
    if len(country) != 2:
        return "Invalid country code", db.get_allowlist()

    # Remove the country from the allowlist
    db.remove_from_allowlist(country.upper())
    return "Country removed.", db.get_allowlist()


def get_user_country(profile_id):
    """
    Get the country code for a Vinted user.

    Makes an API request to retrieve the user's country code.
    Handles rate limiting by trying an alternative endpoint.

    Args:
        profile_id (str): The Vinted user's profile ID

    Returns:
        str: The user's country code (2-letter ISO code) or "XX" if it can't be determined
    """
    # Users are shared between all Vinted platforms, so we can use whatever locale we want
    url = f"https://www.vinted.fr/api/v2/users/{profile_id}?localize=false"
    response = requester.get(url)
    # That's a LOT of requests, so if we get a 429 we wait a bit before retrying once
    if response.status_code == 429:
        # In case of rate limit, we're switching the endpoint. This one is slower, but it doesn't RL as soon. 
        # We're limiting the items per page to 1 to grab as little data as possible
        url = f"https://www.vinted.fr/api/v2/users/{profile_id}/items?page=1&per_page=1"
        response = requester.get(url)
        try:
            user_country = response.json()["items"][0]["user"]["country_iso_code"]
        except KeyError:
            logger.warning("Couldn't get the country due to too many requests. Returning default value.")
            user_country = "XX"
    else:
        user_country = response.json()["user"]["country_iso_code"]
    return user_country


def process_items(queue):
    """
    Process all queries from the database, search for items, and put them in the queue.
    Uses the global items_queue by default, but can accept a custom queue for backward compatibility.

    Args:
        queue (Queue, optional): The queue to put the items in. Defaults to the global items_queue.

    Returns:
        None
    """

    all_queries = db.get_queries()

    # Initialize Vinted
    vinted = Vinted()

    # Get the number of items per query from the database
    items_per_query = int(db.get_parameter("items_per_query"))

    # for each keyword we parse data
    for query in all_queries:
        all_items = vinted.items.search(query[1], nbr_items=items_per_query)
        # Filter to only include new items. This should reduce the amount of db calls.
        data = [item for item in all_items if item.is_new_item()]
        queue.put((data, query[0], query[1], query[3]))
        logger.info(f"Scraped {len(data)} items for query: {query[1]}")


def clear_item_queue(items_queue, new_items_queue):
    """
    Process items from the items_queue.
    This function is scheduled to run frequently.
    """
    if not items_queue.empty():
        payload = items_queue.get()
        if isinstance(payload, (list, tuple)):
            if len(payload) >= 4:
                data, query_id, query_url, stored_name = payload[:4]
            elif len(payload) == 3:
                data, query_id, query_url = payload
                stored_name = None
            else:
                data, query_id = payload[:2]
                query_url = None
                stored_name = None
        else:
            data = payload
            query_id = None
            query_url = None
            stored_name = None

        if query_id is None:
            logger.warning("Received item payload without query id; skipping processing.")
            return

        _, base_search_text = decode_query_name(stored_name)
        if base_search_text is None and query_url:
            parsed_query = urlparse(query_url)
            query_params = parse_qs(parsed_query.query)
            base_search_text = query_params.get('search_text', [None])[0]

        allowlist = db.get_allowlist()
        allowed_countries = allowlist if allowlist != 0 else None

        for item in reversed(data):

            # If already in db, pass
            last_query_timestamp = db.get_last_timestamp(query_id)
            if last_query_timestamp is not None and last_query_timestamp >= item.raw_timestamp:
                pass
            # In case of multiple queries, we need to check if the item is already in the db
            elif db.is_item_in_db_by_id(item.id) is True:
                # We update the timestamp
                db.update_last_timestamp(query_id, item.raw_timestamp)
                pass
            # If there's an allowlist and
            # If the user's country is not in the allowlist, we just update the timestamp
            elif allowed_countries is not None and (
                    get_user_country(item.raw_data["user"]["id"])) not in (allowed_countries + ["XX"]):
                db.update_last_timestamp(query_id, item.raw_timestamp)
                pass
            else:
                fuzzy_result = None
                if base_search_text:
                    fuzzy_result = find_best_fuzzy_match(
                        base_search_text,
                        item.title,
                        item.brand_title,
                        threshold=DEFAULT_FUZZY_THRESHOLD,
                    )
                    if fuzzy_result is None:
                        db.update_last_timestamp(query_id, item.raw_timestamp)
                        logger.debug(
                            "Skipping item %s for query %s due to fuzzy mismatch against '%s'",
                            item.id,
                            query_id,
                            base_search_text,
                        )
                        continue

                valuation_engine = _get_valuation_engine()
                valuation = valuation_engine.evaluate(
                    item.title,
                    item.brand_title,
                    item.price,
                    item.currency,
                    base_search_text,
                    fuzzy_result,
                )

                market_summary = format_market_summary(valuation.sold_summary)
                active_summary = format_active_summary(valuation.active_summary)
                fuzzy_line = format_fuzzy_line(valuation.normalization)
                profit_line = format_profit_estimate(valuation)
                confidence_line = format_confidence_line(valuation)
                reference_line = build_reference_line(valuation)

                # We create the message
                content = configuration_values.MESSAGE.format(
                    title=item.title,
                    price=format_money(item.price, item.currency),
                    brand=item.brand_title or "N/A",
                    market_comps=market_summary,
                    active_listings=active_summary,
                    fuzzy_match=fuzzy_line,
                    profit_estimate=profit_line,
                    confidence=confidence_line,
                    reference=reference_line,
                    image=item.photo or "",
                )
                # add the item to the queue
                new_items_queue.put((content, item.url, "Open Vinted", None, None))
                # new_items_queue.put((content, item.url, "Open Vinted", item.buy_url, "Open buy page"))
                # Add the item to the db
                db.add_item_to_db(id=item.id, timestamp=item.raw_timestamp, price=item.price, title=item.title,
                                  photo_url=item.photo, query_id=query_id, currency=item.currency)


def check_version():
    """
    Check if the application is up to date
    """
    try:
        # Get URL from the database
        github_url = db.get_parameter("github_url")
        # Get version from the database
        ver = db.get_parameter("version")
        # Get latest version from the repository
        url = f"{github_url}/releases/latest"
        response = requests.get(url)

        if response.status_code == 200:
            latest_version = response.url.split('/')[-1]
            is_up_to_date = (ver == latest_version)
            return is_up_to_date, ver, latest_version, github_url
        else:
            # If we can't check, assume it's up to date
            return True, ver, ver, github_url
    except Exception as e:
        logger.error(f"Error checking for new version: {str(e)}", exc_info=True)
        # If we can't check, assume it's up to date
        return True, ver, ver, github_url
