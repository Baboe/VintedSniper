-- init_schema.sql
-- Initial Scheme

PRAGMA foreign_keys = ON;

/* ============================
   Tables
   ============================ */

-- Queries table
CREATE TABLE IF NOT EXISTS queries
(
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    query     TEXT,
    last_item NUMERIC,
    query_name TEXT
);

-- Items table
CREATE TABLE IF NOT EXISTS items
(
    item      NUMERIC,
    title     TEXT,
    price     NUMERIC,
    currency  TEXT,
    timestamp NUMERIC,
    photo_url TEXT,
    query_id  INTEGER,
    FOREIGN KEY (query_id) REFERENCES queries (id)
);

-- Allowlist table
CREATE TABLE IF NOT EXISTS allowlist
(
    country TEXT
);

-- Parameters table
CREATE TABLE IF NOT EXISTS parameters
(
    key   TEXT PRIMARY KEY,
    value TEXT
);

/* ============================
   Initial data
   ============================ */

INSERT INTO parameters (key, value)
VALUES ('telegram_enabled', 'True'),
       ('telegram_token', '8216513583:AAHIhHE6jYq-h3Lz1EStGgrKh3xzPA3Piqw'),
       ('telegram_chat_id', '7024796766'),
       ('telegram_process_running', 'False'),

       ('rss_enabled', 'False'),
       ('rss_port', '8080'),
       ('rss_max_items', '100'),
       ('rss_process_running', 'False'),

       ('version', '1.0.4'),
       ('github_url', 'https://github.com/Fuyucch1/Vinted-Notifications'),

       ('items_per_query', '20'),
       ('query_refresh_delay', '60'),

       ('proxy_list', ''),
       ('proxy_list_link', ''),
       ('check_proxies', 'False'),
       ('last_proxy_check_time', '0');
