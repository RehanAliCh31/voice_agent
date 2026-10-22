-- The bank's tables.
--
-- Money is integer cents everywhere. Floats accumulate rounding error, and a
-- transfer that is a hundredth of a penny wrong is a bug nobody wants to debug.
--
-- `transactions.cents` is NEGATIVE for money leaving the account. Storing the
-- sign rather than a separate direction column means a balance is the sum of
-- its ledger, and a refund is just a positive row.

CREATE TABLE IF NOT EXISTS accounts (
    name          TEXT PRIMARY KEY,
    balance_cents INTEGER NOT NULL CHECK (balance_cents >= 0)
);

CREATE TABLE IF NOT EXISTS cards (
    last4   TEXT PRIMARY KEY,
    account TEXT NOT NULL REFERENCES accounts(name),
    frozen  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bills (
    name         TEXT PRIMARY KEY,
    amount_cents INTEGER NOT NULL,
    paid         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    account     TEXT NOT NULL REFERENCES accounts(name),
    merchant    TEXT NOT NULL,
    cents       INTEGER NOT NULL,
    occurred_at TEXT NOT NULL
);

-- The audit log: what was said, what was called, and whether the numeric guard
-- had to suppress the model's sentence. Survives a reset; the account tables
-- do not.
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn_index  INTEGER NOT NULL,
    user_text   TEXT,
    bot_text    TEXT,
    tool_calls  TEXT,
    guard_hit   INTEGER NOT NULL DEFAULT 0,
    suppressed  TEXT,
    latency_ms  REAL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_tx_occurred   ON transactions(occurred_at);
