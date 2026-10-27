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

-- The number and the dates are held because the details screen shows them,
-- the way a bank's own app does once you are past the lock screen. These are
-- demo cards and the numbers are invented; with real ones this column is
-- where encryption at rest and an access log would belong.
--
-- There is no security code column and there is not meant to be one. The
-- code shown on that screen is minted for a single viewing and forgotten,
-- so there is nothing here to leak, log, or back up by accident.
--
-- The PIN is here, but never as a PIN: a per-card salt and a hash of the
-- two, so the column cannot be read back into the number somebody types.
-- The failure count and the lock are beside it, because a four-digit secret
-- with unlimited guesses is a four-digit secret anybody has.
--
-- The three controls default to on, which is how a card arrives. They are
-- columns rather than a screen's idea, because a switch nothing can change
-- is a lie told in CSS.
CREATE TABLE IF NOT EXISTS cards (
    last4      TEXT PRIMARY KEY,
    account    TEXT NOT NULL REFERENCES accounts(name),
    frozen     INTEGER NOT NULL DEFAULT 0,
    number     TEXT NOT NULL DEFAULT '',
    valid_from TEXT NOT NULL DEFAULT '',
    valid_to   TEXT NOT NULL DEFAULT '',
    pin_hash   TEXT NOT NULL DEFAULT '',
    pin_salt   TEXT NOT NULL DEFAULT '',
    pin_failures     INTEGER NOT NULL DEFAULT 0,
    pin_locked_until TEXT NOT NULL DEFAULT '',
    kind      TEXT NOT NULL DEFAULT 'debit',
    holder    TEXT NOT NULL DEFAULT '',
    ecommerce INTEGER NOT NULL DEFAULT 1,
    retail    INTEGER NOT NULL DEFAULT 1,
    atm       INTEGER NOT NULL DEFAULT 1
);

-- What leaves every month whether or not anybody thinks about it: the
-- electricity, the rent, the groceries, the wifi, the subscriptions. One
-- table rather than two, so there is one answer to "what do I owe" and
-- everything the monthly screen shows can be paid by voice.
CREATE TABLE IF NOT EXISTS bills (
    name         TEXT PRIMARY KEY,
    amount_cents INTEGER NOT NULL,
    paid         INTEGER NOT NULL DEFAULT 0,
    category     TEXT NOT NULL DEFAULT 'bills',
    due_day      INTEGER NOT NULL DEFAULT 1
);

-- Who money may be sent to. A closed list, and deliberately separate from
-- accounts: "send 500 to savings" used to debit checking and credit nothing,
-- because a payee was any string at all. An account cannot be a payee now,
-- so that sentence is refused instead of half-performed.
CREATE TABLE IF NOT EXISTS payees (
    name TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    account     TEXT NOT NULL REFERENCES accounts(name),
    merchant    TEXT NOT NULL,
    cents       INTEGER NOT NULL,
    occurred_at TEXT NOT NULL
);

-- Charges the customer says are wrong. Always open when the bot writes one:
-- deciding a dispute needs evidence the bot has not got and authority it
-- should not be given, so it files and a person rules.
--
-- The charge is copied in rather than referenced. A dispute is a claim about
-- what happened on a date, so it has to stay readable whatever becomes of the
-- ledger row -- and the demo reset empties that table. `transaction_id` is
-- kept to locate the row while it exists, and is UNIQUE so one charge cannot
-- be raised twice; ids are AUTOINCREMENT and never reused, so a cleared
-- ledger cannot collide with an old case.
CREATE TABLE IF NOT EXISTS disputes (
    reference      TEXT PRIMARY KEY,
    transaction_id INTEGER NOT NULL UNIQUE,
    merchant       TEXT NOT NULL,
    amount_cents   INTEGER NOT NULL,
    occurred_at    TEXT NOT NULL,
    reason         TEXT NOT NULL,
    note           TEXT,
    status         TEXT NOT NULL DEFAULT 'open',
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Conversations that left the bot for a person. The transcript is not copied
-- here: `turns` already holds it for the session, and two copies of the same
-- exchange are two things that can disagree. The reference is what the
-- customer is given and what the specialist searches by.
CREATE TABLE IF NOT EXISTS handoffs (
    reference  TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    reason     TEXT NOT NULL,
    note       TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_handoffs_session ON handoffs(session_id);

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
