create table if not exists public.eodhd_symbols (
    symbol text primary key,
    code text,
    exchange_code text,
    name text,
    country text,
    exchange text,
    currency text,
    instrument_type text,
    isin text,
    is_delisted boolean not null default false,
    provider_payload jsonb not null default '{}'::jsonb,
    fetched_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.eodhd_prices (
    symbol text not null,
    price_date date not null,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    adjusted_close double precision,
    volume bigint,
    provider_payload jsonb not null default '{}'::jsonb,
    fetched_at timestamptz not null default now(),
    primary key (symbol, price_date)
);

create table if not exists public.eodhd_splits (
    symbol text not null,
    split_date date not null,
    split_ratio text not null,
    numerator double precision,
    denominator double precision,
    provider_payload jsonb not null default '{}'::jsonb,
    fetched_at timestamptz not null default now(),
    primary key (symbol, split_date, split_ratio)
);

create table if not exists public.eodhd_dividends (
    id bigserial primary key,
    symbol text not null,
    ex_date date not null,
    declaration_date date,
    record_date date,
    payment_date date,
    period text,
    value double precision,
    unadjusted_value double precision,
    currency text,
    provider_payload jsonb not null default '{}'::jsonb,
    fetched_at timestamptz not null default now()
);

create table if not exists public.eodhd_symbol_changes (
    change_date date not null,
    old_symbol text not null,
    new_symbol text not null,
    exchange_code text,
    company_name text,
    provider_payload jsonb not null default '{}'::jsonb,
    fetched_at timestamptz not null default now(),
    primary key (change_date, old_symbol, new_symbol)
);

create table if not exists public.eodhd_fetch_log (
    id bigserial primary key,
    endpoint text not null,
    symbol text,
    date_from date,
    date_to date,
    status text not null,
    http_status integer,
    row_count integer,
    warning text,
    error_message text,
    requested_at timestamptz not null default now(),
    completed_at timestamptz
);

alter table public.performance_validation
    add column if not exists eodhd_symbol text,
    add column if not exists validated_perf_10y double precision,
    add column if not exists validated_perf_20y double precision,
    add column if not exists validated_perf_10y_median_52w double precision,
    add column if not exists validated_perf_20y_median_52w double precision,
    add column if not exists provider_cache_status text not null default 'not_fetched',
    add column if not exists provider_cache_checked_at timestamptz;

create index if not exists eodhd_prices_symbol_date_idx
    on public.eodhd_prices (symbol, price_date);

create unique index if not exists eodhd_dividends_symbol_date_value_idx
    on public.eodhd_dividends (symbol, ex_date, coalesce(value, 0));

create index if not exists eodhd_symbols_delisted_idx
    on public.eodhd_symbols (is_delisted, symbol);

create index if not exists eodhd_fetch_log_symbol_idx
    on public.eodhd_fetch_log (symbol, requested_at desc);

create index if not exists performance_validation_eodhd_symbol_idx
    on public.performance_validation (eodhd_symbol);

alter table public.eodhd_symbols enable row level security;
alter table public.eodhd_prices enable row level security;
alter table public.eodhd_splits enable row level security;
alter table public.eodhd_dividends enable row level security;
alter table public.eodhd_symbol_changes enable row level security;
alter table public.eodhd_fetch_log enable row level security;
