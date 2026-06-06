create table if not exists public.eodhd_fundamentals (
    symbol text primary key,
    provider_version text not null default 'v1.1',
    code text,
    instrument_type text,
    name text,
    exchange text,
    currency text,
    country_name text,
    country_iso text,
    isin text,
    primary_ticker text,
    cik text,
    ipo_date date,
    sector text,
    industry text,
    home_category text,
    is_delisted boolean,
    delisted_date date,
    market_cap double precision,
    revenue_ttm double precision,
    ebitda double precision,
    gross_profit_ttm double precision,
    profit_margin double precision,
    operating_margin_ttm double precision,
    return_on_equity_ttm double precision,
    pe_ratio double precision,
    price_sales_ttm double precision,
    last_split_factor text,
    last_split_date date,
    latest_yearly_income_date date,
    latest_quarterly_income_date date,
    yearly_revenue_first double precision,
    yearly_revenue_last double precision,
    yearly_net_income_first double precision,
    yearly_net_income_last double precision,
    quarterly_revenue_first double precision,
    quarterly_revenue_last double precision,
    quarterly_net_income_first double precision,
    quarterly_net_income_last double precision,
    has_financials boolean not null default false,
    highlights jsonb not null default '{}'::jsonb,
    valuation jsonb not null default '{}'::jsonb,
    splits_dividends jsonb not null default '{}'::jsonb,
    earnings jsonb not null default '{}'::jsonb,
    financials jsonb not null default '{}'::jsonb,
    holders jsonb not null default '{}'::jsonb,
    insider_transactions jsonb not null default '{}'::jsonb,
    etf_data jsonb not null default '{}'::jsonb,
    mutual_fund_data jsonb not null default '{}'::jsonb,
    provider_payload jsonb not null default '{}'::jsonb,
    provider_warnings jsonb not null default '[]'::jsonb,
    fetched_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    constraint eodhd_fundamentals_provider_warnings_array_chk
        check (jsonb_typeof(provider_warnings) = 'array')
);

create index if not exists eodhd_fundamentals_instrument_type_idx
    on public.eodhd_fundamentals (instrument_type, symbol);

create index if not exists eodhd_fundamentals_delisted_idx
    on public.eodhd_fundamentals (is_delisted, delisted_date);

create index if not exists eodhd_fundamentals_sector_idx
    on public.eodhd_fundamentals (sector, industry);

alter table public.eodhd_fundamentals enable row level security;

alter table public.performance_validation
    add column if not exists eodhd_fundamentals_symbol text,
    add column if not exists fundamentals_cache_status text not null default 'not_fetched',
    add column if not exists fundamentals_cache_checked_at timestamptz,
    add column if not exists fundamentals_summary jsonb not null default '{}'::jsonb;

create index if not exists performance_validation_fundamentals_status_idx
    on public.performance_validation (fundamentals_cache_status, fundamentals_cache_checked_at desc);
