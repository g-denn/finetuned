create table if not exists public.performance_yahoo (
    idea_id character varying primary key,
    raw_symbol text,
    yahoo_symbol text,
    source_status text not null,
    source_error text,
    publication_date date,
    base_trade_date date,
    base_close double precision,
    base_adj_close double precision,
    base_adjustment_factor double precision,
    next_trade_date date,
    next_open double precision,
    next_close double precision,
    next_adj_close double precision,
    price_1w double precision,
    adj_price_1w double precision,
    perf_1w double precision,
    short_perf_1w double precision,
    trade_date_1w date,
    price_2w double precision,
    adj_price_2w double precision,
    perf_2w double precision,
    short_perf_2w double precision,
    trade_date_2w date,
    price_1m double precision,
    adj_price_1m double precision,
    perf_1m double precision,
    short_perf_1m double precision,
    trade_date_1m date,
    price_3m double precision,
    adj_price_3m double precision,
    perf_3m double precision,
    short_perf_3m double precision,
    trade_date_3m date,
    price_6m double precision,
    adj_price_6m double precision,
    perf_6m double precision,
    short_perf_6m double precision,
    trade_date_6m date,
    price_1y double precision,
    adj_price_1y double precision,
    perf_1y double precision,
    short_perf_1y double precision,
    trade_date_1y date,
    price_2y double precision,
    adj_price_2y double precision,
    perf_2y double precision,
    short_perf_2y double precision,
    trade_date_2y date,
    price_3y double precision,
    adj_price_3y double precision,
    perf_3y double precision,
    short_perf_3y double precision,
    trade_date_3y date,
    price_5y double precision,
    adj_price_5y double precision,
    perf_5y double precision,
    short_perf_5y double precision,
    trade_date_5y date,
    split_events jsonb not null default '[]'::jsonb,
    dividend_events jsonb not null default '[]'::jsonb,
    checked_at timestamptz not null default now(),
    yahoo_payload_range daterange,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists performance_yahoo_status_idx
    on public.performance_yahoo (source_status);

create index if not exists performance_yahoo_symbol_idx
    on public.performance_yahoo (yahoo_symbol);

create index if not exists performance_yahoo_checked_idx
    on public.performance_yahoo (checked_at);

alter table public.performance_yahoo enable row level security;

create or replace view public.performance_yahoo_quality
with (security_invoker = true) as
select
    py.idea_id,
    py.raw_symbol,
    py.yahoo_symbol,
    py.publication_date,
    py.source_status,
    py.source_error,
    py.base_trade_date,
    py.base_adj_close,
    py.perf_6m,
    py.perf_1y,
    py.perf_3y,
    py.perf_5y,
    p."sixMonthPerf" as legacy_six_month_perf,
    p."oneYearPerf" as legacy_one_year_perf,
    p."threeYearPerf" as legacy_three_year_perf,
    p."fiveYearPerf" as legacy_five_year_perf,
    case
        when p."oneYearPerf" is null or py.perf_1y is null then null
        else abs(py.perf_1y - p."oneYearPerf") / nullif(abs(p."oneYearPerf"), 0)
    end as one_year_relative_diff
from public.performance_yahoo py
left join public.performance p on p.idea_id = py.idea_id;
