create table if not exists public.performance_validation (
    idea_id text primary key,
    raw_symbol text,
    yahoo_symbol text,
    company_name text,
    publication_date date,
    position_type text not null default 'unknown',

    validation_status text not null default 'unreviewed',
    identity_status text not null default 'unknown',
    corporate_action_status text not null default 'unknown',
    label_quality text not null default 'unusable',
    include_in_training boolean not null default false,

    validated_perf_1y double precision,
    validated_perf_3y double precision,
    validated_perf_5y double precision,
    validated_perf_1y_median_52w double precision,
    validated_perf_3y_median_52w double precision,
    validated_perf_5y_median_52w double precision,

    split_adjusted boolean not null default false,
    dividend_adjusted boolean not null default false,
    spin_off_adjusted boolean not null default false,
    merger_adjusted boolean not null default false,

    identity_confidence double precision,
    return_confidence double precision,
    validation_reason text,

    corporate_action_timeline jsonb not null default '[]'::jsonb,
    agent_a_result jsonb not null default '{}'::jsonb,
    agent_b_result jsonb not null default '{}'::jsonb,
    sources jsonb not null default '[]'::jsonb,
    failure_modes jsonb not null default '[]'::jsonb,

    claimed_by text,
    claimed_at timestamptz,
    reviewed_by text,
    reviewed_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    constraint performance_validation_position_type_chk
        check (position_type in ('long', 'short', 'unknown')),
    constraint performance_validation_status_chk
        check (
            validation_status in (
                'unreviewed',
                'in_progress',
                'verified_exact',
                'verified_with_corporate_action',
                'verified_successor_security',
                'verified_delisted_otc',
                'verified_delisted_zero_or_liquidation',
                'identity_conflict',
                'ticker_reuse_conflict',
                'insufficient_evidence',
                'bad_yahoo_adjustment',
                'exclude_from_training',
                'needs_manual_review',
                'provider_error'
            )
        ),
    constraint performance_validation_identity_status_chk
        check (
            identity_status in (
                'same_security',
                'ticker_changed',
                'acquired_cash',
                'acquired_stock',
                'acquired_mixed',
                'delisted_otc',
                'delisted_bankrupt',
                'liquidated',
                'ticker_reuse_suspected',
                'unsupported_instrument',
                'unknown'
            )
        ),
    constraint performance_validation_corporate_action_status_chk
        check (
            corporate_action_status in (
                'none_detected',
                'adjusted_by_provider',
                'manually_modeled',
                'partially_modeled',
                'missing_material_action',
                'conflicting_action_data',
                'unknown'
            )
        ),
    constraint performance_validation_label_quality_chk
        check (label_quality in ('high', 'medium', 'low', 'unusable')),
    constraint performance_validation_identity_confidence_chk
        check (identity_confidence is null or (identity_confidence >= 0 and identity_confidence <= 1)),
    constraint performance_validation_return_confidence_chk
        check (return_confidence is null or (return_confidence >= 0 and return_confidence <= 1)),
    constraint performance_validation_training_gate_chk
        check (
            include_in_training = false
            or (
                validation_status in (
                    'verified_exact',
                    'verified_with_corporate_action',
                    'verified_successor_security',
                    'verified_delisted_otc',
                    'verified_delisted_zero_or_liquidation'
                )
                and label_quality in ('high', 'medium')
                and corporate_action_status in ('none_detected', 'adjusted_by_provider', 'manually_modeled')
                and coalesce(agent_b_result ->> 'reviewer_status', '') = 'pass'
                and coalesce(identity_confidence, 0) >= 0.85
                and coalesce(return_confidence, 0) >= 0.75
            )
        )
);

create index if not exists performance_validation_status_idx
    on public.performance_validation (validation_status, updated_at);

create index if not exists performance_validation_identity_idx
    on public.performance_validation (identity_status);

create index if not exists performance_validation_label_quality_idx
    on public.performance_validation (label_quality);

create index if not exists performance_validation_training_idx
    on public.performance_validation (include_in_training)
    where include_in_training = true;

alter table public.performance_validation enable row level security;

insert into public.performance_validation (
    idea_id,
    raw_symbol,
    yahoo_symbol,
    company_name,
    publication_date,
    position_type,
    validation_status,
    identity_status,
    corporate_action_status,
    label_quality,
    include_in_training,
    split_adjusted,
    dividend_adjusted,
    spin_off_adjusted,
    merger_adjusted,
    corporate_action_timeline,
    sources,
    failure_modes,
    updated_at
)
select
    py.idea_id,
    py.raw_symbol,
    py.yahoo_symbol,
    c.company_name,
    py.publication_date,
    case
        when i.is_short is true then 'short'
        when i.is_short is false then 'long'
        else 'unknown'
    end as position_type,
    'unreviewed',
    'unknown',
    case
        when jsonb_array_length(coalesce(py.split_events, '[]'::jsonb)) > 0
            or jsonb_array_length(coalesce(py.dividend_events, '[]'::jsonb)) > 0
            then 'adjusted_by_provider'
        else 'none_detected'
    end as corporate_action_status,
    'unusable',
    false,
    jsonb_array_length(coalesce(py.split_events, '[]'::jsonb)) > 0,
    jsonb_array_length(coalesce(py.dividend_events, '[]'::jsonb)) > 0,
    false,
    false,
    '[]'::jsonb,
    jsonb_build_array(
        jsonb_build_object(
            'source_id', 'provider:yahoo',
            'url', 'https://query1.finance.yahoo.com/v8/finance/chart',
            'publisher', 'Yahoo Finance',
            'source_type', 'data_vendor',
            'accessed_date', current_date::text,
            'supports', 'price',
            'quote_or_fact', 'Raw provider adjusted close, split, and dividend events used as evidence only.'
        )
    ),
    case
        when py.perf_5y is not null and (py.perf_5y > 20 or py.perf_5y < 0.05)
            then jsonb_build_array('extreme_return_requires_identity_review')
        when jsonb_array_length(coalesce(py.split_events, '[]'::jsonb)) > 0
            then jsonb_build_array('provider_split_requires_identity_review')
        else '[]'::jsonb
    end,
    now()
from public.performance_yahoo py
left join public.ideas i on i.id = py.idea_id
left join public.companies c on c.ticker = py.raw_symbol
on conflict (idea_id) do nothing;

create or replace view public.performance_validation_queue_v1
with (security_invoker = true) as
select
    py.idea_id,
    py.raw_symbol,
    py.yahoo_symbol,
    c.company_name,
    py.publication_date,
    case
        when i.is_short is true then 'short'
        when i.is_short is false then 'long'
        else 'unknown'
    end as position_type,
    py.source_status,
    py.source_error,
    py.base_adj_close,
    py.perf_1y,
    py.perf_3y,
    py.perf_5y,
    py.trade_date_1y,
    py.trade_date_3y,
    py.trade_date_5y,
    py.split_events,
    py.dividend_events,
    coalesce(pv.validation_status, 'unreviewed') as validation_status,
    coalesce(pv.identity_status, 'unknown') as identity_status,
    coalesce(pv.corporate_action_status, 'unknown') as corporate_action_status,
    pv.claimed_by,
    pv.claimed_at,
    case
        when py.perf_5y is not null and (py.perf_5y > 20 or py.perf_5y < 0.05) then 10
        when py.raw_symbol is distinct from py.yahoo_symbol then 20
        when py.raw_symbol ~ '[^A-Za-z0-9 .:-]' then 30
        when py.publication_date < date '2010-01-01' then 40
        when i.is_short is true then 50
        when py.perf_1y is null or py.perf_3y is null or py.perf_5y is null then 60
        else 100
    end as risk_priority
from public.performance_yahoo py
left join public.performance_validation pv on pv.idea_id = py.idea_id
left join public.ideas i on i.id = py.idea_id
left join public.companies c on c.ticker = py.raw_symbol
where
    coalesce(pv.validation_status, 'unreviewed') in (
        'unreviewed',
        'provider_error',
        'needs_manual_review',
        'insufficient_evidence'
    )
    or (
        pv.validation_status = 'in_progress'
        and pv.claimed_at < now() - interval '2 hours'
    );

create or replace view public.performance_training_labels_v1
with (security_invoker = true) as
select
    pv.idea_id,
    pv.raw_symbol,
    pv.yahoo_symbol,
    pv.company_name,
    pv.publication_date,
    pv.position_type,
    pv.validated_perf_1y,
    pv.validated_perf_3y,
    pv.validated_perf_5y,
    pv.validated_perf_1y_median_52w,
    pv.validated_perf_3y_median_52w,
    pv.validated_perf_5y_median_52w,
    pv.validation_status,
    pv.identity_status,
    pv.corporate_action_status,
    pv.label_quality,
    pv.identity_confidence,
    pv.return_confidence,
    pv.validation_reason,
    pv.sources,
    pv.reviewed_at
from public.performance_validation pv
where
    pv.include_in_training = true
    and pv.label_quality in ('high', 'medium')
    and coalesce(pv.agent_b_result ->> 'reviewer_status', '') = 'pass'
    and pv.validation_status in (
        'verified_exact',
        'verified_with_corporate_action',
        'verified_successor_security',
        'verified_delisted_otc',
        'verified_delisted_zero_or_liquidation'
    )
    and pv.corporate_action_status in (
        'none_detected',
        'adjusted_by_provider',
        'manually_modeled'
    )
    and pv.validation_status not in (
        'ticker_reuse_conflict',
        'identity_conflict',
        'insufficient_evidence',
        'needs_manual_review'
    );
