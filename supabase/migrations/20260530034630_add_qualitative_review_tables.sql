alter table public.performance_validation
    add column if not exists agent_c_result jsonb not null default '{}'::jsonb,
    add column if not exists business_quality_status text not null default 'unknown';

alter table public.performance_validation
    drop constraint if exists performance_validation_business_quality_status_chk;

alter table public.performance_validation
    add constraint performance_validation_business_quality_status_chk
        check (
            business_quality_status in (
                'not_required',
                'qualitatively_supported',
                'conflicting_business_reality',
                'unsupported_extreme_return',
                'bad_provider_data',
                'unknown'
            )
        );

alter table public.performance_validation
    drop constraint if exists performance_validation_training_gate_chk;

alter table public.performance_validation
    add constraint performance_validation_training_gate_chk
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
                and coalesce(agent_c_result ->> 'reviewer_status', '') in ('pass', 'not_required')
                and business_quality_status in ('qualitatively_supported', 'not_required')
                and coalesce(identity_confidence, 0) >= 0.85
                and coalesce(return_confidence, 0) >= 0.75
            )
        );

create table if not exists public.company_outcome_research (
    id bigserial primary key,
    idea_id text,
    raw_symbol text not null,
    eodhd_symbol text,
    company_name text,
    publication_date date not null,
    outcome_type text not null default 'unknown',
    business_explanation text,
    revenue_growth_evidence text,
    profitability_evidence text,
    market_cap_evidence text,
    liquidity_evidence text,
    corporate_action_evidence text,
    sources jsonb not null default '[]'::jsonb,
    confidence double precision,
    reviewer text,
    reviewed_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    constraint company_outcome_research_outcome_type_chk
        check (
            outcome_type in (
                'ordinary_public_company',
                'extreme_winner',
                'acquisition_cash',
                'acquisition_stock',
                'merger',
                'bankruptcy',
                'liquidation',
                'delisted_otc',
                'ticker_change',
                'spin_off',
                'bad_provider_data',
                'unknown'
            )
        ),
    constraint company_outcome_research_sources_array_chk
        check (jsonb_typeof(sources) = 'array'),
    constraint company_outcome_research_confidence_chk
        check (confidence is null or (confidence >= 0 and confidence <= 1))
);

create unique index if not exists company_outcome_research_idea_idx
    on public.company_outcome_research (idea_id)
    where idea_id is not null;

create index if not exists company_outcome_research_symbol_date_idx
    on public.company_outcome_research (raw_symbol, publication_date);

create index if not exists company_outcome_research_outcome_type_idx
    on public.company_outcome_research (outcome_type, reviewed_at desc);

alter table public.company_outcome_research enable row level security;

create table if not exists public.performance_manual_review (
    idea_id text not null,
    horizon text not null,
    eodhd_return double precision,
    yahoo_return double precision,
    verified_return double precision,
    verdict text not null default 'manual_review',
    reason text,
    qualitative_summary text,
    sources jsonb not null default '[]'::jsonb,
    agent_a_result jsonb not null default '{}'::jsonb,
    agent_b_result jsonb not null default '{}'::jsonb,
    agent_c_result jsonb not null default '{}'::jsonb,
    reviewer text,
    reviewed_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (idea_id, horizon),

    constraint performance_manual_review_horizon_chk
        check (horizon in ('1y', '3y', '5y', '10y', '20y')),
    constraint performance_manual_review_verdict_chk
        check (verdict in ('pass', 'reject', 'manual_review')),
    constraint performance_manual_review_sources_array_chk
        check (jsonb_typeof(sources) = 'array')
);

create index if not exists performance_manual_review_verdict_idx
    on public.performance_manual_review (verdict, reviewed_at desc);

alter table public.performance_manual_review enable row level security;

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
    pv.business_quality_status,
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
    and coalesce(pv.agent_c_result ->> 'reviewer_status', '') in ('pass', 'not_required')
    and pv.business_quality_status in ('qualitatively_supported', 'not_required')
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
