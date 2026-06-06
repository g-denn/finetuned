alter table public.performance_validation
    add column if not exists math_validation_status text not null default 'unknown',
    add column if not exists review_stage text not null default 'unreviewed',
    add column if not exists training_readiness text not null default 'not_training_ready',
    add column if not exists manual_review_priority integer,
    add column if not exists manual_review_reason text,
    add column if not exists manual_review_tags text[] not null default '{}'::text[],
    add column if not exists review_target_horizon text,
    add column if not exists review_target_multiplier double precision;

alter table public.performance_validation
    drop constraint if exists performance_validation_math_validation_status_chk;

alter table public.performance_validation
    add constraint performance_validation_math_validation_status_chk
        check (
            math_validation_status in (
                'unknown',
                'provider_error',
                'math_incomplete',
                'math_reproduced',
                'manually_verified',
                'rejected'
            )
        );

alter table public.performance_validation
    drop constraint if exists performance_validation_review_stage_chk;

alter table public.performance_validation
    add constraint performance_validation_review_stage_chk
        check (
            review_stage in (
                'unreviewed',
                'provider_error',
                'math_incomplete',
                'provider_warning',
                'math_reproduced_low_risk',
                'manual_review',
                'manual_pass',
                'manual_reject',
                'training_ready'
            )
        );

alter table public.performance_validation
    drop constraint if exists performance_validation_training_readiness_chk;

alter table public.performance_validation
    add constraint performance_validation_training_readiness_chk
        check (
            training_readiness in (
                'not_training_ready',
                'manual_review_required',
                'candidate_low_risk',
                'training_ready',
                'rejected'
            )
        );

alter table public.performance_validation
    drop constraint if exists performance_validation_manual_review_priority_chk;

alter table public.performance_validation
    add constraint performance_validation_manual_review_priority_chk
        check (manual_review_priority is null or manual_review_priority >= 0);

alter table public.performance_validation
    drop constraint if exists performance_validation_review_target_horizon_chk;

alter table public.performance_validation
    add constraint performance_validation_review_target_horizon_chk
        check (review_target_horizon is null or review_target_horizon in ('1y', '3y', '5y', '10y', '20y'));

update public.performance_validation
set
    math_validation_status = 'manually_verified',
    review_stage = 'training_ready',
    training_readiness = 'training_ready'
where
    include_in_training = true
    and validation_status in (
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
    and coalesce(return_confidence, 0) >= 0.75;

alter table public.performance_validation
    drop constraint if exists performance_validation_training_gate_chk;

alter table public.performance_validation
    add constraint performance_validation_training_gate_chk
        check (
            include_in_training = false
            or (
                training_readiness = 'training_ready'
                and review_stage = 'training_ready'
                and math_validation_status = 'manually_verified'
                and validation_status in (
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

create index if not exists performance_validation_review_queue_idx
    on public.performance_validation (
        training_readiness,
        manual_review_priority nulls last,
        publication_date,
        raw_symbol
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
    pv.math_validation_status,
    pv.review_stage,
    pv.training_readiness,
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
    and pv.training_readiness = 'training_ready'
    and pv.review_stage = 'training_ready'
    and pv.math_validation_status = 'manually_verified'
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
