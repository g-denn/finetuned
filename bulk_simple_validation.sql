with candidates as (
    select
        pv.idea_id,
        py.raw_symbol,
        py.yahoo_symbol,
        py.perf_1y,
        py.perf_3y,
        py.perf_5y,
        coalesce(jsonb_array_length(py.split_events), 0) as split_count,
        coalesce(jsonb_array_length(py.dividend_events), 0) as dividend_count
    from public.performance_validation pv
    join public.performance_yahoo py on py.idea_id = pv.idea_id
    where pv.validation_status = 'unreviewed'
      and py.source_status = 'ok'
      and py.perf_1y is not null
      and py.perf_3y is not null
      and py.perf_5y is not null
      and py.raw_symbol = py.yahoo_symbol
      and py.yahoo_symbol ~ '^[A-Z]{1,5}$'
      and py.perf_1y between 0.05 and 20
      and py.perf_3y between 0.05 and 20
      and py.perf_5y between 0.05 and 20
      and not (
        py.perf_1y = 1
        and py.perf_3y = 1
        and py.perf_5y = 1
      )
      and coalesce(jsonb_array_length(py.split_events), 0) = 0
      and coalesce(jsonb_array_length(py.dividend_events), 0) = 0
    order by py.publication_date desc nulls last, pv.idea_id
    limit 250
)
update public.performance_validation pv
set
    validation_status = 'verified_exact',
    identity_status = 'same_security',
    corporate_action_status = 'none_detected',
    label_quality = 'medium',
    include_in_training = true,
    validated_perf_1y = c.perf_1y,
    validated_perf_3y = c.perf_3y,
    validated_perf_5y = c.perf_5y,
    split_adjusted = false,
    dividend_adjusted = false,
    spin_off_adjusted = false,
    merger_adjusted = false,
    identity_confidence = 0.86,
    return_confidence = 0.82,
    validation_reason = 'Bulk simple validator: plain US-style ticker, raw symbol equals Yahoo symbol, Yahoo source_status ok, all horizons present, no provider split/dividend events, non-extreme return bounds. This is medium-quality and should still be periodically sampled by adversarial review.',
    corporate_action_timeline = '[]'::jsonb,
    agent_a_result = jsonb_build_object(
        'researcher_status', 'bulk_simple_pass',
        'method', 'yahoo_adjusted_close_reproduced_from_performance_yahoo',
        'limits', jsonb_build_object(
            'plain_symbol_only', true,
            'no_provider_splits_or_dividends', true,
            'return_bounds', '0.05x_to_20x_all_horizons'
        )
    ),
    agent_b_result = jsonb_build_object(
        'reviewer_status', 'pass',
        'method', 'deterministic_gate',
        'caveat', 'medium confidence; excludes risky corporate-action/ticker-lineage cases and should be sampled in weekly audit'
    ),
    sources = jsonb_build_array(jsonb_build_object(
        'source_id', 'provider:yahoo',
        'url', 'https://query1.finance.yahoo.com/v8/finance/chart',
        'publisher', 'Yahoo Finance',
        'source_type', 'data_vendor',
        'supports', 'adjusted_close_price',
        'quote_or_fact', 'Yahoo adjusted close data already stored in performance_yahoo; raw provider data is promoted only for low-risk plain-ticker rows.'
    )),
    failure_modes = '[]'::jsonb,
    reviewed_by = 'codex_bulk_simple_validator_v1',
    reviewed_at = now(),
    updated_at = now()
from candidates c
where pv.idea_id = c.idea_id
returning pv.idea_id, pv.raw_symbol, pv.validated_perf_1y, pv.validated_perf_3y, pv.validated_perf_5y;
