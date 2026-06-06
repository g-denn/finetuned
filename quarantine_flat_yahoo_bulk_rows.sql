update public.performance_validation
set
    validation_status = 'needs_manual_review',
    label_quality = 'low',
    include_in_training = false,
    return_confidence = least(return_confidence, 0.40),
    validation_reason = concat(
        coalesce(validation_reason, ''),
        ' Quarantined: bulk Yahoo validation produced exact 1.0x for 1y, 3y, and 5y, which is suspicious for old historical rows and may indicate stale/missing provider data rather than real no-change performance.'
    ),
    failure_modes = coalesce(failure_modes, '[]'::jsonb) || '["flat_all_horizons_suspicious_yahoo_data"]'::jsonb,
    reviewed_by = 'codex_quarantine_flat_yahoo_v1',
    reviewed_at = now(),
    updated_at = now()
where reviewed_by = 'codex_bulk_simple_validator_v1'
  and include_in_training = true
  and validated_perf_1y = 1
  and validated_perf_3y = 1
  and validated_perf_5y = 1;
