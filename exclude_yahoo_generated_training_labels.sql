update public.performance_validation
set
    include_in_training = false,
    validation_status = 'exclude_from_training',
    label_quality = 'unusable',
    return_confidence = least(coalesce(return_confidence, 0), 0.10),
    validation_reason = concat(
        coalesce(validation_reason, ''),
        ' Excluded on 2026-05-29: Yahoo-generated validation labels are being removed from training pending replacement by a paid market-data provider.'
    ),
    failure_modes = coalesce(failure_modes, '[]'::jsonb) || '["removed_yahoo_generated_label_pending_paid_provider"]'::jsonb,
    reviewed_by = 'codex_removed_yahoo_generated_v1',
    reviewed_at = now(),
    updated_at = now()
where include_in_training = true
  and idea_id in (
      select idea_id
      from public.performance_training_labels_v1
  );
