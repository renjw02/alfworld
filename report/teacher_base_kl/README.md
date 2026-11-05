# ASCENT teacher-to-base paired KL diagnostic

Frozen Qwen3.5-9B, BF16 forward / FP32 full-vocabulary KL, temperature 1. Both forwards use identical saved student prefix and response tokens. Only the teacher gets the saved privileged prefix. No generation, truncation, LoRA, or optimizer updates.

Action mask includes the action text and its internal whitespace; excludes the Action: marker and EOS. All table KL values are pooled means in nats/token. The sample is deterministic and small: first successful committed episode per task, first/middle/last turns for ASCENT; first turn only for ASCENT+ICL. Two extra ASCENT turns are used only for calibration.

| Group / task | Turns | Action tokens | Action KL | Reasoning KL | First-action KL | Action top-1 flips |
|---|---:|---:|---:|---:|---:|---:|
| ascent | 18 | 118 | 0.00023760403 | 0.17836672 | 0.0013684224 | 0/118 |
| ascent_icl | 3 | 15 | 0.00012664807 | 0.14190125 | 0.00060763842 | 0/15 |
| ascent/pick_and_place_simple | 3 | 24 | 8.0705452e-05 | 0.13428239 | 0.00059525775 | 0/24 |
| ascent/look_at_obj_in_light | 3 | 16 | 8.1130582e-05 | 0.19107117 | 0.00034688003 | 0/16 |
| ascent/pick_clean_then_place_in_recep | 3 | 20 | 0.0002096979 | 0.18391267 | 0.0013721405 | 0/20 |
| ascent/pick_heat_then_place_in_recep | 3 | 19 | 0.0005228562 | 0.10957709 | 0.0032683714 | 0/19 |
| ascent/pick_cool_then_place_in_recep | 3 | 19 | 0.0002821624 | 0.1476705 | 0.0017606584 | 0/19 |
| ascent/pick_two_obj_and_place | 3 | 20 | 0.00026564719 | 0.21045363 | 0.00086722657 | 0/20 |
| ascent_icl/pick_and_place_simple | 1 | 5 | 0.00035920559 | 0.18753771 | 0.0017832711 | 0/5 |
| ascent_icl/look_at_obj_in_light | 1 | 5 | 1.6151316e-05 | 0.14606455 | 2.459697e-05 | 0/5 |
| ascent_icl/pick_clean_then_place_in_recep | 1 | 5 | 4.5872868e-06 | 0.058722072 | 1.5047215e-05 | 0/5 |

## Validation

- Samples: 23/23.
- Initial episode reconstructed KL: 0.107150846189; logged: 0.107150848024.
- Maximum per-turn calibration error: 9.493e-09.
- Null control: {'sample_id': 'ascent-task_type-task_blocked-seed42-20261104T025254Z/run1:0:1', 'max_hidden_absolute_difference': 0.0, 'identical_input_ids': True, 'passed': True}.
- Memory-bounded recurrence: {'output_max_absolute_error': 1.52587890625e-05, 'state_max_absolute_error': 1.1920928955078125e-07, 'reference_chunk_size': 64, 'test_segment_size': 128, 'inference_segment_size': 2048, 'output_absolute_tolerance': 3e-05, 'state_absolute_tolerance': 1e-05, 'passed': True}.
- Real-model original-versus-bounded validation: {'sample': 'ascent-task_type-task_blocked-seed42-20261104T025254Z/run1:49:22', 'original_all_kl': 0.06435259972955161, 'bounded_all_kl': 0.06435259972955161, 'all_kl_absolute_difference': 0.0, 'max_token_kl_absolute_difference': 0.0}.
- Peak CUDA allocator usage: 2.779 GiB (cap 4 GiB).

## Limits

These are conditional next-token divergences on successful student trajectories. The exact student reasoning before Action: is held fixed and may already specify the action. Small action KL therefore does not rule out policy changes mediated through different reasoning. The privileged record contains the successful answers themselves, so large reasoning KL can reflect copying. No claim about held-out success improvement, full command-distribution KL, or post-distillation student quality is made.

## Reproduce

Run from the repository root with the continual environment installed:

```bash
.venv/bin/python report/teacher_base_kl/probe.py
.venv/bin/python report/teacher_base_kl/summarize.py
```

The probe uses the committed successful episodes from the two run directories named in `samples()`.
Those trajectories and model weights are local inputs and are not included in Git.
Default weights: `../models/Qwen3.5-9B`; override with `ALFWORLD_MODEL_PATH`.
Default output: `runs/reports/teacher-base-kl-20261105`; override with `ALFWORLD_KL_OUTPUT`.
The probe skips sample IDs already present in that output directory. Use a fresh output directory for a fresh measurement.

GPU 0 is used by default (`--device` overrides it), with a 4 GiB PyTorch allocator cap,
CPU weight offload, and a startup check requiring at least 6 GiB free.
The bounded delta-rule recurrence is patched only in the diagnostic process.
No training process, installed package, checkpoint, or environment state is modified.

`summary.json`, `turn_metrics.csv`, and `metadata.json` are the measured 2026-11-05 snapshot.
Raw per-token output, traces, and logs remain in the ignored output directory.
