# HMRN reproduction

Independent paper-driven reproduction of the HMRN detector/tracker.

Implemented:

- paper-specific r1 AOI projection and moving tracked-point cache;
- 960×544 current/previous/previous-heatmap inputs;
- compact four-stage IDA backbone with center and displacement heads;
- CenterNet focal loss and center-masked displacement L1;
- 3×3 local-maximum decoding, feedback heatmap and greedy association;
- CUDA training with effective batch 16 through gradient accumulation;
- strict 10 px point matching and basic identity-switch accounting.

The exact DLA variant, output stride, Gaussian widths and association radius are
not disclosed by the paper. Frozen reproduction choices are in
`paper_spec.json`; full training is intentionally deferred.

Evaluation support:

- `calibrate_train.py` selects the center threshold on TRAIN frames 561--611
  over all six recoverable AOIs.
- `evaluate_self_test.py` evaluates one frozen SELF-TEST AOI with strict
  one-to-one point matching, localization error and identity switches.
- `evaluate_all_self_test.py` aggregates AOIs 01, 02, 03, 34, 40 and 41.
- No final SELF-TEST metrics may be reported until full training is complete
  and the threshold has been frozen on TRAIN.
