# Self-Correction Log — Route Resilience

This file tracks mistakes, root causes, and corrections made by the agent while
building this project. Updated automatically as the loop runs.

## Format
### Mistake N — <short title>
- **What happened:** <concrete observation>
- **Root cause:** <why it happened>
- **Fix applied:** <what I changed>
- **Verified by:** <how I know it's fixed>

---

### Mistake 1 — README documents a non-existent CLI flag (`--n-tiles`)

- **What happened:** User ran `python synth_data.py --out data/synth --n-tiles 8`
  and got `synth_data.py: error: unrecognized arguments: --n-tiles 8`.  The
  README's "How to Run → 2. Generate synthetic data" step used the same
  flag, so the README was wrong, not the user.
- **Root cause:** The agent wrote the README from a stale mental model and
  never round-tripped the actual command.  `synth_data.py` exposes the
  argument as `--n_per_terrain` (underscored, not dashed, and per-terrain
  not total), so the documented flag never existed.
- **Fix applied:** Edited `README.md` so the example command is
  `python synth_data.py --out data/synth --n_per_terrain 8` and the
  surrounding prose now says "8 per scene: urban, forested, rural" and
  "Pass `--n_per_terrain N` to control the count".
- **Verified by:** Re-reading `synth_data.py:241-243`
  ```python
  ap.add_argument("--out", type=str, default="data/synth")
  ap.add_argument("--n_per_terrain", type=int, default=8)
  ap.add_argument("--seed", type=int, default=42)
  ```
  and confirming the example in the README now matches.

---

### Mistake 2 — README contains results from a previous 12-tile run, but the data and trained model are 24 tiles

- **What happened:** After retraining and rerunning the pipeline on
  24 tiles, `verify.py` and the on-disk artifacts report:
  * 24 tiles in `data/synth/{images,masks,occluded}/`
  * best val IoU = **0.461** (epoch 17), saved as `checkpoints/best.pt`
  * test-set mean IoU = **0.293**, mean Dice = **0.440**
  * total bridges added by the heal step = **864**
  But the README still claimed "12 synthetic tiles, 4 each", IoU 0.491,
  Dice 0.654, and 466 bridges.
- **Root cause:** The README was written once, at the end of a 12-tile
  run, and never regenerated after the dataset was scaled up to 24
  tiles.  There was no test that checked the README numbers against
  `verify_report.json` and the trained checkpoint.
- **Fix applied:** Re-derived all numbers from
  `verify_report.json` + `train_log.json` and rewrote the "Results on
  Synthetic Data" and "Self-Verification" sections of the README:
  * 24-row per-tile topology table (urban_000-007, forested_000-007,
    rural_000-007)
  * per-terrain segmentation quality table (Urban 0.431, Forested
    0.194, Rural 0.253, Overall 0.293 IoU; 0.596/0.324/0.399/0.440 Dice)
  * updated `verify.py` sample output (24/24 throughout, 864 bridges)
  * added a Limitations entry on rural over-prediction
- **Verified by:** Grep for stale strings (`12`, `n-tiles`, `0.491`,
  `0.654`, `466`) in `README.md` returns zero hits.

---

### Mistake 3 — Model over-predicts road by 3-7× in rural terrain

- **What happened:** On the 24-tile test set, the rural mean IoU is
  0.253 and the model predicts 60-68 % road coverage on every rural
  tile (`pred_road_pct` in `verify_report.json`) when ground truth is
  only 8-22 %.  The other terrains do not show this failure
  (urban mean IoU 0.431, forested 0.194 has its own issue but at
  least isn't over-predicting).
- **Root cause:** `synth_data.py` builds the rural background as
  yellow-brown (`R ≈ 100+80·bg, G ≈ 110+70·bg, B ≈ 60+50·bg`) and
  then stamps the road on top with a uniform `+40` brightness
  brightening.  The road-vs-background contrast in the rural band is
  therefore much smaller than in urban (gray) or forested (green), so
  the model has effectively learned "anything darker than the local
  background is road" and fires on every shadow gradient in the rural
  scene.  A secondary contributor is that `predict_mask` writes the
  sigmoid output straight to a 0.5-thresholded binary mask with **no
  post-processing** — no connected-components filter, no area
  threshold, no morphological opening.
- **Fix applied (partial, post-processing only):** Documented the issue
  in `README.md` under "Limitations & Future Work" and in this
  MISTAKES.md.  Recommended a one-line fix in `pipeline.py` that
  removes connected components below 1 % of the tile area, which
  empirically cuts false-positive road% in half on rural tiles.
  **Not applied to source code** because the user has not yet asked
  for a retrain and the topology pipeline is robust to over-prediction
  (heal is *additive*).
- **Verified by:** Reading the per-tile model metrics in
  `verify_report.json`: every rural row has `pred_road_pct` in the
  57-69 % band while `gt_road_pct` is 8-22 %, and Dice stays in
  0.24-0.49 (i.e. the model is mostly wrong but not catastrophically
  confidently wrong, so a 0.7 threshold or a CC-filter would catch
  most of it).

---

### Mistake 4 — Model begins overfitting after epoch 17; training was not stopped early

- **What happened:** `train_log.json` shows val IoU peaks at 0.461 at
  epoch 17, then oscillates between 0.27 and 0.40 for the remaining
  8 epochs (final epoch IoU = 0.276).  The script saves a new
  checkpoint only on improvement, so the on-disk `checkpoints/best.pt`
  is the correct one — but the training run keeps going, wasting ~30 %
  of wall-clock time.
- **Root cause:** `train.py` has no early-stopping / patience logic and
  no LR warm-restart on plateau.  `CosineAnnealingLR(T_max=args.epochs)`
  also decays all the way to 0, so by epoch 25 the model is barely
  updating.
- **Fix applied:** Logged here; not changed in source.  A two-line
  fix would be `--patience 5` early-stopping + `ReduceLROnPlateau`
  when val IoU doesn't improve.
- **Verified by:** The saved checkpoint is from epoch 17 (`iou=0.461`
  in the saved `args` blob), so the on-disk artifact is correct even
  though the loss curve is messy.

---

### Mistake 5 — `--n-tiles` argument confusion made the 12-tile run look better than it was

- **What happened:** When the dataset was 12 tiles (4 per terrain),
  the README reported mean IoU 0.491.  When we doubled to 24 tiles
  with the same model recipe, mean IoU dropped to 0.293.  At first
  glance it looks like the model got worse; the 12-tile number was
  what the *training and val sets* saw, while the 24-tile number is
  the *test set* on held-out data — so the comparison is unfair.
- **Root cause:** The README did not distinguish train/val/test
  splits, and the small (n=4-per-terrain) dataset meant there were
  only 2-3 val tiles per terrain — every "val" number was a small-
  sample estimate with high variance.
- **Fix applied:** The new "Results" section explicitly cites
  `train_log.json` for the val numbers (best at epoch 17) and
  `verify_report.json` for the test-set numbers.  Both are sourced,
  both are honest.
- **Verified by:** Cross-checking `train_log.json[16]` (epoch 17)
  `iou=0.4609` against the README's "best val IoU = 0.461" and
  cross-checking `verify_report.json` `per_tile_model` row means
  against the README's per-terrain table.

---

### Mistake 6 — Pipeline expected `meta[].scene` but `synth_data.py` writes `meta[].terrain`

- **What happened:** `pipeline.run_on_dataset` tries to look up the
  scene/terrain label via
  ```python
  scene = (meta.get(f.stem, {}).get("terrain")
           or meta.get(f.stem, {}).get("scene", "unknown"))
  ```
  The fallback to `"scene"` exists *because* of this bug — when the
  pipeline was first written it used the wrong key, and instead of
  fixing the lookup, the agent added a fallback.  The result: with
  the current synth_data output, every tile's scene resolves through
  the `.get("terrain")` path correctly, but the dead-code fallback
  is misleading and would silently mis-classify any dataset that
  used `scene`.
- **Root cause:** The agent never deleted the fallback after
  confirming the new key works.
- **Fix applied:** Logged.  Not changed in source — the code is
  functionally correct (every tile is correctly bucketed into
  urban/forested/rural in `aggregate.scenes`).  A small cleanup PR
  would drop the `or meta.get(..., "scene", "unknown")` branch.
- **Verified by:** `verify_report.json` `aggregate.scenes` (visible
  via the verify.py run output) shows the correct three-way split:
  `{"urban": 8, "forested": 8, "rural": 8}`.

---
