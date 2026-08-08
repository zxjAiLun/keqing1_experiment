# Mortal Training Runbook

Updated: 2026-05-09 (status note; original runbook 2026-04-28)

This runbook follows the local `third_party/Mortal` Git repository. Do not use
`xmodel`, `xmodel1`, or `keqingv4` checkpoints, logits, scores, or rollout
outputs as teacher sources.

## 2026-05-09 Status Note

This runbook remains the operational Mortal training reference, but its
KeqingRL teacher-probe and student-handoff sections are historical diagnostics.
The active strength route is no longer KeqingRL imitation. It is Mortal-native
continued training, fine-tuning, selfplay training, reward/GRP/rank-point
experimentation, and checkpoint evaluation using Mortal/libriichi encoding.

```text
libriichi state / Mortal obs_repr
-> Mortal Brain encoder
-> Mortal Dueling DQN action-value head
-> optional future policy / value / rank heads on the same backbone
```

The old best KeqingRL student checkpoint is retained only as archive evidence:

```text
reports/keqingrl_mortal_action_q_imitation_train_20260430_source93_step20000_allseats_lr003_cont1/checkpoint_config_000/policy_iter_0004.pt
teacher_kl=0.401925
teacher_agreement=0.686201
mapping=5422/5422
fail_closed=0
```

Do not read older discard-only, terminal-poor, or paired-proxy notes as the
current training gate. They explain how the project got here; they do not
override the 2026-05-09 Mortal-native mainline.

The current control document for this pivot is:

```text
docs/mortal/mainline_pivot_2026_05_09.md
```

## Repository Contract

Local Mortal is aligned to upstream `Equim-chan/Mortal` main at:

```text
0cff2b52982be5b1163aa9a62fb01f03ce91e0d2
```

The Python extension contract is:

```text
cargo build -p libriichi --lib --release
cp target/release/libriichi.so mortal/libriichi.so
python -c "import libriichi"
```

The rebuilt local module imports as `libriichi`; `third_party/Mortal` is clean.

## Local Environment Status

Current project venv has the Python-side utilities needed by the local finite
Mortal runners:

```text
torch: ok
toml: ok
tqdm: ok
tensorboard: ok
```

If setting up a fresh venv, install at least:

```bash
uv pip install toml tqdm tensorboard
```

Mortal's docs recommend using its own conda env plus a separately installed
PyTorch. Either path is fine as long as `mortal/libriichi.so` is importable from
`third_party/Mortal/mortal`.

## Data Contract

Mortal training scripts expect gzip-compressed mjai NDJSON files:

```text
*.json.gz
```

The local converted data currently is uncompressed:

```text
artifacts/converted_mjai/**/*.mjson
count: 105765
size: 3.9G
```

Passing `.mjson` directly to Mortal fails with `invalid gzip header`. A gzipped
sample was verified through Mortal's loaders:

```text
GameplayLoader(version=4): games=4, obs=(1012, 34), mask_len=46
Grp.load_gz_log_files: grp_feature_shape=(9, 7)
```

Prepare a Mortal dataset directory by gzipping without changing the source data:

```bash
uv run python scripts/prepare_mortal_training.py \
  --source-dir artifacts/converted_mjai \
  --output-dir artifacts/mortal_mjai_gz \
  --training-dir artifacts/mortal_training \
  --val-ratio 0.05 \
  --device cuda:0
```

The script writes:

```text
artifacts/mortal_mjai_gz/train/**/*.json.gz
artifacts/mortal_mjai_gz/val/**/*.json.gz
artifacts/mortal_training/config.toml
artifacts/mortal_training/manifest.json
```

## Training Order

Mortal has two offline stages.

### 1. Train GRP

`mortal/train_grp.py` trains `GRP`, which is later used by
`RewardCalculator` to build DQN rewards.

Relevant config sections:

```toml
[grp]
state_file = '/abs/path/artifacts/mortal_training/grp.pth'

[grp.control]
device = 'cuda:0'
tensorboard_dir = '/abs/path/artifacts/mortal_training/tb_grp'
batch_size = 512
save_every = 2000
val_steps = 400

[grp.dataset]
train_globs = ['/abs/path/artifacts/mortal_mjai_gz/train/**/*.json.gz']
val_globs = ['/abs/path/artifacts/mortal_mjai_gz/val/**/*.json.gz']
file_index = '/abs/path/artifacts/mortal_training/grp_file_index.pth'
file_batch_size = 50
```

Command:

```bash
cd third_party/Mortal/mortal
MORTAL_CFG=/abs/path/artifacts/mortal_training/config.toml python train_grp.py
```

The upstream script loops forever. The project-side finite runner is preferred
for reproducible chunks:

```bash
uv run python scripts/run_mortal_grp_offline.py \
  --config artifacts/mortal_training/config.toml \
  --target-steps 2000 \
  --val-steps 50 \
  --num-workers 0
```

### 2. Train Mortal Brain + DQN

`mortal/train.py` trains:

```text
Brain(version=4)
DQN(version=4)
AuxNet
```

Relevant config sections:

```toml
[control]
version = 4
online = false
state_file = '/abs/path/artifacts/mortal_training/mortal.pth'
best_state_file = '/abs/path/artifacts/mortal_training/mortal_best.pth'
tensorboard_dir = '/abs/path/artifacts/mortal_training/tb_mortal'
device = 'cuda:0'
batch_size = 512
save_every = 400
test_every = 20000

[dataset]
globs = ['/abs/path/artifacts/mortal_mjai_gz/train/**/*.json.gz']
file_index = '/abs/path/artifacts/mortal_training/file_index.pth'
file_batch_size = 15
num_workers = 1
player_names_files = []
num_epochs = 1
enable_augmentation = false

[grp]
state_file = '/abs/path/artifacts/mortal_training/grp.pth'
```

Command:

```bash
cd third_party/Mortal/mortal
MORTAL_CFG=/abs/path/artifacts/mortal_training/config.toml python train.py
```

For finite bootstrap chunks, use the project-side runner:

```bash
uv run python scripts/run_mortal_dqn_offline.py \
  --config artifacts/mortal_training/config.toml \
  --target-steps 400 \
  --num-workers 0
```

For unattended continued training, start the same runner in a detached session
from the repository root. Set `TARGET_STEPS` to the desired absolute checkpoint
step, not the number of additional steps:

```bash
TARGET_STEPS=80000
LOG="logs/mortal_dqn_to_${TARGET_STEPS}_$(date +%Y%m%d%H%M%S).log"
mkdir -p logs
setsid bash -lc "cd /mnt/e/AUbuntuProject/project/keqing1; \
  export TMPDIR=/tmp; \
  echo \"wrapper start \$(date)\"; \
  uv run python scripts/run_mortal_dqn_offline.py \
    --config artifacts/mortal_training/config.toml \
    --target-steps ${TARGET_STEPS} \
    --num-workers 1 \
    --log-every 50; \
  rc=\$?; echo \"wrapper exit \$rc \$(date)\"; exit \$rc" \
  > "$LOG" 2>&1 < /dev/null &
echo "pid=$!"
echo "log=$LOG"
```

Use `TMPDIR=/tmp` for detached runs on this WSL workspace; otherwise PyTorch
DataLoader workers can fail while creating multiprocessing sockets on the
mounted project drive. To check progress later:

```bash
tail -f "$LOG"
pgrep -af 'run_mortal_dqn_offline|python.*mortal'
```

It saves the same checkpoint keys needed by the KeqingRL teacher runtime:

```text
state['mortal']
state['current_dqn']
state['aux_net']
state['config']
```

## Offline Wrapper Note

Upstream `train.py` constructs `TestPlayer()` immediately, before the first
offline batch. `TestPlayer` loads:

```toml
[baseline.test]
state_file = '/path/to/baseline.pth'
```

So first-from-scratch offline training needs either:

- an existing trained Mortal-format baseline checkpoint for `baseline.test`, or
- a small local patch that instantiates `TestPlayer` lazily only when evaluation
  actually runs.

For this project, the baseline checkpoint must also be Mortal-format. Do not use
`xmodel`, `xmodel1`, or `keqingv4` as baseline or teacher artifacts.

The project-side wrapper avoids editing `third_party/Mortal` by monkeypatching
`player.TestPlayer` into a lazy proxy before importing `mortal/train.py`:

```bash
uv run python scripts/run_mortal_train_offline.py \
  --config artifacts/mortal_training/config.toml
```

This lets offline training start without loading `baseline.test.state_file`
until the first configured `test_every` evaluation. Evaluation still needs a
Mortal-format baseline checkpoint once it actually runs.

## Output Needed By KeqingRL

After training, the checkpoint should contain:

```text
state['mortal']       -> Brain state_dict
state['current_dqn']  -> DQN state_dict
state['config']       -> Mortal config
```

KeqingRL should load those into Mortal `Brain + DQN`, call
`MortalEngine.react_batch(obs, masks, invisible_obs)`, and attach the returned
tensors to `PolicyInput.obs.extras`:

```text
mortal_q_values     shape [B, 46]
mortal_action_mask  shape [B, 46]
```

KeqingRL currently supports two Mortal teacher sources:

```text
teacher_source=mortal-discard-q  historical discard-only diagnostic
teacher_source=mortal-action-q   46-id Mortal scorer over KeqingRL legal ActionSpec rows
```

`mortal-action-q` still does not let Mortal own legality or dispatch. KeqingCore
enumerates legal actions, KeqingRL preserves `ActionSpec` identity/order, and
Mortal only supplies Q/ranking over those legal candidates.

The checkpoint runtime wrapper now lives at:

```text
src/keqingrl/mortal_runtime.py
```

It exposes `load_mortal_teacher_runtime(checkpoint_path, mortal_root=...)` and
returns a runtime whose `evaluate(obs, masks)` method produces the standard
extras keys above.

The live observation bridge now lives at:

```text
src/keqingrl/mortal_observation.py
```

It replays KeqingRL mjai events into Mortal `PlayerState.encode_obs(version=4)`
and returns Mortal-format `obs` plus the 46-wide action mask.
`DiscardOnlyMahjongEnv` accepts `mortal_teacher_runtime` and
`mortal_observation_bridge`; on controlled learner turns it attaches
`mortal_q_values` and `mortal_action_mask` to `PolicyInput.obs.extras` when
the Mortal mask is compatible with the KeqingRL legal row. The current
integration has tests for observation parity, action mapping, runtime loading,
extras surviving selfplay collection, and PPO batch collation.

## KeqingRL Pilot Probe

### Deprecated Probe Interpretation

The following older discard-only probe is retained only as a contract smoke:

```bash
uv run python scripts/run_keqingrl_tempered_ratio_pilot.py \
  --candidate-summary artifacts/.../summary.csv \
  --source-config-ids 93 \
  --output-dir artifacts/.../mortal_discard_q_probe \
  --topk-ranking-aux-mode teacher-ce \
  --topk-ranking-aux-coef 0.05 \
  --topk-ranking-k 3 \
  --teacher-source mortal-discard-q \
  --teacher-temperature 1.0 \
  --mortal-teacher-checkpoint artifacts/mortal_training/mortal.pth \
  --mortal-root third_party/Mortal \
  --support-policy-mode support-only-topk \
  --delta-support-topk 3 \
  --rule-score-scales 0.25
```

Do not use that command to claim Mortal is weak or strong. It only tests whether
Mortal q/mask extras can be consumed for discard topK CE. It does not cover
reach, terminal, pass/call, or kan decisions.

The previous reading:

```text
"discard-only Mortal teacher did not pass, so Mortal teacher has little value"
```

is now marked wrong. It was missing the key precondition: the rollout/batch must
cover the relevant terminal/action opportunities before any teacher/topK
conclusion is qualified.

### Current Probe Contract

Current Mortal teacher probes should use `mortal-action-q` with an
opportunity-based terminal coverage gate:

```bash
uv run python scripts/run_keqingrl_tempered_ratio_pilot.py \
  --candidate-summary artifacts/.../summary.csv \
  --source-config-ids 93 \
  --output-dir reports/.../mortal_action_q_contract_probe \
  --topk-ranking-aux-mode teacher-ce \
  --topk-ranking-aux-coef 1.0 \
  --topk-ranking-k 3 \
  --teacher-source mortal-action-q \
  --teacher-temperature 1.0 \
  --mortal-teacher-checkpoint artifacts/mortal_training/mortal.pth \
  --mortal-root third_party/Mortal \
  --support-policy-mode support-only-topk \
  --delta-support-mode topk \
  --delta-support-topk 3 \
  --actor-update-support-mode topk \
  --actor-update-topk 3 \
  --rule-score-scales 0.25 \
  --self-turn-action-types DISCARD REACH_DISCARD TSUMO RYUKYOKU \
  --response-action-types PASS RON PON CHI \
  --forced-autopilot-action-types TSUMO RON RYUKYOKU \
  --no-mortal-teacher-strict-extra-mask \
  --terminal-coverage-gate \
  --terminal-coverage-min-legal-terminal-rows 1 \
  --terminal-coverage-min-legal-agari-rows 1
```

The pilot now writes the main contract artifacts:

```text
contract_scoreboard.csv
contract_scoreboard.md
mortal_action_mapping_audit.csv
mortal_action_mapping_examples.jsonl
```

`contract_scoreboard.*` is the go/no-go surface for Mortal action-Q probes. It
records legal ownership, action identity status, Mortal mapping coverage,
opportunity gate status, and diagnostic-only outcome counters. A config is not
qualified for paired eval if `mortal_action_q_fail_closed_count > 0`.

Because KAN family decisions are deliberately outside this response-stage
learner scope, Mortal id `42` can appear as an extra masked id when KeqingRL is
controlling only `PASS/RON/PON/CHI`. With
`--no-mortal-teacher-strict-extra-mask`, that out-of-scope extra is recorded in
the audit artifacts but does not fail the contract. Missing controlled legal
actions, unsupported actions, and ambiguous KAN mappings still fail closed.

`mortal_action_mapping_audit.*` captures response-window mask parity gaps such
as missing `PASS`, `RON`, `PON`, or `CHI` source ids. These rows are diagnostic
artifacts only; they do not provide fallback teacher probabilities and do not
relax fail-closed training behavior.

The terminal coverage gate is opportunity-based by default:

```text
terminal_coverage_legal_terminal_row_count
terminal_coverage_legal_agari_row_count
terminal_coverage_prepared_legal_terminal_row_count
terminal_coverage_prepared_legal_agari_row_count
```

Outcome counters are diagnostics only by default:

```text
terminal_coverage_score_changed_episode_count
terminal_coverage_score_changed_without_selected_agari_episode_count
terminal_coverage_selected_agari_count
```

Do not gate on actual `hora` count or `score_changed` unless the special flag is
explicitly enabled:

```text
--terminal-coverage-outcome-gate
```

Rationale: actual hora is a stochastic outcome affected by wall luck and seat
rotation. `score_changed` is also not an agari proxy because riichi sticks and
ryukyoku tenpai payments can change scores without any `hora`.

The pilot fails closed if `teacher_source=mortal-discard-q` or
`teacher_source=mortal-action-q` is selected without
`--mortal-teacher-checkpoint`. Non-Mortal diagnostic controls do not load a
Mortal runtime.

### Replay Before Reinterpreting Results

When a probe appears to be "all ryukyoku", "no ron", or "teacher did not move",
export the exact seed before updating conclusions:

```bash
uv run python scripts/export_keqingrl_mjai_replay.py \
  --candidate-summary artifacts/.../summary.csv \
  --source-config-ids 93 \
  --output-dir reports/.../replays \
  --episode-index 0 \
  --seed-base 202604300000 \
  --torch-seed 202604300000 \
  --self-turn-action-types DISCARD REACH_DISCARD TSUMO RYUKYOKU \
  --response-action-types \
  --forced-autopilot-action-types TSUMO RON RYUKYOKU
```

This writes:

```text
episode_*.mjai.jsonl
episode_*.readable.md
episode_*.decisions.csv
```

Use the readable replay to distinguish:

```text
true hora
ryukyoku
riichi-stick score changes
legal terminal/agari opportunities
selected terminal/agari actions
```

### Current Full-Action Gap

Full response-window teacher is not yet cleared for training. A direct full
scope can fail closed when KeqingCore legal actions include response choices
such as `PON + PASS` but Mortal's mask does not mark the corresponding source
ids. This is a contract gap to investigate, not evidence that Mortal is weak.
Do not paper over missing legal keys with silent fallback.

When this happens, inspect `mortal_action_mapping_examples.jsonl` first. Each
row includes legal action keys/types, Mortal mask ids, missing/extra ids,
event tail, hand/last-discard context when available, and a replay hint for
`scripts/export_keqingrl_mjai_replay.py`.

## Local Artifact Status

As of 2026-04-28, the local workspace has completed the first bootstrap pass:

```text
artifacts/mortal_mjai_gz/**/*.json.gz      105765 files, 745M
artifacts/mortal_training/grp.pth          steps=2000
artifacts/mortal_training/mortal_step400.pth  steps=400
artifacts/mortal_training/mortal_step2000.pth steps=2000
artifacts/mortal_training/mortal_step5000.pth steps=5000
artifacts/mortal_training/mortal_step10000.pth steps=10000
artifacts/mortal_training/mortal_step20000.pth steps=20000
artifacts/mortal_training/mortal.pth           steps=20000
```

The current `mortal.pth` is a trained Mortal-format teacher artifact and is
useful for integration probes. It is not yet validated as a strength teacher;
only report it as a candidate after the KeqingRL train/fresh/movement/paired
gates pass. Previous step-400, step-2000, step-5000, step-10000, and
step-20000 checkpoints are preserved for reproducibility.

Runtime smoke already passed:

```text
load_mortal_teacher_runtime(artifacts/mortal_training/mortal.pth)
env.observe(actor).obs.extras['mortal_q_values']     -> shape [1, 46]
env.observe(actor).obs.extras['mortal_action_mask']  -> shape [1, 46]
mortal-discard-q topK teacher context                -> OK
```

The step-10000 runtime smoke on `seed=1` produced a valid discard topK
distribution:

```text
teacher_topk_scores = [-3.5802, -5.9136, -5.9463]
teacher_probs       = [0.8398, 0.0814, 0.0788]
```

The step-20000 runtime smoke on `seed=1` produced a valid discard topK
distribution:

```text
teacher_topk_scores = [-4.6350, -7.2691, -7.2102]
teacher_probs       = [0.8711, 0.0625, 0.0663]
```

Mortal-only KeqingRL probes have been run under:

```text
reports/keqingrl_mortal_discard_q_smoke_20260428_source93_step400
reports/keqingrl_mortal_discard_q_screen_20260428_source93_step400
reports/keqingrl_mortal_discard_q_aggressive_movement_20260428_source93_step400
reports/keqingrl_mortal_discard_q_moderate_grid_20260428_source93_step400
reports/keqingrl_mortal_discard_q_narrow_grid_20260428_source93_step400
reports/keqingrl_mortal_discard_q_narrow_grid_20260428_source93_step2000
reports/keqingrl_mortal_discard_q_penalty_grid_20260428_source93_step2000
reports/keqingrl_mortal_discard_q_narrow_gate_20260428_source93_step10000
reports/keqingrl_mortal_discard_q_penalty_gate_20260428_source93_step10000
reports/keqingrl_mortal_discard_q_narrow_gate_20260428_source93_step20000
reports/keqingrl_mortal_discard_q_penalty_gate_20260428_source93_step20000
```

Observed status:

```text
integration smoke: passed
Mortal q/mask extras: passed
mortal-discard-q teacher CE: passed
moderate movement: possible
train/fresh/movement gate: not passed yet
```

Important correction: the older "train/fresh/movement gate not passed" lines
above describe discard-only or terminal-coverage-poor diagnostics. They must
not be read as a strength judgment on Mortal. Those probes did not cover enough
reach / terminal / response decision opportunities to answer whether Mortal is
a useful Mahjong teacher.

The useful diagnostic facts are:

- Conservative settings (`coef<=1`, `teacher_temperature=1.0`) preserve low
  KL/clip but do not move top1.
- Aggressive settings (`coef=10`, `teacher_temperature=0.1`, high lr/epochs)
  prove the teacher can move topK ordering, but they overmove.
- The best narrow step-400 probe found `top1_changed=0.0556`, `t_kl=0.0123`,
  `t_clip=0.125`, but movement quality failed because changes were concentrated
  on high prior-margin decisions (`changed_prior_margin_p50 ~= 3.0`).
- Step-2000 did not remove this gate issue.
- Weak-margin flip penalty suppresses movement when strong enough and does not
  yet produce a clean pass.
- Step-10000 preserves the same failure shape. The narrow gate found no pass:
  `lr=0.0023` did not move top1, `lr=0.0025` moved top1 by `0.0952` but failed
  movement quality and fresh validation (`fresh_top1=0.4194`,
  `changed_prior_margin_p50=3.0`, `t_clip=0.3571`), and `lr=0.0027` was below
  the train movement threshold while still failing fresh quality.
- Step-10000 weak-margin penalties also found no pass. `coef=0.01` and
  `coef=0.1` suppressed top1 movement to zero; `coef=0.05` reproduced the
  overmoving/high-margin failure.
- Step-20000 still found no pass. The narrow gate again had no-move configs at
  `lr=0.0023` and `lr=0.0027`; `lr=0.0025` moved top1 by `0.0952` but failed
  movement quality and fresh validation (`fresh_top1=0.3548`,
  `changed_prior_margin_p50=3.001`, `t_clip=0.3810`).
- Step-20000 weak-margin penalties also found no pass. `coef=0.01` and
  `coef=0.1` suppressed top1 movement to zero; `coef=0.05` reproduced the
  overmoving/high-margin failure (`fresh_top1=0.3548`, `t_clip=0.3810`).

These are now marked as **discard-teacher consumption diagnostics**, not
teacher-strength results. The 2026-04-29 conclusion was:

```text
Mortal checkpoint: trained and usable as a teacher artifact.
Mortal as strength teacher: not yet validated.
Discard-only no-pass: unqualified as strength evidence.
Full response-window teacher: contract path now passes for the scoped
PASS/RON/PON/CHI response stage.
Next valid probe: learning-signal work that produces nonzero top1 movement
without leaving the now-passing Mortal action-Q contract.
```

At that stage, the Mortal checkpoint was trained and usable as a teacher
artifact but not yet validated as a strength source. The 2026-04-29
`bridge_fix` probe passed the contract and opportunity gates, but the paired
seat-rotation proxy remained rule-prior equivalent:

```text
reports/keqingrl_mortal_action_q_contract_probe_20260429_source93_step20000_bridge_fix_ckpt
reports/keqingrl_mortal_action_q_paired_eval_20260429_source93_step20000_bridge_fix

contract: mapping_available_rate=1.0, fail_closed=0, opportunity_gate=true
paired proxy: delta_vs_zero=0, top1_changed=0
```

The useful conclusion is narrow: the Mortal action-Q plumbing is now
interpretable for the staged response scope, but one conservative PPO update did
not change top-1 behavior in paired eval.
