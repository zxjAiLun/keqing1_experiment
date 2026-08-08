# Mortal Discard Teacher Contract

Updated: 2026-04-28

## Status: Deprecated Diagnostic Scope

This document is historical context for the first `mortal-discard-q` adapter.
It is no longer the active Mortal teacher plan.

Active documents:

```text
docs/mortal_action_contract.md
docs/keqingrl/mortal_training_workflow.md
plans/mortal_training_runbook_2026_04_28.md
```

The corrected 2026-04-28 understanding is:

```text
WRONG: discard-only / terminal-poor no-move probes prove Mortal teacher weakness.
RIGHT: those probes are unqualified unless the batch covers the terminal/action
       opportunities being tested.

WRONG: score_changed is agari coverage.
RIGHT: riichi sticks and ryukyoku tenpai payments can change scores without hora.

WRONG: actual hora / selected agari should be a default gate.
RIGHT: actual hora is outcome luck; default gates must use legal opportunity rows.

WRONG: Mortal should only be used as discard-Q topK teacher.
RIGHT: discard-Q is only a contract diagnostic. Strength-relevant teacher use
       must move to Mortal action-Q over KeqingRL legal ActionSpec rows, with
       KeqingCore/Rust still owning legal actions.
```

Do not use this file to justify a teacher-strength conclusion. Use it only to
understand the first discard-only bridge and its fail-closed parity checks.

## Scope

Mortal is the only allowed teacher source for KeqingRL's teacher experiments.
`xmodel`, `xmodel1`, and `keqingv4` checkpoints or outputs must not be used as
teacher artifacts. The first Mortal integration was intentionally discard-only:

- use Mortal `q_values` and mask only for discard tile ids
- map to KeqingRL `ActionSpec(ActionType.DISCARD, tile=...)`
- consume the mapped scores only inside rule-prior topK, initially `topK=3`
- keep Rust legal enumeration and KeqingRL rollout contracts authoritative

This "do not connect" rule is now superseded for the active plan. The current
route is to connect Mortal action-Q progressively over KeqingRL legal actions,
while failing closed on mask/key mismatches. Full response-window
PASS/RON/PON/CHI training remains blocked until mask parity gaps are understood.

## Mortal Runtime Contract

Relevant files in `third_party/Mortal`:

- `mortal/engine.py`: `MortalEngine.react_batch(obs, masks, invisible_obs)` returns `(actions, q_values, masks, is_greedy)`.
- `mortal/model.py`: `Brain(version=4)` encodes `(1012, 34)` observations; `DQN(version=4)` returns masked Q values over `ACTION_SPACE`.
- `libriichi/src/consts.rs`: `ACTION_SPACE = 46`, `obs_shape(4) = (1012, 34)`.
- `libriichi/src/agent/mortal.rs`: action ids are converted to mjai events.
- `libriichi/src/dataset/gameplay.rs`: offline training labels use the same action id layout.

Mortal action ids:

```text
0..36  discard tile / kan tile choice
37     reach
38     chi low
39     chi mid
40     chi high
41     pon
42     kan decision
43     agari
44     ryukyoku
45     pass
```

Discard tile ids `0..36` are:

```text
0..8    1m..9m
9..17   1p..9p
18..26  1s..9s
27..33  E,S,W,N,P,F,C
34..36  5mr,5pr,5sr
```

`q_values` are DQN Q outputs trained against Mortal's placement/rank reward pipeline. They are useful as relative scores inside a decision but should not be treated as calibrated logits or final policy logits.

## KeqingRL Mapping

KeqingRL `ActionSpec` uses normalized tile34 ids. It does not distinguish red and normal five in `ActionSpec.DISCARD`.

The first adapter therefore collapses Mortal red-five discard ids:

```text
Mortal 4  and 34 -> Keqing tile 5m
Mortal 13 and 35 -> Keqing tile 5p
Mortal 22 and 36 -> Keqing tile 5s
```

When both normal and red ids are legal in Mortal's mask, the adapter uses the max Q value for the normalized Keqing action. This is a contract limitation of the current KeqingRL discard action identity. If KeqingRL later exposes physical red-five discard identity, this contract must bump.

`tsumogiri` is not a separate Mortal action id. Mortal derives it when converting a selected tile id into a `dahai` event. The discard teacher scores therefore ignore `ACTION_FLAG_TSUMOGIRI`; the flag remains part of KeqingRL dispatch identity.

## Observation Bridge

The first live bridge replays KeqingRL's mjai event log into Mortal's
`libriichi.state.PlayerState` and calls:

```text
PlayerState(actor).encode_obs(version=4, at_kan_select=false)
```

This returns:

```text
obs         float tensor-compatible array, shape [1012, 34]
action_mask bool tensor-compatible array,  shape [46]
```

The implementation lives in `src/keqingrl/mortal_observation.py`. It sanitizes
only KeqingRL-local extension events that Mortal does not understand; currently
`kakan_accepted` is skipped because Mortal has already consumed the preceding
`kakan` event. Known mjai events are otherwise replayed unchanged.

`DiscardOnlyMahjongEnv` can now be constructed with:

```python
DiscardOnlyMahjongEnv(
    mortal_teacher_runtime=runtime,
    mortal_observation_bridge=bridge,
)
```

On learner-controlled discard-only turns, `env.observe(actor)` encodes Mortal
obs/mask, asserts discard mask parity, evaluates the trained Mortal runtime, and
attaches standard extras to the returned `PolicyInput`.

## Parity Requirements

For every learner-controlled discard decision:

```text
collapsed Mortal discard mask tile ids == KeqingRL legal DISCARD tile ids
```

Failure modes must fail closed:

- Keqing legal discard is missing from Mortal mask
- Mortal mask exposes an extra collapsed discard tile
- topK contains an action without a Mortal discard score
- legal actions contain non-discard actions in a discard-only teacher path

The implementation lives in `src/keqingrl/mortal_teacher.py`; tests live in `tests/test_mortal_teacher_mapping.py`.
Bridge and selfplay collation tests live in `tests/test_mortal_observation_bridge.py`.

## Historical TopK Teacher Use

The historical discard-only diagnostic signal was:

```text
topK_indices = topK(rule_prior_logits, K=3)
mortal_scores_topK = mapped_mortal_q_values[topK_indices]
teacher_probs_topK = softmax(mortal_scores_topK / teacher_temperature)
loss = CE(policy_logits_topK, teacher_probs_topK)
```

This was combined with the existing support-only topK policy constraint:

```text
support-onlyTopK=3
rule_score_scale=0.25
teacher_source=mortal-discard-q
```

The KeqingRL pilot hook expects Mortal runtime output to be carried through
`PolicyInput.obs.extras` with these keys:

```text
mortal_q_values    float tensor, shape [B, 46]
mortal_action_mask bool tensor,  shape [B, 46]
```

If either tensor is missing, has the wrong shape, or fails discard-mask parity,
`mortal-discard-q` fails closed instead of falling back to another teacher.

Diagnostic controls may remain only for ablation and sanity checks; they must
not be reported as teacher results:

- no topK teacher
- rule-prior-topK teacher
- rule-component-v1 teacher

## Local Mortal Build Note

The local `third_party/Mortal` worktree was reconciled to the Git repository
contract on 2026-04-28:

```text
libriichi/src/lib.rs exports PyInit_libriichi
third_party/Mortal/mortal/libriichi.so imports as libriichi
```

`third_party/Mortal` is clean after rebuilding `cargo build -p libriichi --lib
--release` and copying `target/release/libriichi.so` into `mortal/libriichi.so`.
