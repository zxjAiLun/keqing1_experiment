#!/usr/bin/env python3
"""Summarize raw hanchan outcomes for one trainable replay view.

This complements ``audit_replay_distribution.py``.  GameplayLoader exposes
decision samples and Q support metrics, while this pass reads the native MJAI
events directly to report agari/houjuu/fuuro/riichi counts for the selected
model seat.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_files(index_path: Path) -> list[Path]:
    payload = torch.load(index_path.resolve(), weights_only=False, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload.get("file_list")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"file index has no file_list: {index_path}")
    files: list[Path] = []
    for value in payload:
        path = Path(str(value))
        path = path if path.is_absolute() else REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path.resolve())
    return files


def count_events(events: list[dict[str, Any]], player_id: int) -> dict[str, int]:
    counts = {
        "kyoku": 0,
        "agari": 0,
        "houjuu": 0,
        "fuuro": 0,
        "riichi": 0,
        "ryukyoku": 0,
        "dahai": 0,
        "tsumo": 0,
        "reach_accepted": 0,
    }
    for event in events:
        event_type = str(event.get("type"))
        actor = event.get("actor")
        actor_is_player = actor == player_id
        if event_type == "start_kyoku":
            counts["kyoku"] += 1
        elif event_type == "hora":
            if actor_is_player:
                counts["agari"] += 1
            if event.get("target") == player_id:
                counts["houjuu"] += 1
        elif event_type == "ryukyoku":
            counts["ryukyoku"] += 1
        elif event_type == "reach" and actor_is_player:
            counts["riichi"] += 1
        elif event_type == "reach_accepted" and actor_is_player:
            counts["reach_accepted"] += 1
        elif event_type in {"chi", "pon", "kakan", "ankan", "daiminkan"} and actor_is_player:
            counts["fuuro"] += 1
        elif event_type == "dahai" and actor_is_player:
            counts["dahai"] += 1
        elif event_type == "tsumo" and actor_is_player:
            counts["tsumo"] += 1
    return counts


def final_scores(events: list[dict[str, Any]]) -> list[float] | None:
    """Reconstruct final scores because native logs close with an empty end_game."""

    scores: list[float] | None = None
    for event in events:
        if event.get("type") == "start_kyoku" and isinstance(event.get("scores"), list):
            values = event["scores"]
            if len(values) == 4:
                scores = [float(value) for value in values]
        elif event.get("type") in {"hora", "ryukyoku"} and scores is not None:
            deltas = event.get("deltas")
            if isinstance(deltas, list) and len(deltas) == 4:
                scores = [score + float(delta) for score, delta in zip(scores, deltas, strict=True)]
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")

    files = load_files(args.file_index)
    total = {key: 0 for key in ("kyoku", "agari", "houjuu", "fuuro", "riichi", "ryukyoku", "dahai", "tsumo", "reach_accepted")}
    hanchans_with = {key: 0 for key in ("agari", "houjuu", "fuuro", "riichi")}
    ranks = {str(rank): 0 for rank in range(1, 5)}
    malformed: list[dict[str, str]] = []
    decisions = 0

    for index, path in enumerate(files, start=1):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            if not events or events[0].get("type") != "start_game":
                raise ValueError("missing start_game")
            names = events[0].get("names")
            if not isinstance(names, list) or args.model_label not in names:
                raise ValueError(f"model label {args.model_label!r} not found in names={names!r}")
            player_id = names.index(args.model_label)
            counts = count_events(events, player_id)
            for key, value in counts.items():
                total[key] += value
            for key in hanchans_with:
                hanchans_with[key] += int(counts[key] > 0)
            scores = final_scores(events)
            if scores is not None:
                order = sorted(range(4), key=lambda seat: (-scores[seat], seat))
                ranks[str(order.index(player_id) + 1)] += 1
            decisions += counts["dahai"] + counts["riichi"] + counts["fuuro"]
        except Exception as exc:  # noqa: BLE001
            malformed.append({"path": str(path), "error": str(exc)})
        if index % args.progress_every == 0:
            print(f"[outcome-audit] files {index}/{len(files)} malformed={len(malformed)}", flush=True)

    hanchans = len(files)
    rates = {
        key + "_per_hanchan": total[key] / hanchans if hanchans else 0.0
        for key in ("agari", "houjuu", "fuuro", "riichi")
    }
    rates.update({key + "_hanchan_rate": value / hanchans if hanchans else 0.0 for key, value in hanchans_with.items()})
    report = {
        "schema": "keqing.mortal.trainable_view_outcomes.v1",
        "inputs": {
            "file_index": str(args.file_index.resolve()),
            "model_label": args.model_label,
            "file_count": len(files),
        },
        "hanchans": hanchans,
        "raw_event_counts": total,
        "hanchans_with_event": hanchans_with,
        "rates": rates,
        "final_rank_counts": ranks,
        "decision_like_event_count": decisions,
        "malformed_count": len(malformed),
        "malformed": malformed[:50],
        "definitions": {
            "agari": "hora events where actor is the selected player",
            "houjuu": "hora events where target is the selected player",
            "fuuro": "chi/pon/kakan/ankan/daiminkan events where actor is the selected player",
            "riichi": "reach events where actor is the selected player",
            "final_rank": "rank from the last end_game scores event, stable seat order for ties",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "hanchans": hanchans, "malformed": len(malformed)}, ensure_ascii=False), flush=True)
    if malformed:
        raise SystemExit("trainable-view outcome audit found malformed files")


if __name__ == "__main__":
    main()
