#!/usr/bin/env python3
"""Relabel all-perspective Mortal states with a local external teacher's greedy action.

Phase 1 of the 2026-09-07 mainline shift (现成语料 → 本地 external 重标注 → 随机初始化
Mortal-native 学生 → 纯合法动作策略训练):

- Reuses existing hanchan corpora (S0 pure ext selfplay, V2 mixed pools, D3 mixed pools).
  No new games are generated; files are read only.
- Loads every hanchan with ``GameplayLoader(player_names=None)`` so all four seat
  perspectives are produced.  ``oracle=False`` keeps the observation limited to what
  the acting player can see (information boundary of the student contract).
- The frozen external checkpoint re-answers "what would I do here" on every state.
  The teacher's greedy action over its legal mask becomes the 46-dim hard label.
  Q values are also stored for diagnostics, but the first-stage training target is
  the hard greedy label only.
- Hanchans are split into train/holdout by a deterministic SHA-256 hash of the log
  identity, so the split is stable across runs and never depends on file order.
- Output: one ``.npz`` shard per output chunk plus a JSON manifest binding every
  shard to source file SHA-256s and teacher checkpoint SHA-256.

The logged (behavior) action is kept alongside the teacher label so audits can
measure teacher-vs-behavior disagreement without re-running inference.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel existing Mortal logs with a local external teacher")
    parser.add_argument("--teacher", type=Path, default=Path(r"E:\AUbuntuProject\project\keqing1\artifacts\external_mortal_20240308_best_min.pth"))
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pool",
        action="append",
        required=True,
        help="POOL=DIR competitive-identity pool of hanchan logs; repeatable. Example: S0=E:/.../S0_pure_ext_selfplay_6000h/logs",
    )
    parser.add_argument("--holdout-ratio", type=float, default=0.1)
    parser.add_argument("--holdout-salt", default="keqing.mortal.relabel.v1")
    parser.add_argument("--rows-per-shard", type=int, default=10_000, help="rows per output .npz shard; memory bound is ~137KB/row uncompressed")
    parser.add_argument("--inference-batch", type=int, default=1024)
    parser.add_argument("--enable-amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-files", type=int, default=0, help="debug: only process the first N files per pool")
    parser.add_argument("--manifest-snapshot-every", type=int, default=100, help="persist the resume manifest every N files")
    parser.add_argument("--resume", action="store_true", help="skip pools/chunks already recorded as complete in the manifest")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _holdout_bucket(log_identity: str, salt: str, ratio: float) -> str:
    digest = hashlib.sha256(f"{salt}:{log_identity}".encode("utf-8")).hexdigest()
    value = int(digest[:16], 16) / float(1 << 64)
    return "holdout" if value < ratio else "train"


def _iter_pool_files(pool_spec: str) -> tuple[str, list[Path]]:
    if "=" not in pool_spec:
        raise ValueError(f"--pool must be POOL=DIR, got: {pool_spec}")
    name, _, directory = pool_spec.partition("=")
    name = name.strip()
    directory = directory.strip()
    if not name or not directory:
        raise ValueError(f"--pool must be POOL=DIR, got: {pool_spec}")
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"pool directory does not exist: {root}")
    files = sorted(root.glob("*.json.gz"))
    if not files:
        raise FileNotFoundError(f"pool directory has no .json.gz logs: {root}")
    return name, files


def _load_teacher(teacher_path: Path, mortal_root: Path, device: torch.device):
    mortal_python_dir = (mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))
    from model import Brain, DQN  # noqa: PLC0415

    state = torch.load(teacher_path, weights_only=True, map_location=torch.device("cpu"))
    cfg = state["config"]
    version = int(cfg["control"].get("version", 4))
    mortal = Brain(
        version=version,
        conv_channels=int(cfg["resnet"]["conv_channels"]),
        num_blocks=int(cfg["resnet"]["num_blocks"]),
    ).to(device).eval()
    dqn = DQN(version=version).to(device).eval()
    mortal.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    return mortal, dqn, version


class _ShardWriter:
    """Accumulate rows and flush ``.npz`` shards of fixed row counts.

    Rows are pre-stacked into 2048-row blocks to keep the Python heap compact;
    a shard accumulates at most ``rows_per_shard`` rows (about 137 KB/row
    uncompressed, so the default 10k rows caps peak memory near 1.4 GB).
    """

    _BLOCK_ROWS = 2048

    def __init__(self, output_dir: Path, split: str, rows_per_shard: int):
        self.output_dir = output_dir
        self.split = split
        self.rows_per_shard = int(rows_per_shard)
        self.blocks: list[dict[str, np.ndarray]] = []
        self.block_rows = 0
        self.pending: list[dict[str, Any]] = []
        self.shard_index = 0
        self.rows_written = 0
        self.rows_in_current_shard = 0
        # Per-shard row counts (index i = shard i), for manifest bookkeeping
        # so consumers never have to decompress shards just to count rows.
        self.shard_rows: list[int] = []
        # Phase timing (seconds) for this writer.
        self.pack_seconds = 0.0
        self.write_seconds = 0.0

    def add(self, row: dict[str, Any]) -> None:
        self.pending.append(row)
        if len(self.pending) >= self._BLOCK_ROWS:
            self._pack_block()
        if self.rows_in_current_shard >= self.rows_per_shard:
            self.flush()

    def _pack_block(self) -> None:
        if not self.pending:
            return
        t0 = time.perf_counter()
        self.blocks.append(
            {
                "obs": np.stack([row["obs"] for row in self.pending]),
                "mask": np.stack([row["mask"] for row in self.pending]),
                "teacher_action": np.asarray([row["teacher_action"] for row in self.pending], dtype=np.int64),
                "behavior_action": np.asarray([row["behavior_action"] for row in self.pending], dtype=np.int64),
                "teacher_q": np.stack([row["teacher_q"] for row in self.pending]).astype(np.float32),
                "pool": np.asarray([row["pool_index"] for row in self.pending], dtype=np.int16),
                "player_id": np.asarray([row["player_id"] for row in self.pending], dtype=np.int8),
                "file_row": np.asarray([row["file_row"] for row in self.pending], dtype=np.int32),
            }
        )
        self.block_rows += len(self.pending)
        self.rows_in_current_shard += len(self.pending)
        self.pending.clear()
        self.pack_seconds += time.perf_counter() - t0

    def flush(self) -> None:
        self._pack_block()
        if not self.blocks:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.split}_{self.shard_index:04d}.npz"
        payload = {
            key: (
                self.blocks[0][key]
                if len(self.blocks) == 1
                else np.concatenate([block[key] for block in self.blocks])
            )
            for key in self.blocks[0]
        }
        tmp_path = path.with_name(path.name + ".tmp")
        t0 = time.perf_counter()
        # Compression level 1 instead of numpy's default (zlib 6): 2.8x faster on
        # this workload at ~55% larger files (measured 0.85s vs 2.38s per 2048-row
        # shard; 4.2 vs 2.7 MiB).  npz remains a plain zip, so np.load reads it
        # unchanged.  Write via an explicit file object so the .tmp suffix is
        # preserved for the atomic replace.
        with zipfile.ZipFile(
            tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as zf:
            for name, array in payload.items():
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, array, allow_pickle=False)
                zf.writestr(f"{name}.npy", buffer.getvalue())
        tmp_path.replace(path)
        self.write_seconds += time.perf_counter() - t0
        rows_in_shard = int(len(payload["obs"]))
        self.shard_rows.append(rows_in_shard)
        self.rows_written += rows_in_shard
        self.blocks.clear()
        self.block_rows = 0
        self.shard_index += 1
        self.rows_in_current_shard = 0

    def close(self) -> dict[str, Any]:
        self.flush()
        return {
            "shards": self.shard_index,
            "rows": self.rows_written,
            "shard_rows": list(self.shard_rows),
            "pack_seconds": self.pack_seconds,
            "write_seconds": self.write_seconds,
        }


def _write_manifest(manifest: dict[str, Any], manifest_path: Path, *, incomplete: bool, split_stats: dict, pool_stats: dict, writers: dict | None = None) -> None:
    """Flush shards and atomically persist the manifest as a resume snapshot.

    The writers are flushed FIRST so that every counted row is on disk: the
    snapshot's ``rows``/``flushed_rows`` are always equal.  Snapshot boundaries
    may produce a short tail shard; that is fine because row sequences stay
    gapless and resume continues from the recorded shard index.
    """
    if writers:
        for writer in writers.values():
            writer.flush()
    manifest = dict(manifest)
    manifest["splits"] = {
        split: {
            **split_stats[split],
            "teacher_behavior_agreement": (
                split_stats[split]["behavior_matches"] / split_stats[split]["rows"]
                if split_stats[split]["rows"]
                else None
            ),
            "flushed_shards": (writers[split].shard_index if writers else 0),
            "flushed_rows": (writers[split].rows_written if writers else 0),
            "shard_rows": (list(writers[split].shard_rows) if writers else []),
            "shard_files": sorted(p.name for p in manifest_path.parent.glob(f"{split}_*.npz")),
        }
        for split in ("train", "holdout")
    }
    manifest["pool_stats"] = {name: dict(stats) for name, stats in pool_stats.items()}
    manifest["incomplete"] = bool(incomplete)
    tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(manifest_path)


def _clean_stale_shards(output_dir: Path, previous: dict[str, Any]) -> None:
    """Delete shard files beyond the manifest's recorded flush boundary."""
    for split in ("train", "holdout"):
        recorded = previous.get("splits", {}).get(split, {}).get("flushed_shards", 0)
        for path in output_dir.glob(f"{split}_*.npz"):
            try:
                index = int(path.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if index >= recorded:
                path.unlink()
                print(f"resume: deleted stale shard {path.name}", flush=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.time()
    device = torch.device(args.device)
    if str(device).startswith("cuda"):
        print(f"device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    else:
        print(f"device: {device}", flush=True)

    pools: list[tuple[str, list[Path]]] = []
    for spec in args.pool:
        pools.append(_iter_pool_files(spec))
    # Multiple directories may share one competitive identity (e.g. D3 shards):
    # merge their file lists under the same pool name.
    merged: dict[str, list[Path]] = {}
    for name, files in pools:
        merged.setdefault(name, []).extend(files)
    pools = [(name, files) for name, files in merged.items()]
    pool_names = [name for name, _ in pools]

    teacher_path = args.teacher.resolve()
    if not teacher_path.exists():
        raise FileNotFoundError(f"teacher checkpoint does not exist: {teacher_path}")
    teacher_sha = _sha256_file(teacher_path)
    mortal, dqn, version = _load_teacher(teacher_path, args.mortal_root, device)

    from libriichi.dataset import GameplayLoader  # noqa: PLC0415

    loader = GameplayLoader(
        version=version,
        oracle=False,
        player_names=None,
        excludes=None,
        always_include_kan_select=True,
        augmented=False,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema": "keqing.mortal.ext_relabel.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "teacher_checkpoint": str(teacher_path),
        "teacher_sha256": teacher_sha,
        "teacher_network": {"conv_channels": 256, "num_blocks": 54, "version": version},
        "holdout_ratio": float(args.holdout_ratio),
        "holdout_salt": str(args.holdout_salt),
        "rows_per_shard": int(args.rows_per_shard),
        "inference_batch": int(args.inference_batch),
        "enable_amp": bool(args.enable_amp),
        "pools": {
            name: {
                "directory": str(files[0].parent),
                "directories": sorted({str(f.parent) for f in files}),
                "files": len(files),
            }
            for name, files in pools
        },
        "files": {},
        "splits": {},
    }

    # Resume support: a manifest snapshot records each completed source file and
    # the flush boundary (shard count per split).  On resume: stale tail shards
    # beyond the boundary are deleted; completed source files are skipped; row
    # counters restart from the flushed baseline so new shards continue the
    # sequence without duplicating or dropping rows.
    previous: dict[str, Any] = {}
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("schema") != manifest["schema"]:
            raise RuntimeError("existing manifest schema mismatch; refuse to resume")
        if previous.get("teacher_sha256") != teacher_sha:
            raise RuntimeError("existing manifest checkpoint differs; refuse to resume")
        if previous.get("rows_per_shard") != manifest["rows_per_shard"]:
            raise RuntimeError("existing manifest rows_per_shard differs; refuse to resume")
        completed = previous.get("files", {})
        print(f"resume: manifest records {len(completed)} completed files", flush=True)
        _clean_stale_shards(output_dir, previous)
        manifest["files"] = dict(completed)
    else:
        if manifest_path.exists():
            raise FileExistsError(f"manifest already exists at {manifest_path}; pass --resume or choose a new output dir")

    writers = {
        "train": _ShardWriter(output_dir, "train", args.rows_per_shard),
        "holdout": _ShardWriter(output_dir, "holdout", args.rows_per_shard),
    }
    if previous:
        # Continue shard numbering/row counters from the flushed baseline so
        # shard files remain a gapless row sequence across resumes.
        for split, writer in writers.items():
            prev_split = previous.get("splits", {}).get(split, {})
            writer.shard_index = int(prev_split.get("flushed_shards", 0))
            writer.rows_written = int(prev_split.get("flushed_rows", 0))
            writer.shard_rows = [int(v) for v in prev_split.get("shard_rows", [])]
    split_stats = {
        split: {"hanchans": 0, "rows": 0, "behavior_matches": 0}
        for split in ("train", "holdout")
    }
    pool_stats: dict[str, dict[str, int]] = {name: {"files": 0, "rows": 0} for name, _ in pools}
    if previous:
        # Restore counting baselines from the snapshot so totals reflect all
        # completed files, not just this invocation's tail.
        for split, stats in previous.get("splits", {}).items():
            if split in split_stats:
                split_stats[split]["hanchans"] = int(stats.get("hanchans", 0))
                split_stats[split]["rows"] = int(stats.get("rows", 0))
                split_stats[split]["behavior_matches"] = int(stats.get("behavior_matches", 0))
        for name, stats in previous.get("pool_stats", {}).items():
            if name in pool_stats:
                pool_stats[name] = {k: int(v) for k, v in stats.items()}
        teacher_agreement_total = split_stats["train"]["behavior_matches"] + split_stats["holdout"]["behavior_matches"]
        behavior_rows_total = split_stats["train"]["rows"] + split_stats["holdout"]["rows"]

    # Observation buffer for batched teacher inference.
    obs_buffer: list[np.ndarray] = []
    mask_buffer: list[np.ndarray] = []
    row_meta: list[dict[str, Any]] = []
    file_hashes: dict[str, str] = {}
    if not previous:
        teacher_agreement_total = 0
        behavior_rows_total = 0

    # Per-phase timing (seconds): where the relabel wall-clock actually goes.
    #   parse:    libriichi gzip load + PlayerState replay per source file
    #   sha:      source-file SHA-256 hashing
    #   stack:    np.stack of the inference batch (numpy -> pinned transfer)
    #   inference: teacher forward (mortal + dqn) incl. H2D/D2H copies
    #   write:    npz compression + atomic replace
    #   total:    whole run (includes everything else)
    timing = {
        "parse": 0.0,
        "sha": 0.0,
        "stack": 0.0,
        "inference": 0.0,
        "write": 0.0,
    }

    def _flush_inference() -> None:
        nonlocal teacher_agreement_total, behavior_rows_total
        if not obs_buffer:
            return
        t_stack = time.perf_counter()
        obs_t = torch.as_tensor(np.stack(obs_buffer), device=device)
        mask_t = torch.as_tensor(np.stack(mask_buffer), device=device)
        timing["stack"] += time.perf_counter() - t_stack
        t_infer = time.perf_counter()
        with (
            torch.autocast(device.type, enabled=bool(args.enable_amp)),
            torch.inference_mode(),
        ):
            phi = mortal(obs_t)
            q_out = dqn(phi, mask_t)
        q_np = q_out.to(torch.float32).cpu().numpy()
        greedy = q_np.argmax(axis=-1)
        timing["inference"] += time.perf_counter() - t_infer
        for i, meta in enumerate(row_meta):
            row = {
                "obs": obs_buffer[i],
                "mask": mask_buffer[i],
                "teacher_action": int(greedy[i]),
                "behavior_action": meta["behavior_action"],
                "teacher_q": q_np[i],
                "pool_index": meta["pool_index"],
                "player_id": meta["player_id"],
 "file_row": meta["file_row"],
            }
            writers[meta["split"]].add(row)
            split_stats[meta["split"]]["rows"] += 1
            if row["teacher_action"] == row["behavior_action"]:
                split_stats[meta["split"]]["behavior_matches"] += 1
                teacher_agreement_total += 1
            behavior_rows_total += 1
        obs_buffer.clear()
        mask_buffer.clear()
        row_meta.clear()
        del obs_t, mask_t

    for pool_index, (pool_name, files) in enumerate(pools):
        selected = files if args.limit_files <= 0 else files[: args.limit_files]
        print(f"[{pool_name}] {len(selected)} files", flush=True)
        for file_index, path in enumerate(selected):
            file_key = str(path)
            if file_key in manifest["files"]:
                # Already recorded by a previous snapshot; skip entirely.
                continue
            t_sha = time.perf_counter()
            file_sha = _sha256_file(path)
            timing["sha"] += time.perf_counter() - t_sha
            file_hashes[file_key] = file_sha
            split = _holdout_bucket(file_key, args.holdout_salt, args.holdout_ratio)
            t_parse = time.perf_counter()
            data = loader.load_gz_log_files([str(path)])
            timing["parse"] += time.perf_counter() - t_parse
            if len(data) != 1:
                raise RuntimeError(f"loader returned {len(data)} entries for one file: {path}")
            for game in data[0]:
                obs = game.take_obs()
                masks = game.take_masks()
                actions = game.take_actions()
                player_id = int(game.take_player_id())
                for row_index in range(len(obs)):
                    obs_buffer.append(obs[row_index])
                    mask_buffer.append(masks[row_index])
                    row_meta.append(
                        {
                            "split": split,
                            "pool_index": pool_index,
                            "player_id": player_id,
                            "file_row": row_index,
                            "behavior_action": int(actions[row_index]),
                        }
                    )
                    if len(obs_buffer) >= args.inference_batch:
                        _flush_inference()
            pool_stats[pool_name]["files"] += 1
            pool_stats[pool_name]["rows"] += len(obs) * 4  # all four perspectives contributed
            split_stats[split]["hanchans"] += 1
            manifest["files"][file_key] = {
                "pool": pool_name,
                "sha256": file_sha,
                "split": split,
            }
            if (file_index + 1) % 200 == 0:
                elapsed = time.time() - started_at
                print(
                    f"[{pool_name}] {file_index + 1}/{len(selected)} files, "
                    f"rows={split_stats['train']['rows'] + split_stats['holdout']['rows']:,}, "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )
            # Persist the manifest periodically so an interrupted run can
            # --resume from the last flush boundary: stale tail shards are
            # discarded on resume, and completed source files are skipped.
            if (file_index + 1) % int(args.manifest_snapshot_every) == 0:
                _write_manifest(
                    manifest, manifest_path,
                    incomplete=True,
                    split_stats=split_stats,
                    pool_stats=pool_stats,
                    writers=writers,
                )
        _flush_inference()

    for writer in writers.values():
        writer.close()

    elapsed = time.time() - started_at
    manifest["splits"] = {
        split: {
            **split_stats[split],
            "teacher_behavior_agreement": (
                split_stats[split]["behavior_matches"] / split_stats[split]["rows"]
                if split_stats[split]["rows"]
                else None
            ),
            "flushed_shards": writers[split].shard_index,
            "flushed_rows": writers[split].rows_written,
            "shard_rows": list(writers[split].shard_rows),
            "shard_files": sorted(p.name for p in output_dir.glob(f"{split}_*.npz")),
        }
        for split in ("train", "holdout")
    }
    manifest["pool_stats"] = pool_stats
    manifest["totals"] = {
        "hanchans": split_stats["train"]["hanchans"] + split_stats["holdout"]["hanchans"],
        "rows": split_stats["train"]["rows"] + split_stats["holdout"]["rows"],
        "teacher_behavior_agreement": (
            teacher_agreement_total / behavior_rows_total if behavior_rows_total else None
        ),
    }
    manifest["timing"] = {
        **{key: round(value, 3) for key, value in timing.items()},
        "pack": round(sum(w.pack_seconds for w in writers.values()), 3),
        "write": round(sum(w.write_seconds for w in writers.values()), 3),
        "total": round(elapsed, 3),
        "rows_per_sec_total": (
            round(manifest["totals"]["rows"] / elapsed, 1) if elapsed > 0 else None
        ),
    }
    tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(manifest_path)

    elapsed = time.time() - started_at
    print(json.dumps({
        "totals": manifest["totals"],
        "splits": {s: manifest["splits"][s] for s in ("train", "holdout")},
        "timing": manifest["timing"],
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
