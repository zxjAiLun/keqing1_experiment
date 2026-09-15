"""Export detailed Mortal/libriichi match statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import importlib
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any

_NATIVE_SUFFIXES = {".pyd", ".so", ".dll"}

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.mortal import eval_metrics  # noqa: E402

DEFAULT_RANK_PTS = eval_metrics.TENHOU_RANK_POINTS

RAW_FIELDS = (
    "game",
    "round",
    "oya",
    "point",
    "rank_1",
    "rank_2",
    "rank_3",
    "rank_4",
    "tobi",
    "agari",
    "houjuu",
    "fuuro",
    "fuuro_num",
    "riichi",
    "ryukyoku",
    "yakuman",
    "nagashi_mangan",
)

DERIVED_FIELDS = (
    "rank_1_rate",
    "rank_2_rate",
    "rank_3_rate",
    "rank_4_rate",
    "tobi_rate",
    "avg_rank",
    "avg_point_per_game",
    "avg_point_per_round",
    "agari_rate",
    "houjuu_rate",
    "fuuro_rate",
    "riichi_rate",
    "ryukyoku_rate",
    "avg_point_per_agari",
    "avg_point_per_oya_agari",
    "avg_point_per_ko_agari",
    "avg_point_per_riichi_agari",
    "avg_point_per_fuuro_agari",
    "avg_point_per_dama_agari",
    "avg_point_per_ryukyoku",
    "avg_agari_jun",
    "avg_riichi_agari_jun",
    "avg_fuuro_agari_jun",
    "avg_dama_agari_jun",
    "avg_houjuu_jun",
    "avg_point_per_houjuu",
    "avg_point_per_houjuu_to_oya",
    "avg_point_per_houjuu_to_ko",
    "chasing_riichi_rate",
    "riichi_chased_rate",
    "agari_rate_after_riichi",
    "houjuu_rate_after_riichi",
    "avg_riichi_jun",
    "avg_riichi_point",
    "avg_fuuro_num",
    "agari_rate_after_fuuro",
    "houjuu_rate_after_fuuro",
    "avg_fuuro_point",
    "agari_rate_as_oya",
    "agari_as_oya_rate",
    "houjuu_to_oya_rate",
    "yakuman_rate",
    "nagashi_mangan_rate",
)

DISPLAY_ROWS = (
    ("Games", "game"),
    ("Rounds", "round"),
    ("Rounds as dealer", "oya"),
    ("1st (rate)", ("rank_1", "rank_1_rate")),
    ("2nd (rate)", ("rank_2", "rank_2_rate")),
    ("3rd (rate)", ("rank_3", "rank_3_rate")),
    ("4th (rate)", ("rank_4", "rank_4_rate")),
    ("Tobi(rate)", ("tobi", "tobi_rate")),
    ("Avg rank", "avg_rank"),
    ("Total rank pt", "total_rank_pt"),
    ("Avg rank pt", "avg_rank_pt"),
    ("Total delta score", "point"),
    ("Avg game delta score", "avg_point_per_game"),
    ("Avg round delta score", "avg_point_per_round"),
    ("Win rate", "agari_rate"),
    ("Deal-in rate", "houjuu_rate"),
    ("Fuuro round rate", "fuuro_rate"),
    ("Avg call events per game", "avg_fuuro_num"),
    ("Riichi rate", "riichi_rate"),
    ("Ryukyoku rate", "ryukyoku_rate"),
    ("Avg winning delta score", "avg_point_per_agari"),
    ("Avg winning delta score as dealer", "avg_point_per_oya_agari"),
    ("Avg winning delta score as non-dealer", "avg_point_per_ko_agari"),
    ("Avg riichi winning delta score", "avg_point_per_riichi_agari"),
    ("Avg open winning delta score", "avg_point_per_fuuro_agari"),
    ("Avg dama winning delta score", "avg_point_per_dama_agari"),
    ("Avg ryukyoku delta score", "avg_point_per_ryukyoku"),
    ("Avg winning turn", "avg_agari_jun"),
    ("Avg riichi winning turn", "avg_riichi_agari_jun"),
    ("Avg open winning turn", "avg_fuuro_agari_jun"),
    ("Avg dama winning turn", "avg_dama_agari_jun"),
    ("Avg deal-in turn", "avg_houjuu_jun"),
    ("Avg deal-in delta score", "avg_point_per_houjuu"),
    ("Avg deal-in delta score to dealer", "avg_point_per_houjuu_to_oya"),
    ("Avg deal-in delta score to non-dealer", "avg_point_per_houjuu_to_ko"),
    ("Chasing riichi rate", "chasing_riichi_rate"),
    ("Riichi chased rate", "riichi_chased_rate"),
    ("Winning rate after riichi", "agari_rate_after_riichi"),
    ("Deal-in rate after riichi", "houjuu_rate_after_riichi"),
    ("Avg riichi turn", "avg_riichi_jun"),
    ("Avg riichi delta score", "avg_riichi_point"),
    ("Winning rate after call", "agari_rate_after_fuuro"),
    ("Deal-in rate after call", "houjuu_rate_after_fuuro"),
    ("Avg call delta score", "avg_fuuro_point"),
    ("Dealer wins/all dealer rounds", "agari_rate_as_oya"),
    ("Dealer wins/all wins", "agari_as_oya_rate"),
    ("Deal-in to dealer/all deal-ins", "houjuu_to_oya_rate"),
    ("Yakuman (rate)", ("yakuman", "yakuman_rate")),
    ("Nagashi mangan (rate)", ("nagashi_mangan", "nagashi_mangan_rate")),
)


def stat_class_search_path(mortal_root: str | Path | None) -> list[str]:
    """The ``sys.path`` entry ``import_stat_class`` would add; ``[]`` means "add none".

    Kept separate so the decision is testable without a native module present:
    the default must never move ``sys.path``, and passing a root must.
    """
    if mortal_root is None:
        return []
    return [str((Path(mortal_root) / "mortal").resolve())]


def import_stat_class(mortal_root: str | Path | None = None) -> Any:
    """Import ``libriichi.stat.Stat`` from the *active environment*.

    ``mortal_root`` used to default to ``third_party/Mortal`` and pushed
    ``<root>/mortal`` to the front of ``sys.path``.  That silently shadowed the
    environment's own native module: in a clean process started with the
    mandated venv, ``libriichi`` resolved to the repository copy (``05647bf7…``
    instead of the pinned ``19bb181e…``), so this "standalone" entry point was not
    running the interpreter it was launched with.

    It is now opt-in.  Pass a root only when you *specifically* want the
    repository build; the report records which module actually loaded
    (``native_provenance``) so a cross-run comparison can be checked.
    """
    for entry in stat_class_search_path(mortal_root):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from libriichi.stat import Stat  # noqa: PLC0415

    return Stat


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def native_provenance(mortal_root: str | Path | None = None) -> dict[str, Any]:
    """Which native module this process actually loaded, and whether we moved ``sys.path``.

    The registry carries a hard constraint for cross-run comparisons: record the
    interpreter, the libriichi/native module path and sha256, and whether
    ``sys.path`` was rewritten.  A report that says which binary produced it can
    be compared; one that does not cannot.

    ``libriichi`` is only a shim (``from riichi.stat import *``); the identity that
    matters is the compiled ``riichi`` extension, so that is what is hashed.
    """
    import libriichi  # noqa: PLC0415

    shim_file = Path(libriichi.__file__).resolve()
    native: list[Path] = []
    try:
        riichi = importlib.import_module("riichi")
    except ImportError:
        riichi = None
    riichi_file = Path(getattr(riichi, "__file__", "") or "").resolve() if riichi else None
    if riichi_file and riichi_file.suffix.lower() in _NATIVE_SUFFIXES:
        native.append(riichi_file)

    binaries: list[Path] = [*native]
    if shim_file.suffix.lower() in _NATIVE_SUFFIXES:
        binaries.append(shim_file)
    if not binaries:
        # A pure-python package layout with the extension somewhere beside it.
        for directory in {shim_file.parent, *(p.parent for p in native)}:
            for pattern in ("*.pyd", "*.so", "*.dll"):
                binaries.extend(sorted(directory.glob(pattern)))

    seen: set[str] = set()
    recorded: list[dict[str, Any]] = []
    for path in binaries:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            recorded.append(
                {"name": path.name, "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            )
        except OSError:  # pragma: no cover - an unreadable binary is not fatal here
            continue
    return {
        "module_file": str(shim_file),
        "module_dir": str(shim_file.parent),
        "native_module_file": str(riichi_file) if riichi_file else None,
        "native_module_sha256": next((b["sha256"] for b in recorded if b.get("name") == (riichi_file.name if riichi_file else "")), None),
        "binaries": recorded,
        "interpreter": sys.executable,
        "sys_path_rewritten": mortal_root is not None,
        "requested_mortal_root": str(mortal_root) if mortal_root is not None else None,
    }


def parse_player_specs(specs: Sequence[str]) -> dict[str, str]:
    players: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"player spec must be LABEL=PLAYER_NAME, got: {spec}")
        label, player_name = spec.split("=", 1)
        label = label.strip()
        player_name = player_name.strip()
        if not label or not player_name:
            raise ValueError(f"player spec must be LABEL=PLAYER_NAME, got: {spec}")
        players[label] = player_name
    return players


def stat_to_metrics(stat: Any, *, rank_pts: Sequence[int | float] = DEFAULT_RANK_PTS) -> dict[str, Any]:
    metrics: dict[str, Any] = {"raw": {}, "derived": {}, "text": str(stat)}
    for field in RAW_FIELDS:
        metrics["raw"][field] = _normalize_value(getattr(stat, field))
    for field in DERIVED_FIELDS:
        metrics["derived"][field] = _normalize_value(getattr(stat, field))
    pts = [float(value) for value in rank_pts]
    total_rank_pt = sum(float(getattr(stat, f"rank_{rank}")) * pts[rank - 1] for rank in range(1, 5))
    metrics["derived"]["total_rank_pt"] = _normalize_value(total_rank_pt)
    metrics["derived"]["avg_rank_pt"] = _normalize_value(total_rank_pt / float(stat.game) if stat.game else float("nan"))
    return metrics


def build_stat_report(
    *,
    log_dir: str | Path,
    players: Mapping[str, str],
    mortal_root: str | Path | None = None,
    rank_pts: Sequence[int | float] = DEFAULT_RANK_PTS,
    rank_points_profile: str = "custom",
    require_games: bool = False,
) -> dict[str, Any]:
    stat_cls = import_stat_class(mortal_root)
    player_stats: dict[str, dict[str, Any]] = {}
    for label, player_name in players.items():
        stat = stat_cls.from_dir(str(log_dir), str(player_name), True)
        player_stats[label] = {
            "player_name": str(player_name),
            **stat_to_metrics(stat, rank_pts=rank_pts),
        }
    if require_games:
        # A wrong log dir or a mismatched player name makes libriichi return an
        # empty Stat (~0 games) instead of failing, which silently produces a
        # report full of zeros.  A report request that found nothing is an error.
        empty = [label for label, stats in player_stats.items() if not int(stats["raw"]["game"])]
        if empty:
            raise RuntimeError(
                f"no games found under {log_dir} for {empty}: the log dir is empty or those "
                "names never appear in a start_game event"
            )
    return {
        "schema": "keqing.mortal.libriichi.stat.v1",
        "backend": "libriichi.stat.Stat.from_dir",
        "native": native_provenance(mortal_root),
        "log_dir": str(log_dir),
        "rank_points_profile": str(rank_points_profile),
        "rank_points_values": [float(value) for value in rank_pts],
        "rank_pts": [float(value) for value in rank_pts],
        "players": player_stats,
    }


def write_stat_report(
    *,
    output_dir: str | Path,
    log_dir: str | Path,
    players: Mapping[str, str],
    mortal_root: str | Path | None = None,
    rank_pts: Sequence[int | float] = DEFAULT_RANK_PTS,
    rank_points_profile: str = "custom",
    require_games: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = build_stat_report(
        log_dir=log_dir,
        players=players,
        mortal_root=mortal_root,
        rank_pts=rank_pts,
        rank_points_profile=rank_points_profile,
        require_games=require_games,
    )
    (output_path / "detailed_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_path / "detailed_stats.md").write_text(format_markdown_report(report), encoding="utf-8")
    return report


def format_markdown_report(report: Mapping[str, Any]) -> str:
    players = dict(report["players"])
    labels = list(players)
    lines = [
        "# Mortal Match Detailed Stats",
        "",
        f"- Backend: `{report['backend']}`",
        f"- Log dir: `{report['log_dir']}`",
        f"- Rank point profile: `{report.get('rank_points_profile', 'custom')}`",
        f"- Rank pts: `{report['rank_pts']}`",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---" for _ in labels) + " |",
    ]
    for title, key in DISPLAY_ROWS:
        values = [_format_row_value(players[label], key) for label in labels]
        lines.append("| " + " | ".join([title, *values]) + " |")
    lines.append("")
    return "\n".join(lines)


def _format_row_value(player: Mapping[str, Any], key: str | tuple[str, str]) -> str:
    raw = player["raw"]
    derived = player["derived"]
    if isinstance(key, tuple):
        count = raw[key[0]]
        rate = derived[key[1]]
        return f"{_format_value(count)} ({_format_value(rate)})"
    if key in raw:
        return _format_value(raw[key])
    return _format_value(derived[key])


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if isfinite(value) else None
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export detailed libriichi Stat report from Mortal logs")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mortal-root",
        type=Path,
        default=None,
        help=(
            "opt-in shadowing: prepends <root>/mortal to sys.path so the repository "
            "libriichi is used instead of the active environment's. Leave unset to run "
            "against the interpreter you launched."
        ),
    )
    parser.add_argument(
        "--player",
        action="append",
        required=True,
        help="LABEL=PLAYER_NAME. Can be repeated, e.g. '4.1c (x1)=challenger'.",
    )
    eval_metrics.add_rank_point_args(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _profile, rank_pts = eval_metrics.resolve_rank_points(
        rank_points=getattr(args, "rank_points", None),
        profile=str(getattr(args, "rank_points_profile", "tenhou_reference")),
    )
    report = write_stat_report(
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        players=parse_player_specs(args.player),
        mortal_root=args.mortal_root,
        rank_pts=rank_pts,
        rank_points_profile=_profile,
        require_games=True,
    )
    print(format_markdown_report(report), end="")


if __name__ == "__main__":
    main()
