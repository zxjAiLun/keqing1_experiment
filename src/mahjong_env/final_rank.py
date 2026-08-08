from __future__ import annotations

from math import isfinite
from typing import Sequence


FINAL_RANK_DIM = 4


def _validate_scores(scores: Sequence[int | float]) -> tuple[float, float, float, float]:
    if len(scores) != FINAL_RANK_DIM:
        raise ValueError(f"scores must have length {FINAL_RANK_DIM}, got {len(scores)}")
    normalized = tuple(float(score) for score in scores)
    if not all(isfinite(score) for score in normalized):
        raise ValueError(f"scores must be finite, got {scores}")
    return normalized


def _validate_initial_oya(initial_oya: int) -> int:
    if initial_oya < 0 or initial_oya >= FINAL_RANK_DIM:
        raise ValueError(f"initial_oya must be in [0, {FINAL_RANK_DIM}), got {initial_oya}")
    return int(initial_oya)


def tie_break_order(initial_oya: int) -> tuple[int, int, int, int]:
    dealer = _validate_initial_oya(initial_oya)
    return tuple((dealer + offset) % FINAL_RANK_DIM for offset in range(FINAL_RANK_DIM))


def final_rank_for_seat(
    scores: Sequence[int | float],
    seat: int,
    *,
    initial_oya: int = 0,
) -> int:
    normalized = _validate_scores(scores)
    if seat < 0 or seat >= FINAL_RANK_DIM:
        raise ValueError(f"seat must be in [0, {FINAL_RANK_DIM}), got {seat}")
    order = tie_break_order(initial_oya)
    ordered_seats = sorted(
        range(FINAL_RANK_DIM),
        key=lambda idx: (-normalized[idx], order.index(idx)),
    )
    return ordered_seats.index(seat)


def final_ranks(
    scores: Sequence[int | float],
    *,
    initial_oya: int = 0,
) -> tuple[int, int, int, int]:
    normalized = _validate_scores(scores)
    return tuple(
        final_rank_for_seat(normalized, seat, initial_oya=initial_oya)
        for seat in range(FINAL_RANK_DIM)
    )  # type: ignore[return-value]


def expected_rank_from_probs(rank_probs: Sequence[int | float]) -> float:
    if len(rank_probs) != FINAL_RANK_DIM:
        raise ValueError(
            f"rank_probs must have length {FINAL_RANK_DIM}, got {len(rank_probs)}"
        )
    probs = tuple(float(value) for value in rank_probs)
    if not all(isfinite(value) for value in probs):
        raise ValueError(f"rank_probs must be finite, got {rank_probs}")
    total = sum(probs)
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"rank_probs must sum to 1.0, got {total}")
    return sum((rank + 1) * prob for rank, prob in enumerate(probs))


__all__ = [
    "FINAL_RANK_DIM",
    "expected_rank_from_probs",
    "final_rank_for_seat",
    "final_ranks",
    "tie_break_order",
]
