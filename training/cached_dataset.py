"""共享缓存数据集：统一 npz 读取、shuffle buffer 和花色增强。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from keqing_core.cache_schema import BASE_CACHE_FIELDS

DEFAULT_BUFFER_SIZE = 2000

_SUIT_SLICES = [slice(0, 9), slice(9, 18), slice(18, 27)]
_SUIT_PERMS = [
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]
_TF_SHAPE = (54, 34)
_SC_SHAPE = (48,)


def apply_suit_perm(tile_feat: np.ndarray, mask: np.ndarray, action_idx: int, perm: tuple) -> Tuple[np.ndarray, np.ndarray, int]:
    """在特征层面做花色置换。"""
    src = [_SUIT_SLICES[i] for i in perm]
    new_tile = tile_feat.copy()
    new_tile[:, _SUIT_SLICES[0]] = tile_feat[:, src[0]]
    new_tile[:, _SUIT_SLICES[1]] = tile_feat[:, src[1]]
    new_tile[:, _SUIT_SLICES[2]] = tile_feat[:, src[2]]

    new_action = action_idx
    if action_idx < 27:
        suit_i = action_idx // 9
        rank = action_idx % 9
        new_suit_i = perm.index(suit_i)
        new_action = new_suit_i * 9 + rank

    new_mask = mask.copy()
    new_mask[_SUIT_SLICES[0]] = mask[src[0]]
    new_mask[_SUIT_SLICES[1]] = mask[src[1]]
    new_mask[_SUIT_SLICES[2]] = mask[src[2]]

    return new_tile, new_mask, new_action


def split_cached_files(
    root_dirs: List[Path],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    all_files: List[Path] = []
    for d in root_dirs:
        all_files.extend(sorted(d.glob("*.npz")))

    rng = random.Random(seed)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * val_ratio))
    return all_files[n_val:], all_files[:n_val]


class BaseCacheAdapter:
    extra_fields: tuple[str, ...] = ()
    fail_fast = False

    def load_optional_arrays(self, data, sample_count: int, *, path: Path | None = None) -> Dict[str, np.ndarray]:
        del path
        del data, sample_count
        return {}

    def permute_row_extra(self, row_extra: Dict[str, object], perm: tuple, action_idx: int) -> Dict[str, object]:
        del perm, action_idx
        return row_extra

    def permute_scalar(self, scalar: np.ndarray, perm: tuple) -> np.ndarray:
        del perm
        return scalar

    def build_sample(
        self,
        tile_feat: np.ndarray,
        scalar: np.ndarray,
        mask: np.ndarray,
        action_idx: int,
        value: float,
        row_extra: Dict[str, object],
    ) -> Tuple:
        del row_extra
        return tile_feat, scalar, mask, action_idx, value

    @staticmethod
    def collate(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
        tile_feats, scalars, masks, action_idxs, values = zip(*batch)
        return (
            torch.from_numpy(np.stack(tile_feats)),
            torch.from_numpy(np.stack(scalars)),
            torch.from_numpy(np.stack(masks)),
            torch.tensor(action_idxs, dtype=torch.long),
            torch.tensor(values, dtype=torch.float32),
        )


class GenericCachedMjaiDataset(IterableDataset):
    def __init__(
        self,
        file_paths: List[Path],
        *,
        adapter: BaseCacheAdapter,
        shuffle: bool = True,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        seed: Optional[int] = None,
        aug_perms: int = 2,
    ):
        self.file_paths = file_paths
        self.adapter = adapter
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed if seed is not None else random.randint(0, 2**31)
        self.aug_perms = aug_perms

    def __iter__(self) -> Iterator[Tuple]:
        worker_info = torch.utils.data.get_worker_info()
        paths = self.file_paths
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        rng = random.Random(self.seed)
        if self.shuffle:
            rng.shuffle(paths)

        buffer: List[Tuple] = []

        for path in paths:
            try:
                data = np.load(path, allow_pickle=True)
                # Keep on-disk dtypes here to avoid whole-file host-side copies.
                # Cast only where required later in collate / device transfer.
                tile_feats = data[BASE_CACHE_FIELDS[0]]
                scalars = data[BASE_CACHE_FIELDS[1]]
                masks = data[BASE_CACHE_FIELDS[2]]
                action_idxs = data[BASE_CACHE_FIELDS[3]]
                values = data[BASE_CACHE_FIELDS[4]]
                extra_arrays = self.adapter.load_optional_arrays(data, len(tile_feats), path=path)
            except Exception:
                if self.adapter.fail_fast:
                    raise
                continue

            perms_to_use = []
            if self.aug_perms > 0:
                perms_to_use = rng.sample(_SUIT_PERMS, min(self.aug_perms, len(_SUIT_PERMS)))

            for i in range(len(tile_feats)):
                action_idx = int(action_idxs[i])
                row_extra = {key: extra_arrays[key][i] for key in extra_arrays}
                buffer.append(
                    self.adapter.build_sample(
                        tile_feats[i],
                        scalars[i],
                        masks[i],
                        action_idx,
                        values[i],
                        row_extra,
                    )
                )

                for perm in perms_to_use:
                    tf, mk, new_ai = apply_suit_perm(tile_feats[i], masks[i], action_idx, perm)
                    sc = self.adapter.permute_scalar(scalars[i], perm)
                    perm_extra = self.adapter.permute_row_extra(row_extra, perm, new_ai)
                    buffer.append(
                        self.adapter.build_sample(
                            tf,
                            sc,
                            mk,
                            new_ai,
                            values[i],
                            perm_extra,
                        )
                    )

                while len(buffer) >= self.buffer_size:
                    idx = rng.randrange(len(buffer))
                    yield buffer[idx]
                    buffer[idx] = buffer[-1]
                    buffer.pop()

        if self.shuffle:
            rng.shuffle(buffer)
        for sample in buffer:
            yield sample


__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "BaseCacheAdapter",
    "GenericCachedMjaiDataset",
    "apply_suit_perm",
    "split_cached_files",
]
