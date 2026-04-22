from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Sequence

from torch.utils.data import BatchSampler, Subset


def resolve_sample_group_values(dataset, field_name: str) -> list[str]:
    if isinstance(dataset, Subset):
        parent_values = resolve_sample_group_values(dataset.dataset, field_name)
        return [str(parent_values[int(index)]) for index in dataset.indices]
    getter = getattr(dataset, "get_sample_group_values", None)
    if getter is None:
        raise ValueError(f"Dataset of type {type(dataset).__name__} does not expose get_sample_group_values('{field_name}')")
    values = getter(field_name)
    return [str(value) for value in values]


class GroupedBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        group_field: str = "dataset",
        shuffle: bool = True,
        drop_last: bool = False,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.group_field = str(group_field)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.world_size = max(1, int(world_size))
        self.rank = max(0, int(rank))
        self.seed = int(seed)
        self.epoch = 0
        self._grouped_indices = self._build_grouped_indices()

    def _build_grouped_indices(self) -> dict[str, list[int]]:
        group_values = resolve_sample_group_values(self.dataset, self.group_field)
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(group_values):
            grouped[str(value)].append(index)
        if not grouped:
            raise ValueError(f"No groups found for field '{self.group_field}'")
        return dict(grouped)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        grouped = {key: list(indices) for key, indices in self._grouped_indices.items()}
        batches: list[list[int]] = []
        for group_name, indices in grouped.items():
            if self.shuffle:
                rng.shuffle(indices)
            start = 0
            while start < len(indices):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    break
                batches.append(batch)
                start += self.batch_size
        if self.shuffle:
            rng.shuffle(batches)
        if self.world_size > 1:
            if self.drop_last:
                usable_batch_count = (len(batches) // self.world_size) * self.world_size
                batches = batches[:usable_batch_count]
            elif batches:
                target_batch_count = math.ceil(len(batches) / self.world_size) * self.world_size
                source_batches = [list(batch) for batch in batches]
                source_index = 0
                while len(batches) < target_batch_count:
                    batches.append(list(source_batches[source_index % len(source_batches)]))
                    source_index += 1
            batches = batches[self.rank :: self.world_size]
        return batches

    def __iter__(self):
        for batch in self._build_batches():
            yield batch

    def __len__(self) -> int:
        total_batches = 0
        for indices in self._grouped_indices.values():
            if self.drop_last:
                total_batches += len(indices) // self.batch_size
            else:
                total_batches += math.ceil(len(indices) / self.batch_size)
        if self.world_size <= 1:
            return total_batches
        if self.drop_last:
            return total_batches // self.world_size
        return math.ceil(total_batches / self.world_size)
