from collections import defaultdict
from typing import Dict, List

from .types import EngramHistory, EngramsInfo, Firing


class HistoryManager:
    """Managing History of engram summaries.

    Attributes:
        timestep: Current timestep.
        summaries: List of engram summaries.
        engram_creation_times: Dictionary of engram creation times.
        engram_deletion_times: Dictionary of engram deletion times.
        engram_durations: Dictionary of engram durations.
        engram_firing_times: Dictionary of engram firing times.
        engram_firings: Dictionary of engram firings.
        engram_ids: List of engram ids.
        alive_engram_ids: List of alive engram ids.
        deleted_engram_ids: List of deleted engram ids.
    """

    def __init__(self):
        self.summaries: List[EngramsInfo] = []
        self.engram_creation_times: Dict[int, int] = {}
        self.engram_deletion_times: Dict[int, int] = {}
        self.engram_durations: Dict[int, int] = {}
        self.engram_firing_times: Dict[int, List[int]] = defaultdict(list)
        self.engram_firings: Dict[int, List[Firing]] = defaultdict(list)

        self.alive_engram_ids: List[int] = []
        self.deleted_engram_ids: List[int] = []

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, index: int):
        return self.summaries[index]

    @property
    def timestep(self):
        return len(self)

    @property
    def engram_ids(self):
        return list(self.engram_creation_times.keys())

    def add_summary(self, summary: EngramsInfo):
        for engram_id, engram in summary.engrams.items():
            if engram_id not in self.alive_engram_ids:
                self.engram_creation_times[engram_id] = self.timestep
                self.alive_engram_ids.append(engram_id)
            elif (
                engram_id in self.summaries[-1].engrams
                and engram.lifespan > self.summaries[-1].engrams[engram_id].lifespan
            ):
                self.engram_firing_times[engram_id].append(self.timestep)
                self.engram_firings[engram_id].append(
                    Firing(
                        timestep=self.timestep,
                        engram_id=engram_id,
                        lifespan_gain=engram.lifespan - self.summaries[-1].engrams[engram_id].lifespan + 1.0,
                    )
                )

        for engram_id in self.alive_engram_ids:
            if engram_id not in summary.engrams:
                self.engram_deletion_times[engram_id] = self.timestep
                self.deleted_engram_ids.append(engram_id)
                self.engram_durations[engram_id] = self.timestep - self.engram_creation_times[engram_id]
        self.alive_engram_ids = list(summary.engrams.keys())

        self.summaries.append(summary)

    def inspect(self, engram_id: int) -> EngramHistory:
        """Inspect the history of an engram.

        Args:
            engram_id: Engram ID to inspect.
        Returns:
            EngramHistory: Historical information of the engram.
        """
        creation_time = self.engram_creation_times[engram_id]
        deletion_time = self.engram_deletion_times.get(engram_id)
        duration = self.engram_durations.get(engram_id)
        firing_times = self.engram_firing_times.get(engram_id, [])
        firings = self.engram_firings.get(engram_id, [])
        related_summaries = self.summaries[creation_time:deletion_time]

        return EngramHistory(
            id=engram_id,
            creation_timestep=creation_time,
            deletion_timestep=deletion_time,
            duration=duration,
            firing_times=firing_times,
            firings=firings,
            summaries=related_summaries,
        )
