import gzip
import pickle
from collections import defaultdict

from .types import EngramHistory, EngramInfo, EngramsInfo, Firing


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
        firings_per_time: List of firings per timestep.
        engram_fire_counts: Dictionary of engram fire counts.
        engram_ids: List of engram ids.
        alive_engram_ids: List of alive engram ids.
        deleted_engram_ids: List of deleted engram ids.
    """

    def __init__(self):
        self.summaries: list[EngramsInfo] = []
        self.engram_creation_times: dict[int, int] = {}
        self.engram_deletion_times: dict[int, int] = {}
        self.engram_durations: dict[int, int] = {}
        self.engram_firing_times: dict[int, list[int]] = defaultdict(list)
        self.engram_firings: dict[int, list[Firing]] = defaultdict(list)
        self.firings_per_time: list[list[Firing]] = []

        self.alive_engram_ids: list[int] = []
        self.deleted_engram_ids: list[int] = []

    def __len__(self) -> int:
        return len(self.summaries)

    def __getitem__(self, index: int) -> EngramsInfo:
        return self.summaries[index]

    def save(self, path: str) -> None:
        """Save the history manager to a compressed data file.

        Args:
            path: Path to save the history manager.
        """
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "HistoryManager":
        """Load the history manager from a compressed data file.

        Args:
            path: Path to load the history manager.
        Returns:
            HistoryManager: Loaded history manager.
        """
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    @property
    def timestep(self) -> int:
        """Get the current timestep."""
        return len(self)

    @property
    def engram_ids(self) -> list[int]:
        """Get the list of engram IDs."""
        return list(self.engram_creation_times.keys())

    @property
    def engram_fire_counts(self) -> dict[int, int]:
        """Get the fire counts of the engrams."""
        return {engram_id: len(firings) for engram_id, firings in self.engram_firings.items()}

    @property
    def engram_lastest_alive_timestep(self) -> dict[int, int]:
        """Get the latest alive timestep of the engrams."""
        return {
            engram_id: creation_time + self.engram_durations[engram_id] - 1
            for engram_id, creation_time in self.engram_creation_times.items()
        }

    @property
    def latest_engram_infos(self) -> dict[int, EngramInfo]:
        """Get the latest engram information before dying."""
        last_timestep = self.engram_lastest_alive_timestep
        return {engram_id: self.summaries[last_timestep[engram_id]].engrams[engram_id] for engram_id in self.engram_ids}

    def add_summary(self, summary: EngramsInfo) -> None:
        firings = []
        for engram_id, engram in summary.engrams.items():
            if engram_id not in self.alive_engram_ids:
                self.engram_creation_times[engram_id] = self.timestep
                self.alive_engram_ids.append(engram_id)
            elif (
                engram_id in self.summaries[-1].engrams
                and engram.fire_count > self.summaries[-1].engrams[engram_id].fire_count
            ):
                self.engram_firing_times[engram_id].append(self.timestep)
                firing = Firing(
                    timestep=self.timestep,
                    engram_id=engram_id,
                    lifespan_gain=engram.lifespan - self.summaries[-1].engrams[engram_id].lifespan + 1.0,
                )
                self.engram_firings[engram_id].append(firing)
                firings.append(firing)
        self.firings_per_time.append(firings)

        for engram_id in self.alive_engram_ids:
            if engram_id not in summary.engrams:
                self.engram_deletion_times[engram_id] = self.timestep
                self.deleted_engram_ids.append(engram_id)
        self.alive_engram_ids = list(summary.engrams.keys())
        for engram_id in self.alive_engram_ids:
            self.engram_durations[engram_id] = self.timestep - self.engram_creation_times[engram_id] + 1

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
