from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(slots=True, frozen=True)
class EngramConnection:
    """Data structure for engram connections."""

    #: Source engram ID.
    source_id: int
    #: Target engram ID.
    target_id: int
    #: Connection weight (Probability).
    weight: float
    #: Cofire count.
    cofire_count: int


@dataclass(slots=True, frozen=True)
class EngramInfo:
    """Data structure for engram information."""

    #: Engram ID.
    id: int
    #: Engram type.
    type: Literal["WORKING", "SHORTTERM", "LONGTERM"]
    #: Lifetime of the engram.
    lifespan: int
    #: The age of the engram.
    age: Optional[int]
    #: Fire count of the engram.
    fire_count: int
    #: The outgoing edges of the engram.
    outgoings: Tuple[EngramConnection]
    #: The incoming edges of the engram.
    incoming: Tuple[EngramConnection]

    @property
    def cofire_counts(self) -> dict[int, int]:
        """Get the cofire counts of the engram."""
        return {edge.target_id: edge.cofire_count for edge in self.outgoings}


@dataclass(slots=True, frozen=True)
class EngramsInfo:
    """Data structure for engrams information."""

    #: Engram ID to EngramInfo mapping.
    engrams: dict[int, EngramInfo]
    #: All engram connections mapping from source and target engram IDs.
    edges: dict[Tuple[int, int], EngramConnection]
    #: Working memory engram IDs.
    working: Tuple[int]
    #: Short-term memory engram IDs.
    shortterm: Tuple[int]
    #: Long-term memory engram IDs.
    longterm: Tuple[int]


@dataclass(slots=True, frozen=True)
class Firing:
    """Data structure for firing information."""

    #: Firing timestep.
    timestep: int
    #: Engram ID.
    engram_id: int
    #: Lifespan Gain.
    lifespan_gain: float


@dataclass(slots=True, frozen=True)
class EngramHistory:
    """Historical information of an engram."""

    #: Engram ID.
    id: int
    #: Creation time of the engram.
    creation_timestep: int
    #: Deletion time of the engram.
    deletion_timestep: Optional[int]
    #: Duration of the engram.
    duration: Optional[int]
    #: Firing times of the engram.
    firing_times: list[int]
    #: Firing information of the engram.
    firings: list[Firing]
    #: Summaries of the engram.
    summaries: list[EngramsInfo]
