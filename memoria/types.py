from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


@dataclass
class EngramConnection:
    """Data structure for engram connections."""

    #: Source engram ID.
    source_id: int
    #: Target engram ID.
    target_id: int
    #: Connection weight (Probability).
    weight: float


@dataclass
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
    #: fire count of the engram.
    fire_count: int
    #: The outgoing edges of the engram.
    outgoings: List[EngramConnection]
    #: The incoming edges of the engram.
    incoming: List[EngramConnection]


@dataclass
class EngramsInfo:
    """Data structure for engrams information."""

    #: Engram ID to EngramInfo mapping.
    engrams: Dict[int, EngramInfo]
    #: All engram connections mapping from source and target engram IDs.
    edges: Dict[Tuple[int, int], EngramConnection]
    #: Working memory engram IDs.
    working: List[int]
    #: Short-term memory engram IDs.
    shortterm: List[int]
    #: Long-term memory engram IDs.
    longterm: List[int]
