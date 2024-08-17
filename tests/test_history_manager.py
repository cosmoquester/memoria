from memoria.history_manager import HistoryManager
from memoria.types import EngramInfo, EngramsInfo, Firing


def test_history_manager():
    history_manager = HistoryManager()
    assert len(history_manager) == 0
    assert history_manager.timestep == 0
    assert history_manager.engram_ids == []
    assert history_manager.alive_engram_ids == []
    assert history_manager.deleted_engram_ids == []

    engrams = {
        1: EngramInfo(id=1, type="WORKING", lifespan=4, age=0, fire_count=0, outgoings=[], incoming=[]),
        2: EngramInfo(id=2, type="SHORTTERM", lifespan=5, age=0, fire_count=0, outgoings=[], incoming=[]),
        3: EngramInfo(id=3, type="LONGTERM", lifespan=6, age=0, fire_count=0, outgoings=[], incoming=[]),
    }
    history_manager.add_summary(EngramsInfo(engrams=engrams, edges={}, working=[1], shortterm=[2], longterm=[3]))
    assert len(history_manager) == 1
    assert history_manager.timestep == 1
    assert history_manager.engram_ids == [1, 2, 3]
    assert history_manager.alive_engram_ids == [1, 2, 3]
    assert history_manager.deleted_engram_ids == []

    engrams = {
        2: EngramInfo(id=2, type="SHORTTERM", lifespan=4, age=0, fire_count=0, outgoings=[], incoming=[]),
        3: EngramInfo(id=3, type="LONGTERM", lifespan=8, age=0, fire_count=0, outgoings=[], incoming=[]),
        4: EngramInfo(id=4, type="LONGTERM", lifespan=6, age=0, fire_count=0, outgoings=[], incoming=[]),
    }
    history_manager.add_summary(EngramsInfo(engrams=engrams, edges={}, working=[], shortterm=[2], longterm=[3, 4]))
    assert len(history_manager) == 2
    assert history_manager.timestep == 2
    assert history_manager.engram_ids == [1, 2, 3, 4]
    assert history_manager.alive_engram_ids == [2, 3, 4]
    assert history_manager.deleted_engram_ids == [1]
    assert history_manager.engram_firing_times == {3: [1]}
    assert history_manager.engram_firings == {3: [Firing(timestep=1, engram_id=3, lifespan_gain=3.0)]}

    engram_history = history_manager.inspect(1)
    assert engram_history.id == 1
    assert engram_history.creation_timestep == 0
    assert engram_history.deletion_timestep == 1
    assert engram_history.duration == 1
    assert engram_history.firing_times == []
    assert engram_history.firings == []
